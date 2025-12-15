import argparse
import os
import torch
import numpy as np
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from diffusers.models import AutoencoderKL

# --- 尝试导入 pynvml 用于监控 GPU 利用率 ---
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

def get_gpu_stats(local_rank, nvml_handle=None):
    """获取当前 GPU 的显存和利用率信息"""
    stats = {}
    mem_reserved = torch.cuda.memory_reserved() / 1024**3
    mem_allocated = torch.cuda.memory_allocated() / 1024**3
    stats['mem'] = f"{mem_allocated:.1f}/{mem_reserved:.1f}GB"
    
    if nvml_handle:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
            stats['util'] = f"{util.gpu}%"
        except Exception:
            stats['util'] = "N/A"
    return stats

def main():
    parser = argparse.ArgumentParser(description="Encode ImageNet to Latents (Pre-calc for DiT)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to raw ImageNet train directory")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save latents")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32, help="Original images per batch per GPU") 
    parser.add_argument("--num_crops", type=int, default=10, help="Number of crops per image (only for random crop)")
    parser.add_argument("--num_workers", type=int, default=16, help="Workers per GPU")
    
    # [新增] 关键参数：控制验证集生成模式
    parser.add_argument("--center_crop", action="store_true", help="Use CenterCrop (for Validation Set). Sets num_crops=1.")
    
    args = parser.parse_args()

    # --- 1. DDP 初始化 ---
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_main_process = (rank == 0)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main_process = True

    if is_main_process:
        print(f"[Init] World Size: {world_size} | Device: {device}")

    # --- 监控初始化 ---
    nvml_handle = None
    if is_main_process and HAS_PYNVML:
        try:
            pynvml.nvmlInit()
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
        except Exception:
            pass

    # --- 2. 准备 VAE & 精度设置 ---
    if is_main_process:
        print(f"Loading VAE...")
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if is_main_process:
        print(f"Using precision: {dtype}")

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    
    if is_main_process:
        print("Compiling VAE...")
    try:
        vae = torch.compile(vae, mode="max-autotune")
    except Exception as e:
        if is_main_process: print(f"Warning: torch.compile failed, using eager mode. {e}")

    # --- 3. 准备数据增强 (区分 训练/验证) ---
    if args.center_crop:
        if is_main_process: print("Mode: Validation (Center Crop, 1 Crop/Img)")
        # [新增] 验证集标准预处理
        transform = transforms.Compose([
            transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        args.num_crops = 1 # 验证集只需要一张
    else:
        if is_main_process: print(f"Mode: Training (Random Crop, {args.num_crops} Crops/Img)")
        # [原有] 训练集增强
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size, scale=(0.25, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    # --- 4. 扫描文件 ---
    if is_main_process:
        print(f"Scanning files in {args.data_path}...")
    
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    image_paths = []
    
    for root, _, files in os.walk(args.data_path):
        for fname in files:
            if fname.lower().endswith(valid_exts):
                src_path = os.path.join(root, fname)
                rel_path = os.path.relpath(src_path, args.data_path)
                base_name = os.path.splitext(rel_path)[0]
                dst_base = os.path.join(args.save_path, base_name)
                image_paths.append((src_path, dst_base))
    
    image_paths.sort(key=lambda x: x[0])
    my_paths = image_paths[rank::world_size]
    
    if is_main_process:
        print(f"Total images: {len(image_paths)}. Each GPU processing: {len(my_paths)}")

    # --- 5. Dataset ---
    class AugmentedDataset(Dataset):
        def __init__(self, paths, transform, num_crops):
            self.paths = paths
            self.transform = transform
            self.num_crops = num_crops
        
        def __len__(self):
            return len(self.paths)
        
        def __getitem__(self, idx):
            src, dst_base = self.paths[idx]
            try:
                img = Image.open(src).convert("RGB")
                crops = []
                for _ in range(self.num_crops):
                    crops.append(self.transform(img))
                # shape: [num_crops, 3, H, W]
                img_tensor = torch.stack(crops)
                return img_tensor, dst_base
            except Exception as e:
                # 打印完整错误路径，方便排查坏图
                print(f"[Rank {rank}] Error loading {src}: {e}")
                # 返回全黑图，但通常建议在这里记录日志以便后续删除坏图
                return torch.zeros(self.num_crops, 3, args.image_size, args.image_size), dst_base

    dataset = AugmentedDataset(my_paths, transform, args.num_crops)
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2, 
        persistent_workers=True
    )

    # --- 6. 编码循环 ---
    if is_main_process:
        print(f"Start encoding... Outputting Unscaled Latents (std≈5.5).")
    
    iterator = tqdm(loader, desc=f"GPU {rank}", disable=not is_main_process)
    
    with torch.no_grad():
        for i, (batch_imgs, batch_dst_bases) in enumerate(iterator):
            # batch_imgs shape: [B, N, 3, H, W]
            B, N, C, H, W = batch_imgs.shape
            
            flat_imgs = batch_imgs.view(-1, C, H, W).to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', dtype=dtype):
                # 编码: 使用 mode() 取均值
                dist_post = vae.encode(flat_imgs).latent_dist
                latents = dist_post.mode() 
            
            # 移回 CPU: [B, N, 4, h, w]
            latents = latents.cpu()
            latents = latents.view(B, N, *latents.shape[1:])
            
            # 保存循环
            for k, dst_base in enumerate(batch_dst_bases):
                try:
                    save_dir = os.path.dirname(dst_base)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    save_path = f"{dst_base}.pt"
                    # 如果是 Center Crop，latents[k] 是 [1, 4, 32, 32]，保存即可
                    # 如果是 Random Crop，latents[k] 是 [10, 4, 32, 32]
                    torch.save(latents[k].clone(), save_path)
                    
                except OSError as e:
                    print(f"Error saving {dst_base}: {e}")
            
            if is_main_process and i % 10 == 0:
                stats = get_gpu_stats(local_rank, nvml_handle)
                postfix_str = f"Mem:{stats.get('mem')}"
                if 'util' in stats:
                    postfix_str += f" | Util:{stats.get('util')}"
                iterator.set_postfix_str(postfix_str)

    if is_main_process:
        print(f"Done! Latents saved to {args.save_path}")
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()