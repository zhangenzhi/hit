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
    # 建议: batch_size 可以设大一点，VAE 推理不怎么吃显存
    parser.add_argument("--batch_size", type=int, default=32, help="Original images per batch per GPU") 
    parser.add_argument("--num_crops", type=int, default=10, help="Number of crops per image")
    parser.add_argument("--num_workers", type=int, default=16, help="Workers per GPU")
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
    
    # 开启 TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # 自动选择精度: H100/A100 优先使用 BFloat16 以防溢出，老卡使用 Float16
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if is_main_process:
        print(f"Using precision: {dtype}")

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    
    # 编译优化
    # 注意: 如果遇到 RuntimeError，可以尝试注释掉这一行，或改为 mode='reduce-overhead'
    if is_main_process:
        print("Compiling VAE...")
    try:
        vae = torch.compile(vae, mode="max-autotune")
    except Exception as e:
        if is_main_process: print(f"Warning: torch.compile failed, using eager mode. {e}")

    # --- 3. 准备数据增强 ---
    crop_transform = transforms.Compose([
        # [修改] 将 scale 下限从 0.08 提高到 0.25。
        # 对于生成任务，太小的 crop 会导致 latent 只有纹理没有语义，影响 DiT 学习。
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
    
    # 使用 os.scandir 通常比 os.walk 更快一点
    for root, _, files in os.walk(args.data_path):
        for fname in files:
            if fname.lower().endswith(valid_exts):
                src_path = os.path.join(root, fname)
                # 计算相对路径，用于保持输出目录结构
                rel_path = os.path.relpath(src_path, args.data_path)
                base_name = os.path.splitext(rel_path)[0]
                dst_base = os.path.join(args.save_path, base_name)
                image_paths.append((src_path, dst_base))
    
    # 排序确保所有 GPU 拿到的列表顺序一致
    image_paths.sort(key=lambda x: x[0])

    # 数据分片
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
                print(f"[Rank {rank}] Error loading {src}: {e}")
                # 返回全黑图作为 fallback
                return torch.zeros(self.num_crops, 3, args.image_size, args.image_size), dst_base

    dataset = AugmentedDataset(my_paths, crop_transform, args.num_crops)
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2, # 根据 CPU 性能调整
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
            
            # Flatten 这里的维度，变成 [B*N, 3, H, W] 喂给 VAE
            flat_imgs = batch_imgs.view(-1, C, H, W).to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', dtype=dtype):
                # 编码
                dist_post = vae.encode(flat_imgs).latent_dist
                
                # [核心修改 1] 使用 mode() 取均值，不要 sample()
                # 因为我们要离线保存，sample() 会导致噪声被“冻结”，这是错误的。
                latents = dist_post.mode() 
                
                # [核心修改 2] 不乘 0.18215
                # 根据你的需求，我们在训练时的 Loader 里进行缩放。
                # 此时 latents 的 std 约为 5.5
            
            # 移回 CPU 并恢复维度: [B, N, 4, h, w]
            latents = latents.cpu()
            latents = latents.view(B, N, *latents.shape[1:])
            
            # 保存循环
            for k, dst_base in enumerate(batch_dst_bases):
                try:
                    save_dir = os.path.dirname(dst_base)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # [核心修改 3] 将一张图的所有 crops 存为一个文件
                    # 文件路径: .../n01440764/n01440764_10026.pt
                    # Tensor Shape: [10, 4, 32, 32]
                    # 避免生成千万级小文件导致 Inode 耗尽
                    save_path = f"{dst_base}.pt"
                    torch.save(latents[k].clone(), save_path)
                    
                except OSError as e:
                    print(f"Error saving {dst_base}: {e}")
            
            # 打印监控日志
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