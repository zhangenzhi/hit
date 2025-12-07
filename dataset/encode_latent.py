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
    """
    获取当前 GPU 的显存和利用率信息
    """
    stats = {}
    
    # 1. 显存信息 (PyTorch Native, 最准确)
    # reserved: PyTorch 缓存分配器持有的总显存 (包括未使用的碎片) -> 接近 nvidia-smi 显示的值
    # allocated: Tensor 实际占用的显存
    mem_reserved = torch.cuda.memory_reserved() / 1024**3
    mem_allocated = torch.cuda.memory_allocated() / 1024**3
    stats['mem'] = f"{mem_allocated:.1f}/{mem_reserved:.1f}GB"
    
    # 2. 利用率信息 (Optional, via pynvml)
    if nvml_handle:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
            stats['util'] = f"{util.gpu}%"
        except Exception:
            stats['util'] = "N/A"
            
    return stats

def main():
    parser = argparse.ArgumentParser(description="Encode ImageNet to Latents with Data Augmentation (DDP Support)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to raw ImageNet train directory")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save latents")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64, help="Number of *original images* per batch per GPU.")
    parser.add_argument("--num_crops", type=int, default=32, help="Number of RandomResizedCrop augmentations per image")
    parser.add_argument("--num_workers", type=int, default=8, help="Workers per GPU")
    args = parser.parse_args()

    # --- 1. DDP 初始化 ---
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        # 强制设置当前进程可见的 GPU
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_main_process = (rank == 0)
        
        print(f"[Init] Global Rank {rank} | Local Rank {local_rank} | Device: {torch.cuda.current_device()} | Name: {torch.cuda.get_device_name(device)}")
    else:
        # 单卡模式
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main_process = True
        print(f"[Init] Single Process Mode | Device: {device}")

    if is_main_process:
        print(f"Global Rank 0: Initialized DDP with world_size={world_size}")

    # --- 监控初始化 (仅 Rank 0 需要，因为只有 Rank 0 打印进度条) ---
    nvml_handle = None
    if is_main_process and HAS_PYNVML:
        try:
            pynvml.nvmlInit()
            # 注意: 这里假设 local_rank 直接对应 nvml 的设备索引
            # 在某些复杂的容器配置中可能不一致，但在标准 torchrun 环境下通常是匹配的
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
        except Exception as e:
            print(f"Warning: Failed to initialize NVML for monitoring: {e}")
            nvml_handle = None

    # --- 2. 准备 VAE ---
    if is_main_process:
        print(f"Loading VAE to {device}...")
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    
    # --- 3. 准备数据增强 ---
    crop_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # --- 4. 扫描文件 & 数据分片 ---
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

    if is_main_process:
        print(f"Found {len(image_paths)} total images.")
    
    my_paths = image_paths[rank::world_size]
    
    if is_main_process:
        print(f"Each GPU processing approx {len(my_paths)} images.")

    # --- 5. 数据集定义 ---
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
                img_tensor = torch.stack(crops)
                return img_tensor, dst_base
            except Exception as e:
                print(f"Rank {rank} Error loading {src}: {e}")
                return torch.zeros(self.num_crops, 3, args.image_size, args.image_size), dst_base

    dataset = AugmentedDataset(my_paths, crop_transform, args.num_crops)
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # --- 6. 编码循环 ---
    if is_main_process:
        print(f"Start encoding with DDP... (Batch Size per GPU: {args.batch_size} imgs * {args.num_crops} crops = {args.batch_size * args.num_crops} inputs)")
    
    # 只有 Rank 0 显示进度条，并在这里更新 GPU 状态
    iterator = tqdm(loader, desc=f"GPU {rank}", disable=not is_main_process)
    
    with torch.no_grad():
        for i, (batch_imgs, batch_dst_bases) in enumerate(iterator):
            # batch_imgs: (B, num_crops, 3, H, W)
            B, N, C, H, W = batch_imgs.shape
            
            flat_imgs = batch_imgs.view(-1, C, H, W).to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', dtype=torch.float16):
                dist_post = vae.encode(flat_imgs).latent_dist
                latents = dist_post.sample()
            
            latents = latents.cpu()
            latents = latents.view(B, N, *latents.shape[1:])
            
            for k, dst_base in enumerate(batch_dst_bases):
                try:
                    os.makedirs(os.path.dirname(dst_base), exist_ok=True)
                    for j in range(N):
                        save_path = f"{dst_base}_{j}.pt"
                        torch.save(latents[k, j].clone(), save_path)
                except OSError:
                    pass
            
            # --- 更新监控信息 (每5个batch更新一次以减少开销) ---
            if is_main_process and i % 5 == 0:
                stats = get_gpu_stats(local_rank, nvml_handle)
                # Mem: Allocated / Reserved
                postfix_str = f"Mem:{stats.get('mem')} "
                if 'util' in stats:
                    postfix_str += f"Util:{stats.get('util')}"
                iterator.set_postfix_str(postfix_str)

    if is_main_process:
        print(f"Done! Latents saved to {args.save_path}")
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()