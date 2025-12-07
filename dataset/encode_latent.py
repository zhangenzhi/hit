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

def main():
    parser = argparse.ArgumentParser(description="Encode ImageNet to Latents with Data Augmentation (DDP Support)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to raw ImageNet train directory")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save latents")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8, help="Number of *original images* per batch per GPU.")
    parser.add_argument("--num_crops", type=int, default=10, help="Number of RandomResizedCrop augmentations per image")
    parser.add_argument("--num_workers", type=int, default=8, help="Workers per GPU")
    args = parser.parse_args()

    # --- 1. DDP 初始化 ---
    # 检查是否通过 torchrun 启动
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_main_process = (rank == 0)
    else:
        # 单卡回退模式
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main_process = True
        print("Running in single-process mode (Not DDP).")

    if is_main_process:
        print(f"Global Rank 0: Initialized DDP with world_size={world_size}")

    # --- 2. 准备 VAE ---
    if is_main_process:
        print(f"Loading VAE to {device}...")
    
    # 开启 tf32 以在 Ampere/Hopper 架构上加速 FP32 计算
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
    
    # 注意：在拥有数百万文件的 ImageNet 上，os.walk 可能较慢。
    # 理想情况下只在 rank 0 扫描然后广播，但为了代码简单且文件系统通常能扛住 4 个并发读取，这里每个进程独立扫描。
    # 关键：必须排序，确保所有进程看到的文件列表顺序完全一致！
    for root, _, files in os.walk(args.data_path):
        for fname in files:
            if fname.lower().endswith(valid_exts):
                src_path = os.path.join(root, fname)
                rel_path = os.path.relpath(src_path, args.data_path)
                
                base_name = os.path.splitext(rel_path)[0]
                dst_base = os.path.join(args.save_path, base_name)
                
                image_paths.append((src_path, dst_base))
    
    # 【核心】排序以保证确定性
    image_paths.sort(key=lambda x: x[0])

    if is_main_process:
        print(f"Found {len(image_paths)} total images.")
    
    # 【核心】根据 rank 进行切片分发
    # 比如 4 卡：Rank 0 处理 0, 4, 8...; Rank 1 处理 1, 5, 9...
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
        print("Start encoding with DDP...")
    
    # 只有 Rank 0 显示进度条，避免刷屏
    iterator = tqdm(loader, desc=f"GPU {rank}", disable=not is_main_process)
    
    with torch.no_grad():
        for batch_imgs, batch_dst_bases in iterator:
            # batch_imgs: (B, num_crops, 3, H, W)
            B, N, C, H, W = batch_imgs.shape
            
            flat_imgs = batch_imgs.view(-1, C, H, W).to(device)
            
            # 自动混合精度 (H100 强烈推荐)
            with torch.amp.autocast('cuda', dtype=torch.float16): # 或 bfloat16
                try:
                    dist_post = vae.encode(flat_imgs).latent_dist
                    latents = dist_post.sample()
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"Rank {rank} OOM, retrying with chunks...")
                        torch.cuda.empty_cache()
                        latents_list = []
                        chunk_size = 4 # 减小 chunk
                        for i in range(0, flat_imgs.shape[0], chunk_size):
                            chunk = flat_imgs[i:i+chunk_size]
                            dist_chunk = vae.encode(chunk).latent_dist
                            latents_list.append(dist_chunk.sample())
                        latents = torch.cat(latents_list, dim=0)
                    else:
                        raise e

            latents = latents.cpu()
            latents = latents.view(B, N, *latents.shape[1:])
            
            for i, dst_base in enumerate(batch_dst_bases):
                try:
                    os.makedirs(os.path.dirname(dst_base), exist_ok=True)
                    for j in range(N):
                        save_path = f"{dst_base}_{j}.pt"
                        # 检查文件是否已存在（可选，防止重复工作，但在 DDP 切片模式下理论上不会冲突）
                        # if not os.path.exists(save_path):
                        torch.save(latents[i, j].clone(), save_path)
                except OSError as e:
                    # 处理多进程创建目录可能的竞争条件 (虽然 makedirs exist_ok=True 通常是安全的)
                    pass

    if is_main_process:
        print(f"Done! Latents saved to {args.save_path}")
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()