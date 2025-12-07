import argparse
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from diffusers.models import AutoencoderKL

def main():
    parser = argparse.ArgumentParser(description="Encode ImageNet to Latents with Data Augmentation")
    parser.add_argument("--data_path", type=str, required=True, help="Path to raw ImageNet train directory")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save latents")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8, help="Number of *original images* per batch. Real batch size will be batch_size * num_crops")
    parser.add_argument("--num_crops", type=int, default=10, help="Number of RandomResizedCrop augmentations per image")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # 1. 准备 VAE
    print(f"Loading VAE to {args.device}...")
    # 使用 float16 加速推理，如果显存足够可以用 float32
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(args.device)
    vae.eval()
    
    # 2. 准备数据增强 (DiT 论文核心: RandomResizedCrop)
    # 我们需要对同一张图应用多次这个 Transform
    crop_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 3. 扫描文件
    print(f"Scanning files in {args.data_path}...")
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    image_paths = []
    
    for root, _, files in os.walk(args.data_path):
        for fname in files:
            if fname.lower().endswith(valid_exts):
                src_path = os.path.join(root, fname)
                rel_path = os.path.relpath(src_path, args.data_path)
                
                # 构造基础保存路径 (去除扩展名)
                # 例如: train/n0144/img.jpg -> save_path/train/n0144/img
                base_name = os.path.splitext(rel_path)[0]
                dst_base = os.path.join(args.save_path, base_name)
                
                image_paths.append((src_path, dst_base))

    print(f"Found {len(image_paths)} original images.")
    print(f"Expected generated latents: {len(image_paths) * args.num_crops}")

    # 4. 数据集定义 (Multi-Crop Logic)
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
                # 生成 num_crops 个增强版本
                crops = []
                for _ in range(self.num_crops):
                    crops.append(self.transform(img))
                
                # Stack: (num_crops, 3, H, W)
                img_tensor = torch.stack(crops)
                return img_tensor, dst_base
            except Exception as e:
                print(f"Error loading {src}: {e}")
                # 返回零张量占位
                return torch.zeros(self.num_crops, 3, args.image_size, args.image_size), dst_base

    dataset = AugmentedDataset(image_paths, crop_transform, args.num_crops)
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, # 这里 batch_size 指的是原始图片数量
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # 5. 编码循环
    print("Start encoding with augmentation...")
    
    with torch.no_grad():
        for batch_imgs, batch_dst_bases in tqdm(loader):
            # batch_imgs shape: (B, num_crops, 3, H, W)
            B, N, C, H, W = batch_imgs.shape
            
            # Flatten to (B*N, 3, H, W) for efficient VAE inference
            flat_imgs = batch_imgs.view(-1, C, H, W).to(args.device)
            
            # VAE Encode
            # 注意: 如果 B*N 很大导致显存不足，可以进一步 chunk 处理
            try:
                dist = vae.encode(flat_imgs).latent_dist
                latents = dist.sample() # (B*N, 4, h, w)
            except RuntimeError as e:
                # 显存不足回退逻辑
                print("OOM, switching to chunked inference...")
                latents_list = []
                chunk_size = 16
                for i in range(0, flat_imgs.shape[0], chunk_size):
                    chunk = flat_imgs[i:i+chunk_size]
                    latents_list.append(vae.encode(chunk).latent_dist.sample())
                latents = torch.cat(latents_list, dim=0)

            # 转移回 CPU
            latents = latents.cpu()
            
            # Reshape back to (B, N, 4, h, w)
            latents = latents.view(B, N, *latents.shape[1:])
            
            # 保存逻辑
            for i, dst_base in enumerate(batch_dst_bases):
                # 确保父目录存在 (每个 batch 创建一次即可，不需要对每个 crop 检查)
                os.makedirs(os.path.dirname(dst_base), exist_ok=True)
                
                # 保存 N 个 Crop
                for j in range(N):
                    # 文件名格式: original_name_{crop_index}.pt
                    # 例如: n01440764_10026_0.pt, n01440764_10026_1.pt ...
                    save_path = f"{dst_base}_{j}.pt"
                    
                    # Clone 出来保存，释放引用
                    torch.save(latents[i, j].clone(), save_path)

    print(f"Done! Latents saved to {args.save_path}")

if __name__ == "__main__":
    main()