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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to raw ImageNet directory (e.g. /path/to/imagenet)")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save latents")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16) # 增加 worker 数以加速 IO
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # 1. 准备 VAE
    print(f"Loading VAE to {args.device}...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(args.device)
    vae.eval()

    # 2. 准备数据增强
    transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 3. 递归扫描所有文件 (使用 os.walk 解决目录嵌套问题)
    print(f"Scanning files in {args.data_path}...")
    image_paths = []
    
    # 支持的图片扩展名
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    
    for root, dirs, files in os.walk(args.data_path):
        for fname in files:
            if fname.lower().endswith(valid_exts):
                src_path = os.path.join(root, fname)
                
                # 计算相对路径，用于保持目录结构
                # 例如 src: /data/imagenet/train/n0144/img.jpg -> rel: train/n0144/img.jpg
                rel_path = os.path.relpath(src_path, args.data_path)
                
                # 构造目标路径: /save/imagenet_latent/train/n0144/img.pt
                dst_path = os.path.join(args.save_path, os.path.splitext(rel_path)[0] + ".pt")
                
                image_paths.append((src_path, dst_path))

    print(f"Total images found: {len(image_paths)}")
    if len(image_paths) == 0:
        print("Error: No images found! Please check data_path.")
        return

    # 4. 数据集定义
    class ImageListDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform
        
        def __len__(self):
            return len(self.paths)
        
        def __getitem__(self, idx):
            src, dst = self.paths[idx]
            try:
                img = Image.open(src).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, dst
            except Exception as e:
                print(f"Error loading {src}: {e}")
                # 返回一个 dummy 数据防止崩溃 (实际训练中最好清理坏图)
                return torch.zeros(3, args.image_size, args.image_size), dst

    dataset = ImageListDataset(image_paths, transform)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 5. 开始编码与保存
    print("Start encoding...")
    # 提前创建所有需要的父目录，避免多线程冲突 (虽然这里是单线程创建)
    # 为了效率，我们在保存循环中动态创建
    
    with torch.no_grad():
        for batch_imgs, batch_dsts in tqdm(loader):
            batch_imgs = batch_imgs.to(args.device)
            
            # VAE Encode -> Dist -> Sample
            dist = vae.encode(batch_imgs).latent_dist
            latents = dist.sample() # (B, 4, 32, 32)
            
            # 转移回 CPU
            latents = latents.cpu()
            
            # 保存
            for i, dst in enumerate(batch_dsts):
                # 确保目录存在
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                # 保存 Latent Tensor
                torch.save(latents[i].clone(), dst)

    print(f"Done! Latents saved to {args.save_path}")

if __name__ == "__main__":
    main()