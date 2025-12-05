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
    parser.add_argument("--data_path", type=str, required=True, help="Path to raw ImageNet train directory (e.g. /path/to/imagenet/train)")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save latents (e.g. /path/to/imagenet_latents/train)")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # 1. 准备 VAE
    print(f"Loading VAE to {args.device}...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(args.device)
    vae.eval()

    # 2. 准备数据增强 (必须与训练时一致: Resize + CenterCrop)
    transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 3. 遍历文件夹结构
    # 假设 ImageNet 结构: root/class_xxx/img_xxx.JPEG
    classes = sorted([d for d in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, d))])
    print(f"Found {len(classes)} classes.")

    # 收集所有图片路径
    image_paths = []
    for cls in classes:
        cls_dir = os.path.join(args.data_path, cls)
        target_dir = os.path.join(args.save_path, cls)
        os.makedirs(target_dir, exist_ok=True)
        
        fnames = sorted(os.listdir(cls_dir))
        for fname in fnames:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                src = os.path.join(cls_dir, fname)
                # 保存为 .pt，文件名保持一致
                dst = os.path.join(target_dir, os.path.splitext(fname)[0] + ".pt")
                image_paths.append((src, dst))

    print(f"Total images to process: {len(image_paths)}")

    # 简单的数据集类
    class ImageListDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform
        
        def __len__(self):
            return len(self.paths)
        
        def __getitem__(self, idx):
            src, dst = self.paths[idx]
            img = Image.open(src).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, dst

    dataset = ImageListDataset(image_paths, transform)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 4. 开始编码循环
    print("Start encoding...")
    with torch.no_grad():
        for batch_imgs, batch_dsts in tqdm(loader):
            batch_imgs = batch_imgs.to(args.device)
            
            # VAE Encode -> Dist -> Sample
            # 注意：我们这里保存的是未缩放 (Unscaled) 的 Sample
            # Trainer 中加载时会乘以 0.18215
            dist = vae.encode(batch_imgs).latent_dist
            latents = dist.sample() # (B, 4, 32, 32)
            
            # 转移回 CPU 并保存
            latents = latents.cpu()
            for i, dst in enumerate(batch_dsts):
                # 每個样本单独保存，保持目录结构，方便 Dataset 加载
                torch.save(latents[i].clone(), dst)

    print(f"Done! Latents saved to {args.save_path}")

if __name__ == "__main__":
    main()