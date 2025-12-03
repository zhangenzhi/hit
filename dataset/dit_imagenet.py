import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler

def build_dit_transform(is_train, img_size):
    """
    构建适合 DiT (Diffusion Transformer) 训练的数据增强。
    
    策略：
    - 归一化: 映射到 [-1, 1]，即 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    - 训练: Resize(img_size) -> CenterCrop(img_size) -> RandomHorizontalFlip
      (注: DiT 原文通常不使用 RandomResizedCrop，以保持生成图像的结构一致性)
    - 验证: Resize(img_size) -> CenterCrop(img_size)
    """
    # DiT/Diffusion Model 标准归一化: map [0, 1] to [-1, 1]
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    if is_train:
        transform = transforms.Compose([
            # Resize shortest edge to img_size, maintaining aspect ratio
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    return transform

def build_dit_dataloaders(args):
    """
    DiT 专用 DataLoader 构建函数。
    """
    # 获取参数
    data_dir = getattr(args, 'data_path', getattr(args, 'data_dir', None))
    if data_dir is None:
        raise ValueError("args must contain 'data_path' or 'data_dir'")

    # --- 兼容性处理逻辑 ---
    # 尝试获取 model 和 training 字典 (来自 yaml config)
    model_conf = getattr(args, 'model', {})
    training_conf = getattr(args, 'training', {})
    
    # 1. 获取 img_size (DiT 通常是 256 或 512)
    if isinstance(model_conf, dict) and 'image_size' in model_conf:
        img_size = model_conf['image_size']
    elif isinstance(model_conf, dict) and 'img_size' in model_conf:
        img_size = model_conf['img_size']
    else:
        img_size = getattr(args, 'image_size', getattr(args, 'img_size', 256))

    # 2. 获取 batch_size
    if isinstance(training_conf, dict) and 'batch_size' in training_conf:
        batch_size = training_conf['batch_size']
    else:
        batch_size = getattr(args, 'batch_size', 32)

    num_workers = getattr(args, 'num_workers', 4)

    rank = int(os.environ.get('RANK', 0)) # 安全获取 rank 用于打印
    if rank == 0:
        print(f"构建 DiT Dataloaders | Img Size: {img_size} | Batch Size: {batch_size} | Workers: {num_workers}")

    # 1. 构建 Transforms
    train_transform = build_dit_transform(is_train=True, img_size=img_size)
    # DiT 训练通常不需要验证集上的 transform 差异太大，且为了计算 FID 通常需要固定预处理
    val_transform = build_dit_transform(is_train=False, img_size=img_size)

    # 2. 构建 Datasets
    train_root = os.path.join(data_dir, 'train')
    val_root = os.path.join(data_dir, 'val')
    
    # 检查路径是否存在，如果不存在 val 文件夹，为了防止报错，可能只返回 train
    has_val = os.path.exists(val_root)

    train_dataset = datasets.ImageFolder(train_root, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_root, transform=val_transform) if has_val else None

    # 3. 分布式采样器 (DistributedSampler)
    use_ddp = torch.distributed.is_initialized()
    
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if has_val else None
    else:
        train_sampler = None
        val_sampler = None

    # 4. 构建 DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            sampler=val_sampler,
            drop_last=False,
            persistent_workers=True if num_workers > 0 else False
        )

    return {'train': train_loader, 'val': val_loader}

# --- 测试代码块 ---
if __name__ == '__main__':
    import time
    import argparse
    import numpy as np
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description='DiT Dataloader Test')
    # 兼容之前的参数命名习惯
    parser.add_argument('--data_dir', type=str, default="/work/c30636/dataset/imagenet/", help='Path to the ImageNet dataset.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--image_size', type=int, default=256, help='Image size.')
    parser.add_argument('--num_workers', type=int, default=4, help='Workers.')
    parser.add_argument('--visualize', action='store_true', help='Visualize a batch.')
    args = parser.parse_args()

    if os.path.exists(args.data_dir):
        print(f"\n--- 开始 DiT DataLoader 测试 ---")
        
        loaders = build_dit_dataloaders(args)
        train_loader = loaders['train']
        
        if args.visualize:
            print("Generating visualization...")
            
            def denormalize_dit(tensor):
                # DiT [-1, 1] -> [0, 1]
                return (tensor / 2 + 0.5).clamp(0, 1)

            # 获取一个 batch
            batch_imgs, batch_labels = next(iter(train_loader))
            print(f"Batch Shape: {batch_imgs.shape}, Labels Shape: {batch_labels.shape}")
            
            # 取第一张图
            img = denormalize_dit(batch_imgs[0]).permute(1, 2, 0).numpy()
            
            plt.figure(figsize=(5, 5))
            plt.imshow(img)
            plt.title(f"Label: {batch_labels[0].item()}")
            plt.axis('off')
            plt.savefig("dit_dataloader_viz.png")
            print("Visualization saved to 'dit_dataloader_viz.png'")
        
        # 速度测试
        print("\n--- Speed Test (Train Loader) ---")
        start = time.time()
        count = 0
        max_steps = 20
        
        for i, (imgs, labels) in enumerate(train_loader):
            count += imgs.shape[0]
            if i % 5 == 0:
                print(f"Step {i}: Loaded batch shape {imgs.shape}")
            if i >= max_steps:
                break
                
        duration = time.time() - start
        print(f"Processed {count} images in {duration:.2f}s")
        print(f"Throughput: {count/duration:.2f} img/s")

    else:
        print(f"Error: Data directory not found at {args.data_dir}")