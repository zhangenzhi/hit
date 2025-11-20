import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler

def build_hit_transform(is_train, img_size):
    """
    构建适合 HiT/MAE 预训练的数据增强。
    策略：
    - 训练：RandomResizedCrop (0.2-1.0) + HorizontalFlip + Normalize
    - 验证：Resize + CenterCrop + Normalize
    注意：不使用 ColorJitter，以保持像素分布的真实性，利于生成任务。
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224), interpolation=transforms.InterpolationMode.BICUBIC), # 保持纵横比 Resize
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    return transform

def build_hit_dataloaders(args):
    """
    HiT 专用 DataLoader 构建函数。
    兼容 args 包含 config 字典 (args.model['img_size']) 或 直接属性 (args.img_size) 的情况。
    """
    # 获取参数
    data_dir = args.data_dir
    
    # --- 兼容性处理逻辑 ---
    # 尝试获取 model 和 training 字典 (来自 yaml config)
    model_conf = getattr(args, 'model', {})
    training_conf = getattr(args, 'training', {})
    
    # 1. 获取 img_size: 优先从 args.model['img_size'] 取，否则从 args.img_size 取，默认 224
    if isinstance(model_conf, dict) and 'img_size' in model_conf:
        img_size = model_conf['img_size']
    else:
        img_size = getattr(args, 'img_size', 224)

    # 2. 获取 batch_size: 优先从 args.training['batch_size'] 取，否则从 args.batch_size 取，默认 32
    if isinstance(training_conf, dict) and 'batch_size' in training_conf:
        batch_size = training_conf['batch_size']
    else:
        batch_size = getattr(args, 'batch_size', 32)

    num_workers = args.num_workers

    print(f"构建 HiT Dataloaders | Img Size: {img_size} | Batch Size: {batch_size} | Workers: {num_workers}")

    # 1. 构建 Transforms
    train_transform = build_hit_transform(is_train=True, img_size=img_size)
    val_transform = build_hit_transform(is_train=False, img_size=img_size)

    # 2. 构建 Datasets
    # 假设目录结构为标准的 ImageNet 格式: root/train/class_xxx/xxx.jpg
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)

    # 3. 分布式采样器 (DistributedSampler)
    # 自动处理数据切分，确保每张 GPU 看到不同的数据
    if torch.distributed.is_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    # 4. 构建 DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None), # 如果有 Sampler，shuffle 必须为 False
        num_workers=num_workers,
        pin_memory=True,                 # 加速 CPU -> GPU 传输
        sampler=train_sampler,
        drop_last=True,                  # 丢弃最后不足一个 batch 的数据，保持形状一致
        persistent_workers=True if num_workers > 0 else False, # 保持 worker 进程存活，减少 epoch 间开销
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        sampler=val_sampler,
        persistent_workers=True if num_workers > 0 else False
    )

    return {'train': train_loader, 'val': val_loader}

# --- 测试代码块 ---
if __name__ == '__main__':
    import time
    import argparse
    
    parser = argparse.ArgumentParser(description='SHF Quadtree Dataloader with Timm Augmentation Test')
    parser.add_argument('--data_dir', type=str, default="/work/c30636/dataset/imagenet/", help='Path to the ImageNet dataset.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for DataLoader.')
    parser.add_argument('--num_workers', type=int, default=48, help='Number of workers for DataLoader.')
    parser.add_argument('--visualize', action='store_true', help='Generate and save a visualization of one batch.')
    args = parser.parse_args()

    # 默认测试用 img_size (如果 build_hit_dataloaders 需要)
    args.img_size = 224

    if os.path.exists(args.data_dir):
        print(f"\n--- 开始 DataLoader 测试 ---")
        print(f"Data Dir: {args.data_dir}")
        
        loaders = build_hit_dataloaders(args)
        
        if args.visualize:
            print("Generating visualization...")
            # 简单的可视化保存逻辑
            import matplotlib.pyplot as plt
            import numpy as np
            
            def denormalize(tensor):
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                return tensor * std + mean

            batch_imgs, _ = next(iter(loaders['val']))
            img = denormalize(batch_imgs[0]).permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            plt.imsave("dataloader_viz.png", img)
            print("Visualization saved to 'dataloader_viz.png'")
        
        # 速度测试
        print("\n--- Speed Test (Train Loader) ---")
        start = time.time()
        count = 0
        max_steps = 20
        
        for i, (imgs, labels) in enumerate(loaders['train']):
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