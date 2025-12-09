import os
import torch
import logging
import random
from torchvision import datasets
from torch.utils.data import DataLoader, DistributedSampler, Dataset

# [Fix 2] Fallback 形状修正
# 假设预处理时 num_crops=10。如果不确定，这里返回 None，在 getitem 里处理会更稳健
def robust_loader(path):
    try:
        # map_location='cpu' 避免多线程加载时占用过多 GPU 显存
        return torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None  # 返回 None，让 Dataset 处理

class LatentFolder(datasets.DatasetFolder):
    """
    专门用于读取预编码 Latent 的 Dataset。
    支持从包含多个 crops 的 .pt 文件中进行采样。
    """
    def __init__(self, root, is_train=True):
        super().__init__(
            root,
            loader=robust_loader, 
            extensions=('.pt',), 
            transform=None 
        )
        self.is_train = is_train

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        
        # [Fix 2] 处理加载失败的情况 (Fallback)
        if sample is None:
            # 假设标准尺寸，这里需要根据你的实际 Latent 尺寸硬编码或者动态获取
            # 如果 sample 为 None，我们生成一个全0的 latent 避免崩溃
            # 注意：这里的形状是最终喂给模型的形状 [4, 32, 32]
            return torch.zeros(4, 32, 32), target

        # [Fix 1] 处理多 Crop 维度 [Num_Crops, C, H, W] -> [C, H, W]
        if sample.dim() == 4: 
            num_crops = sample.shape[0]
            if self.is_train:
                # 训练时：随机选一个 crop，实现数据增强效果
                idx = random.randint(0, num_crops - 1)
                sample = sample[idx]
            else:
                # 验证时：固定选第一个，保证验证指标稳定
                # (前提是所有样本也是这样处理的)
                sample = sample[0]
        
        # 此时 sample 形状应为 [4, 32, 32]
        return sample, target

def build_dit_dataloaders(args):
    """
    DiT 专用 DataLoader 构建函数。
    """
    data_dir = getattr(args, 'data_path', getattr(args, 'data_dir', None))
    if data_dir is None:
        raise ValueError("args must contain 'data_path' or 'data_dir'")

    # 获取参数
    model_conf = getattr(args, 'model', {})
    training_conf = getattr(args, 'training', {})
    
    if isinstance(training_conf, dict) and 'batch_size' in training_conf:
        batch_size = training_conf['batch_size']
    else:
        batch_size = getattr(args, 'batch_size', 32)

    num_workers = getattr(args, 'num_workers', 16)
    rank = int(os.environ.get('RANK', 0))
    
    train_root = os.path.join(data_dir, 'train')
    val_root = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_root):
        train_root = data_dir
        val_root = None 

    # --- 自动检测模式 ---
    is_latent = False
    try:
        with os.scandir(train_root) as it:
            first_entry = next(it)
            if first_entry.is_dir():
                with os.scandir(first_entry.path) as it_files:
                    for _ in range(5):
                        f = next(it_files, None)
                        if f and f.name.endswith('.pt'):
                            is_latent = True
                            break
    except (StopIteration, FileNotFoundError, PermissionError):
        pass 

    if rank == 0:
        mode_str = "Latent (.pt) [High Throughput]" if is_latent else "Pixel (Image) [Online Encoding]"
        print(f"Build DiT Dataloaders | Mode: {mode_str} | Batch: {batch_size} | Workers: {num_workers}")

    # --- 构建 Dataset ---
    if is_latent:
        # [Modify] 传入 is_train 标志
        train_dataset = LatentFolder(train_root, is_train=True)
        val_dataset = LatentFolder(val_root, is_train=False) if (val_root and os.path.exists(val_root)) else None
        
    else:
        # Pixel 模式：保持原样
        from torchvision import transforms
        
        if isinstance(model_conf, dict) and 'image_size' in model_conf:
            img_size = model_conf['image_size']
        else:
            img_size = getattr(args, 'image_size', 256)

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        train_dataset = datasets.ImageFolder(train_root, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_root, transform=val_transform) if (val_root and os.path.exists(val_root)) else None

    # --- 构建 Loader ---
    use_ddp = torch.distributed.is_initialized()
    
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if val_dataset else None
    else:
        train_sampler = None
        val_sampler = None

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