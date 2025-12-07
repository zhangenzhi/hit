import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler

class LatentFolder(datasets.DatasetFolder):
    def __init__(self, root):
        super().__init__(
            root,
            loader=torch.load, 
            extensions=('.pt',), 
            transform=None      
        )

def build_dit_transform(is_train, img_size):
    """
    构建适合 DiT (Diffusion Transformer) 训练的数据增强 (仅针对 Pixel 数据)。
    """
    # DiT/Diffusion Model 标准归一化: map [0, 1] to [-1, 1]
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    if is_train:
        transform = transforms.Compose([
            # 【关键修复】使用 RandomResizedCrop 以获得更好的 FID
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
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
    model_conf = getattr(args, 'model', {})
    training_conf = getattr(args, 'training', {})
    
    if isinstance(model_conf, dict) and 'image_size' in model_conf:
        img_size = model_conf['image_size']
    elif isinstance(model_conf, dict) and 'img_size' in model_conf:
        img_size = model_conf['img_size']
    else:
        img_size = getattr(args, 'image_size', getattr(args, 'img_size', 256))

    if isinstance(training_conf, dict) and 'batch_size' in training_conf:
        batch_size = training_conf['batch_size']
    else:
        batch_size = getattr(args, 'batch_size', 32)

    num_workers = getattr(args, 'num_workers', 4)

    rank = int(os.environ.get('RANK', 0))
    
    # 1. 路径检查
    train_root = os.path.join(data_dir, 'train')
    val_root = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_root):
        train_root = data_dir
        val_root = None 

    # 2. 自动检测数据类型 (Pixel vs Latent)
    is_latent = False
    try:
        with os.scandir(train_root) as it:
            first_entry = next(it)
            if first_entry.is_dir():
                with os.scandir(first_entry.path) as it_files:
                    first_file = next(it_files)
                    if first_file.name.endswith('.pt'):
                        is_latent = True
    except (StopIteration, FileNotFoundError):
        pass 

    if rank == 0:
        mode_str = "Latent (.pt) [Fast Mode]" if is_latent else "Pixel (Image) [Standard Mode]"
        print(f"构建 DiT Dataloaders | Mode: {mode_str} | Size: {img_size} | Batch: {batch_size}")

    # 3. 构建 Dataset
    if is_latent:
        # Latent 模式: 使用 LatentFolder，无 Transform
        # 注意: 离线 Latent 通常是固定 Crop 的，这会影响 FID
        train_dataset = LatentFolder(train_root)
        val_dataset = LatentFolder(val_root) if (val_root and os.path.exists(val_root)) else None
    else:
        # Pixel 模式: 使用 ImageFolder，带 RandomResizedCrop
        train_transform = build_dit_transform(is_train=True, img_size=img_size)
        val_transform = build_dit_transform(is_train=False, img_size=img_size)
        
        train_dataset = datasets.ImageFolder(train_root, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_root, transform=val_transform) if (val_root and os.path.exists(val_root)) else None

    # 4. 分布式采样器
    use_ddp = torch.distributed.is_initialized()
    
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if val_dataset else None
    else:
        train_sampler = None
        val_sampler = None

    # 5. 构建 DataLoaders
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