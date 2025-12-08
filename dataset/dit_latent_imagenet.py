import os
import torch
import logging
from torchvision import datasets
from torch.utils.data import DataLoader, DistributedSampler, Dataset

# 安全加载函数，防止损坏的文件中断训练
def robust_loader(path):
    try:
        # map_location='cpu' 避免多线程加载时占用过多 GPU 显存
        return torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"Error loading {path}: {e}")
        # 返回一个全 0 的 Latent 作为 Fallback (4, 32, 32) 是 DiT-B/2/4 在 256px 下的 Latent 尺寸
        return torch.zeros(4, 32, 32)

class LatentFolder(datasets.DatasetFolder):
    """
    专门用于读取预编码 Latent 的 Dataset。
    由于数据增强已经在 Encode 阶段完成（保存了 img_0.pt, img_1.pt 等），
    这里不需要再做 RandomResizedCrop。
    """
    def __init__(self, root):
        super().__init__(
            root,
            loader=robust_loader, 
            extensions=('.pt',), 
            transform=None # Latent 模式下不进行 Transform
        )

def build_dit_dataloaders(args):
    """
    DiT 专用 DataLoader 构建函数。
    支持：
    1. Pixel Mode (原始图片): 在线 RandomResizedCrop + VAE Encode (慢，但在显存充足时更灵活)
    2. Latent Mode (预处理特征): 离线增强并保存 (极快，IO 密集型)
    """
    data_dir = getattr(args, 'data_path', getattr(args, 'data_dir', None))
    if data_dir is None:
        raise ValueError("args must contain 'data_path' or 'data_dir'")

    # 获取参数
    model_conf = getattr(args, 'model', {})
    training_conf = getattr(args, 'training', {})
    
    # Batch Size
    if isinstance(training_conf, dict) and 'batch_size' in training_conf:
        batch_size = training_conf['batch_size']
    else:
        batch_size = getattr(args, 'batch_size', 32)

    num_workers = getattr(args, 'num_workers', 16)
    rank = int(os.environ.get('RANK', 0))
    
    train_root = os.path.join(data_dir, 'train')
    val_root = os.path.join(data_dir, 'val')
    
    # 容错：如果 data_dir 直接就是 train 目录
    if not os.path.exists(train_root):
        train_root = data_dir
        val_root = None 

    # --- 自动检测模式 ---
    is_latent = False
    try:
        # 简单探测：检查第一个文件夹里是否有 .pt 文件
        with os.scandir(train_root) as it:
            first_entry = next(it)
            if first_entry.is_dir():
                with os.scandir(first_entry.path) as it_files:
                    # 尝试找几个文件
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
        if is_latent:
            print(f"NOTE: Ensure latents were generated with Data Augmentation (RandomResizedCrop) for best results.")

    # --- 构建 Dataset ---
    if is_latent:
        # Latent 模式：直接读取
        # 由于我们生成了 img_0.pt, img_1.pt，DatasetFolder 会自动将它们视为同一类别的不同样本
        # 从而自然地实现了数据集扩充 (10x)。
        train_dataset = LatentFolder(train_root)
        
        # 验证集通常不需要 Augmentation，如果是 Latent 格式，直接读取即可
        # (通常 FID 评估我们还是建议用 Pixel 模式的 DataLoader 读取真实图片，
        # 但如果是单纯算 Validation Loss，用 Latent 没问题)
        val_dataset = LatentFolder(val_root) if (val_root and os.path.exists(val_root)) else None
        
    else:
        # Pixel 模式：需要在线 Transform
        from torchvision import transforms
        
        # 从 args 获取 image_size
        if isinstance(model_conf, dict) and 'image_size' in model_conf:
            img_size = model_conf['image_size']
        else:
            img_size = getattr(args, 'image_size', 256)

        # DiT Training Transform (Online)
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