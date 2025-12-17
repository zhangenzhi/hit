import os
import torch
import logging
import random
from torchvision import datasets
from torch.utils.data import DataLoader, DistributedSampler, Dataset

def robust_loader(path):
    # [修正 1] 移除 try-except。
    # 必须让错误暴露出来！如果文件坏了或路径不对，程序应该崩溃而不是静默失败。
    # 否则模型会训练在全0数据上，导致生成全是噪声。
    return torch.load(path, map_location='cpu')

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
        
        # 直接加载，如果有问题（如文件损坏），这里会直接报错终止训练
        sample = self.loader(path)
        
        # [修正 2] 移除 "return torch.zeros" 的 Fallback 逻辑
        # 严禁给模型喂全0数据！
        
        # 处理维度 [Num_Crops, C, H, W] -> [C, H, W]
        # 只有当数据包含 Crop 维度时才处理
        if sample.dim() == 4: 
            num_crops = sample.shape[0]
            if self.is_train:
                # 训练时：随机选一个 crop，实现数据增强
                idx = random.randint(0, num_crops - 1)
                sample = sample[idx]
            else:
                # 验证时：固定选第一个
                # [重要提示] 请确保验证集的 .pt 是用 CenterCrop 生成的 (num_crops=1)
                # 否则这里取到的就是随机裁剪图，会导致 FID 虚高。
                sample = sample[0]
        
        # 此时 sample 形状应为 [C, H, W] (e.g. [4, 32, 32])
        return sample, target

def build_dit_dataloaders(args):
    """
    DiT 专用 DataLoader 构建函数。
    """
    data_dir = getattr(args, 'data_path', getattr(args, 'data_dir', None))
    if data_dir is None:
        raise ValueError("args must contain 'data_path' or 'data_dir'")

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
    
    # [修正 3] 更智能的路径回退检查
    # 如果标准的 train/val 结构不存在，检查是否 data_dir 本身就是 train root
    if not os.path.exists(train_root):
        if rank == 0:
            print(f"Warning: '{train_root}' not found. Trying to use '{data_dir}' as train root.")
        train_root = data_dir
        # 如果连 val 文件夹都不存在，则禁用验证集
        if not os.path.exists(val_root):
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
        val_status = f"Found at {val_root}" if (val_root and os.path.exists(val_root)) else "NOT FOUND (FID will be inaccurate)"
        print(f"Build DiT Dataloaders | Mode: {mode_str} | Batch: {batch_size} | Workers: {num_workers}")
        print(f"Validation Set: {val_status}")

    # --- 构建 Dataset ---
    if is_latent:
        train_dataset = LatentFolder(train_root, is_train=True)
        if val_root and os.path.exists(val_root):
            val_dataset = LatentFolder(val_root, is_train=False)
        else:
            val_dataset = None
        
    else:
        # Pixel 模式：用于调试或原始数据
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
        if val_root and os.path.exists(val_root):
            val_dataset = datasets.ImageFolder(val_root, transform=val_transform)
        else:
            val_dataset = None

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