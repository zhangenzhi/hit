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

# --- 尝试导入 pynvml 用于监控 GPU 利用率 ---
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

def get_gpu_stats(local_rank, nvml_handle=None):
    """
    获取当前 GPU 的显存和利用率信息
    """
    stats = {}
    
    # 1. 显存信息 (PyTorch Native, 最准确)
    mem_reserved = torch.cuda.memory_reserved() / 1024**3
    mem_allocated = torch.cuda.memory_allocated() / 1024**3
    stats['mem'] = f"{mem_allocated:.1f}/{mem_reserved:.1f}GB"
    
    # 2. 利用率信息 (Optional, via pynvml)
    if nvml_handle:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
            stats['util'] = f"{util.gpu}%"
        except Exception:
            stats['util'] = "N/A"
            
    return stats

def main():
    parser = argparse.ArgumentParser(description="Encode ImageNet to Latents with Data Augmentation (DDP Support)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to raw ImageNet train directory")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save latents")
    parser.add_argument("--image_size", type=int, default=256)
    # [优化] 默认 Batch 增大到 32。对于 10 crops，实际 batch 是 320。H100 显存足够。
    parser.add_argument("--batch_size", type=int, default=40, help="Number of *original images* per batch per GPU. (Actual input to VAE = batch_size * num_crops)")
    parser.add_argument("--num_crops", type=int, default=10, help="Number of RandomResizedCrop augmentations per image")
    # [优化] 默认 Worker 增大到 16，充分利用 H100 配套的高性能 CPU。
    parser.add_argument("--num_workers", type=int, default=32, help="Workers per GPU")
    args = parser.parse_args()

    # --- 1. DDP 初始化 ---
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_main_process = (rank == 0)
        
        print(f"[Init] Global Rank {rank} | Local Rank {local_rank} | Device: {torch.cuda.current_device()} | Name: {torch.cuda.get_device_name(device)}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main_process = True
        print(f"[Init] Single Process Mode | Device: {device}")

    if is_main_process:
        print(f"Global Rank 0: Initialized DDP with world_size={world_size}")

    # --- 监控初始化 ---
    nvml_handle = None
    if is_main_process and HAS_PYNVML:
        try:
            pynvml.nvmlInit()
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
        except Exception as e:
            print(f"Warning: Failed to initialize NVML for monitoring: {e}")
            nvml_handle = None

    # --- 2. 准备 VAE & 编译优化 ---
    if is_main_process:
        print(f"Loading VAE to {device}...")
    
    # [优化] 开启 TF32 和 CuDNN Benchmark
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True # 针对固定输入尺寸寻找最优卷积算法

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    # [优化] 使用 torch.compile 编译模型
    # mode="max-autotune" 在 H100 上通常能获得最佳推理吞吐量，但编译时间稍长
    # mode="reduce-overhead" 启动更快，适合小 batch，但吞吐量可能略低
    if is_main_process:
        print("Compiling VAE with torch.compile (mode='max-autotune')...")
    
    # 编译整个 VAE 可能会有动态控制流问题，通常只编译 VAE 的 Encode 部分或整个 VAE 模块
    # 这里直接编译 vae，PyTorch 2.x 对此支持已经很好了
    try:
        vae = torch.compile(vae, mode="max-autotune")
    except Exception as e:
        print(f"Warning: torch.compile failed ({e}), falling back to eager mode.")

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
    
    for root, _, files in os.walk(args.data_path):
        for fname in files:
            if fname.lower().endswith(valid_exts):
                src_path = os.path.join(root, fname)
                rel_path = os.path.relpath(src_path, args.data_path)
                base_name = os.path.splitext(rel_path)[0]
                dst_base = os.path.join(args.save_path, base_name)
                image_paths.append((src_path, dst_base))
    
    image_paths.sort(key=lambda x: x[0])

    if is_main_process:
        print(f"Found {len(image_paths)} total images.")
    
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
                # 这里的循环是在 Worker 进程（CPU）中执行的
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
        pin_memory=True, # [优化] 配合 .to(non_blocking=True) 实现异步传输
        drop_last=False,
        prefetch_factor=4, # [优化] 增加预取因子，让 CPU 跑得比 GPU 快
        persistent_workers=True # [优化] 保持 Worker 进程存活，避免反复初始化
    )

    # --- 6. 编码循环 ---
    if is_main_process:
        print(f"Start encoding... (Batch: {args.batch_size} imgs * {args.num_crops} crops = {args.batch_size * args.num_crops} inputs/iter)")
    
    iterator = tqdm(loader, desc=f"GPU {rank}", disable=not is_main_process)
    
    with torch.no_grad():
        for i, (batch_imgs, batch_dst_bases) in enumerate(iterator):
            B, N, C, H, W = batch_imgs.shape
            
            # [优化] non_blocking=True 配合 pin_memory=True，实现 CPU->GPU 拷贝与 GPU 计算重叠
            flat_imgs = batch_imgs.view(-1, C, H, W).to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', dtype=torch.float16):
                # 第一次运行时 torch.compile 会进行 JIT 编译，可能会稍微卡一下
                dist_post = vae.encode(flat_imgs).latent_dist
                latents = dist_post.sample()
            
            latents = latents.cpu()
            latents = latents.view(B, N, *latents.shape[1:])
            
            for k, dst_base in enumerate(batch_dst_bases):
                try:
                    os.makedirs(os.path.dirname(dst_base), exist_ok=True)
                    for j in range(N):
                        save_path = f"{dst_base}_{j}.pt"
                        torch.save(latents[k, j].clone(), save_path)
                except OSError:
                    pass
            
            if is_main_process and i % 5 == 0:
                stats = get_gpu_stats(local_rank, nvml_handle)
                postfix_str = f"Mem:{stats.get('mem')} "
                if 'util' in stats:
                    postfix_str += f"Util:{stats.get('util')}"
                iterator.set_postfix_str(postfix_str)

    if is_main_process:
        print(f"Done! Latents saved to {args.save_path}")
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()