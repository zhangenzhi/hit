import argparse
import os
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from diffusers.models import AutoencoderKL
import sys

# --- 尝试导入 pynvml 用于监控 GPU 利用率 ---
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

def get_gpu_stats(local_rank, nvml_handle=None):
    """获取当前 GPU 的显存和利用率信息"""
    stats = {}
    mem_reserved = torch.cuda.memory_reserved() / 1024**3
    mem_allocated = torch.cuda.memory_allocated() / 1024**3
    stats['mem'] = f"{mem_allocated:.1f}/{mem_reserved:.1f}GB"
    
    if nvml_handle:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
            stats['util'] = f"{util.gpu}%"
        except Exception:
            stats['util'] = "N/A"
    return stats

def main():
    parser = argparse.ArgumentParser(description="Encode ImageNet to Latents (Strict Mode: Stops on Bad Image)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to raw ImageNet directory")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save latents")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32, help="Images per batch per GPU") 
    parser.add_argument("--num_workers", type=int, default=16, help="Workers per GPU")
    
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
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main_process = True

    if is_main_process:
        print(f"[Init] World Size: {world_size} | Device: {device}")

    # --- 监控初始化 ---
    nvml_handle = None
    if is_main_process and HAS_PYNVML:
        try:
            pynvml.nvmlInit()
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
        except Exception:
            pass

    # --- 2. 准备 VAE & 精度设置 ---
    if is_main_process:
        print(f"Loading VAE...")
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if is_main_process:
        print(f"Using precision: {dtype}")

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    
    if is_main_process:
        print("Compiling VAE...")
    try:
        vae = torch.compile(vae, mode="max-autotune")
    except Exception as e:
        if is_main_process: print(f"Warning: torch.compile failed, using eager mode. {e}")

    # --- 3. 准备数据增强 ---
    if is_main_process: 
        print(f"Mode: Deterministic (Resize -> CenterCrop {args.image_size})")
    
    transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # --- 4. 扫描文件 ---
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
    my_paths = image_paths[rank::world_size]
    
    if is_main_process:
        print(f"Total images: {len(image_paths)}. Each GPU processing approx: {len(my_paths)}")

    # --- 5. Dataset (包含错误处理) ---
    class StrictDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform
        
        def __len__(self):
            return len(self.paths)
        
        def __getitem__(self, idx):
            src, dst_base = self.paths[idx]
            try:
                # 尝试加载和预处理图片
                img = Image.open(src).convert("RGB")
                img_tensor = self.transform(img)
                return img_tensor, dst_base
            except Exception as e:
                # --- 发现坏图时的处理逻辑 ---
                error_log_file = "bad_images.txt"
                error_msg = f"[Rank {rank}] ❌ FATAL ERROR: Bad image found: {src}"
                
                # 1. 在控制台打印红色错误信息 (如果终端支持)
                print(f"\033[91m{error_msg}\033[0m", file=sys.stderr)
                print(f"Exception details: {e}", file=sys.stderr)
                
                # 2. 将坏图路径写入本地文件
                try:
                    with open(error_log_file, "a") as f:
                        f.write(f"{src}\n")
                    print(f"Path saved to local file: {error_log_file}", file=sys.stderr)
                except Exception as file_e:
                    print(f"Failed to write to log file: {file_e}", file=sys.stderr)
                
                # 3. 抛出 RuntimeError，这会导致 DataLoader worker 崩溃，进而终止主进程
                raise RuntimeError(f"Terminating encoding due to corrupt image: {src}")

    dataset = StrictDataset(my_paths, transform)
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2, 
        persistent_workers=True
    )

    # --- 6. 编码循环 ---
    if is_main_process:
        print(f"Start encoding... Outputting Latents.")
        print(f"NOTE: If a bad image is found, the path will be saved to 'bad_images.txt' and the script will exit.")
    
    # 使用 try-except 捕获 DataLoader 的异常，确保可以优雅地看到错误信息
    try:
        iterator = tqdm(loader, desc=f"GPU {rank}", disable=not is_main_process)
        
        with torch.no_grad():
            for i, (batch_imgs, batch_dst_bases) in enumerate(iterator):
                batch_imgs = batch_imgs.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda', dtype=dtype):
                    dist_post = vae.encode(batch_imgs).latent_dist
                    latents = dist_post.mode()
                
                latents = latents.cpu()
                
                for k, dst_base in enumerate(batch_dst_bases):
                    try:
                        save_dir = os.path.dirname(dst_base)
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(latents[k].clone(), f"{dst_base}.pt")
                    except OSError as e:
                        print(f"Error saving file to disk {dst_base}: {e}")
                
                if is_main_process and i % 10 == 0:
                    stats = get_gpu_stats(local_rank, nvml_handle)
                    postfix_str = f"Mem:{stats.get('mem')}"
                    if 'util' in stats:
                        postfix_str += f" | Util:{stats.get('util')}"
                    iterator.set_postfix_str(postfix_str)

    except RuntimeError as e:
        # 当 Worker 抛出异常时，主进程会在这里捕获
        print(f"\n[Rank {rank}] 🛑 Process Interrupted!", file=sys.stderr)
        print(f"Reason: {e}", file=sys.stderr)
        sys.exit(1) # 非零退出码，通知 DDP 或外部调度器任务失败

    if is_main_process:
        print(f"Done! Latents saved to {args.save_path}")
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()