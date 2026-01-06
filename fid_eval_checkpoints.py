import argparse
import os
import sys
import glob
import re
import yaml
import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from diffusers.models import AutoencoderKL
from types import SimpleNamespace
from collections import OrderedDict
from torchmetrics.image.fid import FrechetInceptionDistance

# -----------------------------------------------------------------------------
# 路径设置
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

try:
    from model.dit import DiT
    from diffusion.gaussian_diffusion import GaussianDiffusion
    from dataset.dit_latent_imagenet import build_dit_dataloaders
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------
def dict_to_namespace(d):
    x = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(x, k, dict_to_namespace(v))
        else:
            setattr(x, k, v)
    return x

def flatten_config(config_ns):
    flat_cfg = SimpleNamespace()
    def _extract(ns):
        for k, v in ns.__dict__.items():
            if isinstance(v, SimpleNamespace):
                _extract(v)
            else:
                setattr(flat_cfg, k, v)
    _extract(config_ns)
    return flat_cfg

def extract_epoch_from_filename(filename):
    match = re.search(r'checkpoint_(\d+).pt', filename)
    return int(match.group(1)) if match else -1

def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def decode_latents(latents, vae):
    """
    通用解码函数：将 Latent (B, 4, H, W) 解码为 RGB (B, 3, H*8, W*8)
    """
    # 1. 检查是否需要 scaling (SD VAE 默认 scale factor 是 0.18215)
    # 训练时通常是 latents = encoder(x) * 0.18215
    # 所以解码时要除以 0.18215
    latents = latents.to(torch.float32) / 0.18215
    
    with torch.no_grad():
        imgs = vae.decode(latents).sample
    
    # 2. 归一化 [-1, 1] -> [0, 1]
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    return imgs

# -----------------------------------------------------------------------------
# 核心逻辑
# -----------------------------------------------------------------------------

def prepare_real_statistics(loader, vae, fid_metric, device, num_batches, rank):
    """
    预先计算真实图片的统计信息
    [Fix]: 如果 Loader 返回的是 4 通道 Latent，必须先用 VAE 解码成 3 通道 RGB
    """
    if rank == 0:
        print(f"--> Pre-calculating statistics for REAL images ({num_batches} batches)...")
    
    with torch.no_grad():
        for i, (real_imgs, _) in enumerate(tqdm(loader, disable=(rank != 0), desc="Real Stats")):
            if i >= num_batches:
                break
            
            real_imgs = real_imgs.to(device)
            
            # [CRITICAL FIX] 检查通道数
            if real_imgs.shape[1] == 4:
                # 这是一个 Latent Tensor，需要解码
                # 假设 loader 返回的数据已经被缩放过 (matches training distribution)
                # 通常 DiT dataset loader 返回的是预先计算好的 latents
                if vae is None:
                    raise ValueError("Loader yields 4-channel latents but VAE is None. Cannot decode to RGB for FID.")
                
                # 显存保护：如果 batch 很大，VAE decode 可能会 OOM，这里直接解
                real_imgs_rgb = decode_latents(real_imgs, vae)
            
            elif real_imgs.shape[1] == 3:
                # 已经是 RGB 图片
                if real_imgs.min() < 0: # 假设是 [-1, 1]
                    real_imgs_rgb = (real_imgs + 1.0) / 2.0
                else:
                    real_imgs_rgb = real_imgs
                real_imgs_rgb = real_imgs_rgb.clamp(0, 1)
            else:
                raise ValueError(f"Unexpected channel count: {real_imgs.shape[1]}")

            # 转为 uint8 并 update FID
            real_imgs_uint8 = (real_imgs_rgb * 255.0).to(torch.uint8)
            fid_metric.update(real_imgs_uint8, real=True)
    
    if rank == 0:
        print("--> Real statistics calculated and cached.")

def evaluate_checkpoint(
    ckpt_path, 
    model, 
    diffusion, 
    vae, 
    fid_metric, 
    config, 
    device, 
    rank, 
    world_size, 
    num_gen_batches
):
    epoch = extract_epoch_from_filename(os.path.basename(ckpt_path))
    if rank == 0:
        print(f"\n>>> Evaluating Epoch {epoch} | {os.path.basename(ckpt_path)}")

    # 1. 加载权重
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Failed to load {ckpt_path}: {e}")
        return 0.0

    if "ema" in checkpoint:
        state_dict = checkpoint["ema"]
        if rank == 0: print("    Using EMA weights.")
    else:
        state_dict = checkpoint["model"]
        if rank == 0: print("    Using standard weights.")
        
    state_dict = remove_module_prefix(state_dict)
    
    # 2. 加载到模型
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # 3. 生成过程
    seed = epoch * 100 + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Reset fake stats only
    fid_metric.reset()

    pbar = tqdm(total=num_gen_batches, disable=(rank != 0), desc=f"Gen Epoch {epoch}")
    
    for _ in range(num_gen_batches):
        n_samples = config.batch_size
        labels = torch.randint(0, config.num_classes, (n_samples,), device=device)
        size = (config.in_channels, config.input_size, config.input_size)
        dtype = torch.bfloat16 if config.use_amp else torch.float32
        
        with torch.no_grad():
            # Sample returns (B, 4, H, W) latents usually
            z = diffusion.sample_ddpm(
                model=model,
                labels=labels,
                size=size,
                num_classes=config.num_classes,
                cfg_scale=4.0, 
                use_amp=config.use_amp,
                dtype=dtype,
                is_latent=(vae is not None)
            )
            
            # [CRITICAL FIX] Decode fake latents to RGB
            if vae is not None:
                x_rgb = decode_latents(z, vae)
            else:
                # Pixel diffusion
                x_rgb = (z / 2 + 0.5).clamp(0, 1)
            
            x_uint8 = (x_rgb * 255.0).to(torch.uint8)
            fid_metric.update(x_uint8, real=False)
            
        pbar.update(1)
    
    pbar.close()
    
    # 4. 计算 FID
    if rank == 0: print("    Computing FID score...")
    fid_score = fid_metric.compute()
    
    return fid_score.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/dit-b_IN1K.yaml")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--num_fid_batches", type=int, default=50)
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument("--local-rank", type=int, default=-1)
    args = parser.parse_args()

    # --- 1. DDP 初始化 ---
    if args.local_rank != -1:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=0, world_size=1)
        print("Warning: Running in single process mode.")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # --- 2. Config ---
    with open(args.config, "r") as f:
        config = dict_to_namespace(yaml.safe_load(f))
    
    if args.data_path: config.data.data_path = args.data_path
    flat_config = flatten_config(config)
    flat_config.local_rank = local_rank
    flat_config.use_ddp = True 
    
    # --- 3. 模型初始化 ---
    if rank == 0: print(f"Initializing models on {device}...")
    
    # VAE (必须加载！)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    for p in vae.parameters(): p.requires_grad = False
    
    # DiT
    model = DiT(
        input_size=config.model.input_size,
        patch_size=config.model.patch_size,
        in_channels=config.model.in_channels,
        hidden_size=config.model.hidden_size,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        class_dropout_prob=0.1,
        num_classes=config.model.num_classes,
        learn_sigma=getattr(config.model, 'learn_sigma', True)
    ).to(device)
    model.eval()

    # Diffusion
    diffusion = GaussianDiffusion(
        num_timesteps=config.diffusion.num_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        schedule="cosine",
        device=device
    )

    # --- 4. FID Metric ---
    fid_metric = FrechetInceptionDistance(feature=2048, reset_real_features=False).to(device)

    # --- 5. Data Loader & Real Stats ---
    loaders = build_dit_dataloaders(flat_config)
    val_loader = loaders['val'] if loaders['val'] is not None else loaders['train']

    # [FIX] 传入 VAE 以便将 Latents 解码为 RGB
    prepare_real_statistics(val_loader, vae, fid_metric, device, num_batches=args.num_fid_batches, rank=rank)

    # --- 6. Eval Loop ---
    ckpt_files = glob.glob(os.path.join(args.checkpoint_dir, "checkpoint_*.pt"))
    ckpt_files.sort(key=lambda x: extract_epoch_from_filename(os.path.basename(x)))
    
    if args.interval > 1:
        ckpt_files = [f for f in ckpt_files if extract_epoch_from_filename(os.path.basename(f)) % args.interval == 0]

    if rank == 0:
        log_file = os.path.join(args.checkpoint_dir, "fid_results_fix.txt")
        if not os.path.exists(log_file):
            with open(log_file, "w") as f: f.write("Epoch, FID\n")

    for ckpt_path in ckpt_files:
        try:
            score = evaluate_checkpoint(
                ckpt_path, model, diffusion, vae, fid_metric, 
                flat_config, device, rank, world_size, 
                num_gen_batches=args.num_fid_batches
            )
            
            if rank == 0:
                print(f"*** Epoch {extract_epoch_from_filename(os.path.basename(ckpt_path))} FID: {score:.4f} ***")
                with open(log_file, "a") as f:
                    f.write(f"{extract_epoch_from_filename(os.path.basename(ckpt_path))}, {score:.4f}\n")
        except Exception as e:
            if rank == 0: print(f"Error evaluating {ckpt_path}: {e}")
            continue
            
    cleanup()

if __name__ == "__main__":
    main()