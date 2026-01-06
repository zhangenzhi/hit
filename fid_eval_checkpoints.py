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

# -----------------------------------------------------------------------------
# 核心逻辑
# -----------------------------------------------------------------------------

def prepare_real_statistics(loader, fid_metric, device, num_batches, rank):
    """
    预先计算真实图片的统计信息 (只需运行一次)
    """
    if rank == 0:
        print(f"--> Pre-calculating statistics for REAL images ({num_batches} batches)...")
    
    # 确保不计算梯度
    with torch.no_grad():
        for i, (real_imgs, _) in enumerate(tqdm(loader, disable=(rank != 0), desc="Real Stats")):
            if i >= num_batches:
                break
            
            real_imgs = real_imgs.to(device)
            
            # 假设 Loader 输出是 Latent，需要 VAE Decode 吗？
            # 通常 FID 是在 Pixel 空间计算的。
            # 如果 Loader 此时输出的是 Pixel (ImageNet Raw)，则归一化处理。
            # 如果 Loader 输出的是 Latent，这里需要 decode。
            # 为了通用性，这里假设 Loader 输出的是 Pixel [-1, 1] 或者 [0, 1]
            # 请根据你的 Dataset 实际输出调整。
            
            # 假设: real_imgs 是 [-1, 1] 的 Pixel tensor
            real_imgs = (real_imgs + 1.0) / 2.0  # -> [0, 1]
            real_imgs = real_imgs.clamp(0, 1)
            real_imgs_uint8 = (real_imgs * 255.0).to(torch.uint8)
            
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
    # weights_only=False 是为了兼容旧版 PyTorch 保存习惯，新版建议 True 但可能报错
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    if "ema" in checkpoint:
        state_dict = checkpoint["ema"]
        if rank == 0: print("    Using EMA weights.")
    else:
        state_dict = checkpoint["model"]
        if rank == 0: print("    Using standard weights.")
        
    state_dict = remove_module_prefix(state_dict)
    
    # 2. 加载到模型 (In-place)
    # 注意：这里不需要 unwrap_model，因为 model 本身就没有被 DDP 包裹
    msg = model.load_state_dict(state_dict, strict=True)
    # if rank == 0: print(f"    Load status: {msg}")
    
    model.eval()
    
    # 3. 生成过程
    # 必须设置不同的种子，否则所有 Rank 生成一样的图
    seed = epoch * 100 + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    total_gen = 0
    
    # FID metric 的 reset_real_features=False，所以调用 reset() 只会清空 fake 图片
    fid_metric.reset()

    # 使用 tqdm 显示进度 (只在 Rank 0)
    pbar = tqdm(total=num_gen_batches, disable=(rank != 0), desc=f"Gen Epoch {epoch}")
    
    for _ in range(num_gen_batches):
        n_samples = config.batch_size # 每个 GPU 的 batch size
        labels = torch.randint(0, config.num_classes, (n_samples,), device=device)
        
        # 构造输入形状
        size = (config.in_channels, config.input_size, config.input_size)
        
        # 采样 (直接调用 diffusion，传入裸模型)
        # 强制使用 float32 或 bfloat16 进行采样，避免精度问题
        dtype = torch.bfloat16 if config.use_amp else torch.float32
        
        with torch.no_grad():
            # 这里调用 sample_ddpm 或 sample_ddim
            # 确保你的 diffusion.sample_xxx 支持传入 model
            z = diffusion.sample_ddpm(
                model=model,
                labels=labels,
                size=size,
                num_classes=config.num_classes,
                cfg_scale=4.0, # 标准 FID 配置通常是 1.5 - 4.0，需保持一致
                use_amp=config.use_amp,
                dtype=dtype,
                is_latent=(vae is not None)
            )
            
            # 解码 (Latent -> Pixel)
            if vae is not None:
                # VAE decode 最好用 float32 避免溢出
                z = z.to(torch.float32) / 0.18215
                x = vae.decode(z).sample
            else:
                x = z
            
            # 归一化并转 uint8
            x = (x / 2 + 0.5).clamp(0, 1)
            x_uint8 = (x * 255.0).to(torch.uint8)
            
            # 更新 Metric
            fid_metric.update(x_uint8, real=False)
            
        total_gen += n_samples
        pbar.update(1)
    
    pbar.close()
    
    # 4. 计算 FID
    # torchmetrics 的 compute() 会自动处理跨进程同步 (AllGather)
    # 注意：确保所有进程都执行到这里
    if rank == 0: print("    Computing FID score (synchronizing across GPUs)...")
    fid_score = fid_metric.compute()
    
    return fid_score.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/dit-b_IN1K.yaml")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--num_fid_batches", type=int, default=5, help="Total batches per GPU for FID (Rec: >=50)")
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
    
    # 覆盖参数
    if args.data_path: config.data.data_path = args.data_path
    
    flat_config = flatten_config(config)
    flat_config.local_rank = local_rank
    flat_config.use_ddp = True # 虽然我们不用 DDP Wrapper，但数据加载可能需要知道
    
    # --- 3. 模型初始化 ---
    if rank == 0: print(f"Initializing models on {device}...")
    
    # VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    for p in vae.parameters(): p.requires_grad = False
    
    # DiT (注意：这里不要用 DDP 包裹！只在 load_state_dict 时加载权重即可)
    # 这样采样时每个卡跑自己的，没有通信开销
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
    # 彻底禁用 compile 以避免加载权重时的动态重编译问题
    # model = torch.compile(model) 

    # Diffusion
    diffusion = GaussianDiffusion(
        num_timesteps=config.diffusion.num_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        schedule="cosine",
        device=device
    )

    # --- 4. FID Metric 初始化 ---
    # feature=2048 是 FID 的标准
    # reset_real_features=False 是核心：我们在循环外只算一次 Real Stats
    fid_metric = FrechetInceptionDistance(feature=2048, reset_real_features=False).to(device)

    # --- 5. 数据加载 ---
    loaders = build_dit_dataloaders(flat_config)
    # 如果验证集很大，我们不需要完整的 Loader，只需要能取够 num_fid_batches 即可
    # 这里的 val_loader 需要支持 DistributedSampler，确保不同 Rank 读不同数据
    val_loader = loaders['val']
    if val_loader is None:
        if rank == 0: print("Warning: No validation loader found. Using train loader for Real Stats.")
        val_loader = loaders['train']

    # --- 6. 预计算 Real Stats (Golden Reference) ---
    # 这一步只做一次！
    prepare_real_statistics(val_loader, fid_metric, device, num_batches=args.num_fid_batches, rank=rank)

    # --- 7. Checkpoint 扫描 ---
    ckpt_files = glob.glob(os.path.join(args.checkpoint_dir, "checkpoint_*.pt"))
    ckpt_files.sort(key=lambda x: extract_epoch_from_filename(os.path.basename(x)))
    
    # 过滤 Interval
    if args.interval > 1:
        ckpt_files = [f for f in ckpt_files if extract_epoch_from_filename(os.path.basename(f)) % args.interval == 0]

    if rank == 0:
        print(f"Found {len(ckpt_files)} checkpoints to evaluate.")
        log_file = os.path.join(args.checkpoint_dir, "fid_results_standalone.txt")
        if not os.path.exists(log_file):
            with open(log_file, "w") as f: f.write("Epoch, FID\n")

    # --- 8. 循环评估 ---
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
            print(f"Error evaluating {ckpt_path}: {e}")
            # 继续下一个，不要因为一个损坏的 checkpoint 停止
            continue
            
    cleanup()

if __name__ == "__main__":
    main()