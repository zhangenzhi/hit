import argparse
import os
import sys
import glob
import re
import yaml
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from types import SimpleNamespace
from collections import OrderedDict

# -----------------------------------------------------------------------------
# 路径设置 (根据实际环境调整)
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# 假设用户的目录结构如下
# ./model/dit.py
# ./diffusion/gaussian_diffusion.py
# ./train/dit_imagenet.py
# ./dataset/dit_latent_imagenet.py

try:
    from model.dit import DiT
    from diffusion.gaussian_diffusion import GaussianDiffusion
    from train.dit_imagenet import DiTImangenetTrainer
    from dataset.dit_latent_imagenet import build_dit_dataloaders
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure your python path includes the project root and the file structure is correct.")
    # Fallback/Debug imports if running in a flat structure (optional)
    # from dit_imagenet import DiTImangenetTrainer

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

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def extract_epoch_from_filename(filename):
    # 匹配 checkpoint_10.pt 中的数字
    match = re.search(r'checkpoint_(\d+).pt', filename)
    return int(match.group(1)) if match else -1

def remove_module_prefix(state_dict):
    """
    移除 DDP 保存时产生的 'module.' 前缀
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict

def main():
    parser = argparse.ArgumentParser(description="Evaluate FID for a sequence of DiT checkpoints")
    parser.add_argument("--config", type=str, default="./configs/dit-b_IN1K.yaml", help="Path to config yaml")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory containing checkpoint_*.pt files")
    parser.add_argument("--data_path", type=str, default=None, help="Override data path (if needed)")
    parser.add_argument("--num_fid_batches", type=int, default=15, help="Number of batches for FID evaluation (default 15 * 256 ~= 3840 images)")
    parser.add_argument("--interval", type=int, default=1, help="Interval for evaluating checkpoints (e.g. 10 means evaluate every 10th checkpoint)")
    parser.add_argument("--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)), help="Local rank for DDP")
    
    args = parser.parse_args()

    # 1. 加载配置
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        return

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    config = dict_to_namespace(config_dict)
    
    # 覆盖配置
    if args.data_path:
        config.data.data_path = args.data_path
    
    # 设置评估结果输出目录 (复用 checkpoint 目录)
    config.training.results_dir = args.checkpoint_dir 
    
    config.local_rank = args.local_rank
    config.use_ddp = True

    # 2. 初始化分布式环境
    if not dist.is_initialized():
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group("nccl")
        else:
            print("Not running in DDP mode, initializing mock process group for single GPU.")
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            dist.init_process_group("nccl", rank=0, world_size=1)
            config.use_ddp = False 

    torch.cuda.set_device(args.local_rank)
    flat_config = flatten_config(config)
    flat_config.local_rank = args.local_rank
    flat_config.use_ddp = config.use_ddp

    if flat_config.local_rank == 0:
        print(f"Scanning checkpoints in: {args.checkpoint_dir}")

    # 3. 扫描并排序 Checkpoints
    ckpt_files = glob.glob(os.path.join(args.checkpoint_dir, "checkpoint_*.pt"))
    # 按 epoch 从小到大排序
    ckpt_files.sort(key=lambda x: extract_epoch_from_filename(os.path.basename(x)))
    
    # [Modify] 过滤 Checkpoints
    if args.interval > 1:
        ckpt_files = [f for f in ckpt_files if extract_epoch_from_filename(os.path.basename(f)) % args.interval == 0]

    if len(ckpt_files) == 0:
        if flat_config.local_rank == 0:
            print(f"No checkpoints found matching interval {args.interval}!")
        return

    # 4. 构建 DataLoaders
    # flat_config 需要包含 batch_size, num_workers 等信息，通常在 yaml 的 data 部分
    # 这里的 flat_config 是 flatten 后的，确保 key 存在
    loaders = build_dit_dataloaders(flat_config)
    train_loader = loaders['train'] 
    val_loader = loaders['val']
    
    if val_loader is None and flat_config.local_rank == 0:
        print("!!! WARNING: Validation loader is None. FID will be calculated on Training set (Incorrect Protocol). Check your data path.")

    # 5. 初始化模型组件
    if flat_config.local_rank == 0: print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    for param in vae.parameters():
        param.requires_grad = False
    
    if flat_config.local_rank == 0: print("Initializing DiT Model...")
    
    # 获取 learn_sigma 配置，默认为 True
    learn_sigma = getattr(config.model, 'learn_sigma', True)
    
    model = DiT(
        input_size=config.model.input_size,
        patch_size=config.model.patch_size,
        in_channels=config.model.in_channels,
        hidden_size=config.model.hidden_size,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        class_dropout_prob=0.1, # 评估时此参数不影响 inference
        num_classes=config.model.num_classes,
        learn_sigma=learn_sigma
    )
    
    diffusion = GaussianDiffusion(
        num_timesteps=config.diffusion.num_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        schedule="cosine", # 确保与训练时一致
        device=f"cuda:{args.local_rank}"
    )
    
    # 6. 初始化 Trainer
    trainer = DiTImangenetTrainer(model, diffusion, vae, train_loader, val_loader, flat_config)

    # 7. 循环评估
    log_file = os.path.join(args.checkpoint_dir, "fid_eval_summary.txt")
    
    for ckpt_path in ckpt_files:
        epoch = extract_epoch_from_filename(os.path.basename(ckpt_path))
        
        if flat_config.local_rank == 0:
            print(f"\n>>> Processing Epoch {epoch} | Checkpoint: {ckpt_path}")
        
        try:
            # map_location 避免加载到 CPU 导致 OOM，或 GPU id 不匹配
            checkpoint = torch.load(ckpt_path, map_location=trainer.device, weights_only=False)
            
            # [Key Logic] 优先加载 EMA 权重进行评估
            # Trainer 中的 self.ema 被注释掉了，所以我们手动将 EMA 权重加载到 trainer.model 中
            if "ema" in checkpoint:
                if flat_config.local_rank == 0: print("Found EMA weights, using EMA for evaluation.")
                state_dict = checkpoint["ema"]
            else:
                if flat_config.local_rank == 0: print("No EMA weights found, using standard model weights.")
                state_dict = checkpoint["model"]
            
            # 处理 state_dict 的 key (移除 module. 前缀) 以适配 trainer.model
            # Trainer 内部虽然包了 DDP，但 load_state_dict 最好对齐 key
            clean_state_dict = remove_module_prefix(state_dict)
            
            # 如果 trainer.model 是 DDP 包装过的，它期望 key 有 module. 
            # 或者我们可以直接加载到 trainer.model.module
            if isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel):
                trainer.model.module.load_state_dict(clean_state_dict)
            else:
                trainer.model.load_state_dict(clean_state_dict)

            # 运行评估
            # evaluate_fid 已经修复了死锁问题
            trainer.evaluate_fid(epoch, num_gen_batches=args.num_fid_batches)
            
            # 记录结果
            if flat_config.local_rank == 0:
                fid_log_path = os.path.join(args.checkpoint_dir, "fid_log.txt")
                if os.path.exists(fid_log_path):
                    with open(fid_log_path, 'r') as f:
                        lines = f.readlines()
                        last_line = lines[-1].strip() if lines else "No result"
                    
                    with open(log_file, 'a') as f:
                        f.write(f"Checkpoint: {os.path.basename(ckpt_path)} | {last_line}\n")
                    print(f"Recorded: {last_line}")

        except Exception as e:
            if flat_config.local_rank == 0:
                print(f"Failed to evaluate {ckpt_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    cleanup()

if __name__ == "__main__":
    main()