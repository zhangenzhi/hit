import argparse
import os
import sys
import yaml
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from types import SimpleNamespace

# -----------------------------------------------------------------------------
# 1. 路径设置
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from model.dit import DiT
from diffusion.gaussian_diffusion import GaussianDiffusion
from train.dit_imagenet import DiTImangenetTrainer
from dataset.dit_latent_imagenet import build_dit_dataloaders
from train.utilz import save_checkpoint, resume_checkpoint

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

def main():
    parser = argparse.ArgumentParser(description="Train DiT on ImageNet")
    parser.add_argument("--config", type=str, default="./configs/dit_imagenet.yaml", help="Path to config yaml")
    parser.add_argument("--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)), help="Local rank for DDP")
    
    parser.add_argument("--data_path", type=str, default=None, help="Override data path")
    parser.add_argument("--results_dir", type=str, default=None, help="Override results dir")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()

    # 1. 加载 YAML 配置
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    config = dict_to_namespace(config_dict)
    
    # 2. 覆盖配置
    if args.data_path:
        config.data.data_path = args.data_path
    if args.results_dir:
        config.training.results_dir = args.results_dir
    
    config.local_rank = args.local_rank
    config.use_ddp = True

    # 3. 初始化分布式环境
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
    
    # [新增] 注入新优化的默认参数 (防止 YAML 中缺失)
    if not hasattr(flat_config, 'ema_update_every'): flat_config.ema_update_every = 10
    if not hasattr(flat_config, 'compile_mode'): flat_config.compile_mode = 'max-autotune'
    if not hasattr(flat_config, 'label_dropout_prob'): flat_config.label_dropout_prob = 0.1
    
    if flat_config.local_rank == 0:
        os.makedirs(flat_config.results_dir, exist_ok=True)
        print(f"Loading config from {args.config}")
        print(f"Results will be saved to {flat_config.results_dir}")
        print(f"Training on {torch.cuda.device_count()} GPUs")

    # 4. 构建 DataLoaders (使用优化后的 Loader)
    loaders = build_dit_dataloaders(flat_config)
    train_loader = loaders['train']
    val_loader = loaders['val']

    # 5. 初始化 VAE
    if flat_config.local_rank == 0:
        print("Loading VAE (for decoding/evaluation)...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    for param in vae.parameters():
        param.requires_grad = False
    
    # 6. 初始化 DiT 模型
    if flat_config.local_rank == 0:
        print("Initializing DiT Model...")
    
    model = DiT(
        input_size=config.model.input_size,
        patch_size=config.model.patch_size,
        in_channels=config.model.in_channels,
        hidden_size=config.model.hidden_size,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        learn_sigma=getattr(config.model, 'learn_sigma', True),
        class_dropout_prob=0.1,
        num_classes=config.model.num_classes
    )
    
    # 7. 初始化扩散过程
    diffusion = GaussianDiffusion(
        num_timesteps=config.diffusion.num_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        schedule="cosine", 
        device=f"cuda:{args.local_rank}"
    )
    
    # 8. 初始化 Trainer
    if flat_config.local_rank == 0:
        print("Start Training...")
        
    trainer = DiTImangenetTrainer(model, diffusion, vae, train_loader, val_loader, flat_config)

    # [修改] 检查恢复训练 (使用 utilz 中的函数)
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            # 将 trainer 作为参数传入
            start_epoch = resume_checkpoint(trainer, args.resume)
        else:
            if flat_config.local_rank == 0:
                print(f"Warning: No checkpoint found at '{args.resume}', starting from scratch.")

    if flat_config.local_rank == 0:
        print(f"Training Loop: Epoch {start_epoch} -> {flat_config.epochs}")
    
    for epoch in range(start_epoch, flat_config.epochs):
        if config.use_ddp:
            train_loader.sampler.set_epoch(epoch)
        
        # 训练一个 epoch (内部已移除繁重的同步)
        trainer.train_one_epoch(epoch)
        
        # [修改] 保存 checkpoint (使用 utilz 中的函数)
        save_checkpoint(trainer, epoch)

    cleanup()

if __name__ == "__main__":
    main()