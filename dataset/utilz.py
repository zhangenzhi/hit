import argparse
import os
import sys
import yaml
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from types import SimpleNamespace

# -----------------------------------------------------------------------------
# 1. 路径设置: 将项目根目录添加到 sys.path 以便导入模块
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from models.dit import DiT
from diffusion.gaussian_diffusion import GaussianDiffusion
from trainer import DiTImangenetTrainer # 确保 Trainer 类名一致
from data.dataloader import build_dit_dataloaders

def dict_to_namespace(d):
    """
    将嵌套字典转换为 SimpleNamespace 对象，以便使用 .attr 访问
    """
    x = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(x, k, dict_to_namespace(v))
        else:
            setattr(x, k, v)
    return x

def flatten_config(config_ns):
    """
    将嵌套的配置打平，方便 Trainer 访问
    """
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
    
    # 允许命令行覆盖关键参数
    parser.add_argument("--data_path", type=str, default=None, help="Override data path")
    parser.add_argument("--results_dir", type=str, default=None, help="Override results dir")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()

    # 1. 加载 YAML 配置
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # 将字典转为对象结构
    config = dict_to_namespace(config_dict)
    
    # 2. 覆盖配置 (命令行参数优先)
    if args.data_path:
        config.data.data_path = args.data_path
    if args.results_dir:
        config.training.results_dir = args.results_dir
    
    # 设置 DDP 参数到 config
    config.local_rank = args.local_rank
    config.use_ddp = True # 默认开启 DDP 逻辑

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
    
    # 创建 flat config 供 Trainer 使用
    flat_config = flatten_config(config)
    flat_config.local_rank = args.local_rank
    flat_config.use_ddp = config.use_ddp
    
    # 确保输出目录存在
    if flat_config.local_rank == 0:
        os.makedirs(flat_config.results_dir, exist_ok=True)
        print(f"Loading config from {args.config}")
        print(f"Results will be saved to {flat_config.results_dir}")
        print(f"Training on {torch.cuda.device_count()} GPUs")

    # 4. 构建 DataLoaders
    # 这里的 build_dit_dataloaders 现在会根据 flat_config.data_path 的内容
    # 自动决定是返回 ImageFolder (Pixel) 还是 LatentFolder (Latent)
    loaders = build_dit_dataloaders(flat_config)
    train_loader = loaders['train']

    # 5. 初始化 VAE (即使是 Latent 模式，为了 FID 评估的 Decode 阶段，VAE 依然需要)
    if flat_config.local_rank == 0:
        print("Loading VAE (for decoding/evaluation)...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    for param in vae.parameters():
        param.requires_grad = False
    
    # 6. 初始化 DiT 模型
    if flat_config.local_rank == 0:
        print("Initializing DiT Model...")
    
    # 如果是 Latent 模式，输入已经是压缩后的特征，input_size 应该是 32 (256/8)
    # 我们的 Config 默认就是 32，所以不需要改动。
    # 只是如果换了 Pixel 模式，DataLoader 会负责 Resize 到 256，
    # 而 Trainer 内部会负责 Encode 到 32，逻辑是闭环的。
    model = DiT(
        input_size=config.model.input_size,
        patch_size=config.model.patch_size,
        in_channels=config.model.in_channels,
        hidden_size=config.model.hidden_size,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        # learn_sigma=config.model.learn_sigma, # 你的新 DiT 实现可能没有这个参数，如果是标准 DiT 则内置了
        class_dropout_prob=0.1,
        num_classes=config.model.num_classes
    )
    
    # 7. 初始化扩散过程
    diffusion = GaussianDiffusion(
        num_timesteps=config.diffusion.num_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        schedule="cosine", # 强烈建议使用 Cosine Schedule
        device=f"cuda:{args.local_rank}"
    )
    
    # 8. 开始训练
    if flat_config.local_rank == 0:
        print("Start Training...")
        
    trainer = DiTImangenetTrainer(model, diffusion, vae, train_loader, flat_config)

    # 检查恢复训练
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            start_epoch = trainer.resume_checkpoint(args.resume)
        else:
            if flat_config.local_rank == 0:
                print(f"Warning: No checkpoint found at '{args.resume}', starting from scratch.")

    if flat_config.local_rank == 0:
        print(f"Training Loop: Epoch {start_epoch} -> {flat_config.epochs}")
    
    for epoch in range(start_epoch, flat_config.epochs):
        if config.use_ddp:
            train_loader.sampler.set_epoch(epoch)
        trainer.train_one_epoch(epoch)
        trainer.save_checkpoint(epoch)

    cleanup()

if __name__ == "__main__":
    main()