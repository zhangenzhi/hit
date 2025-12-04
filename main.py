import argparse
import os
import sys
import yaml
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from types import SimpleNamespace

# 导入你的模块
from model.dit import DiT
from diffusion.ddim import GaussianDiffusion
from train.dit_imagenet import DiTImangenetTrainer
# 关键：导入你提供的 dataloader 构建函数
from dataset.dit_imagenet import build_dit_dataloaders

# -----------------------------------------------------------------------------
# 1. 路径设置: 将项目根目录添加到 sys.path 以便导入模块
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)



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
    将嵌套的配置打平，方便 Trainer 访问 (兼容旧 Trainer 代码)
    """
    flat_cfg = SimpleNamespace()
    
    # 递归提取所有属性
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
    parser.add_argument("--config", type=str, default="./configs/dit-b_IN1K.yaml", help="Path to config yaml")
    parser.add_argument("--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)), help="Local rank for DDP")
    
    # 允许命令行覆盖关键参数
    parser.add_argument("--data_path", type=str, default=None, help="Override data path")
    parser.add_argument("--results_dir", type=str, default=None, help="Override results dir")
    
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
    try:
        # 检查是否已经初始化 (防止重复初始化)
        dist.get_world_size()
    except RuntimeError:
        # 如果是 torchrun 启动，env 变量已设置
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group("nccl")
        else:
            # 单机单卡调试模式
            print("Not running in DDP mode, initializing mock process group for single GPU.")
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            dist.init_process_group("nccl", rank=0, world_size=1)
            config.use_ddp = False # 标记为非 DDP，虽然 Trainer 可能还是用 DDP wrapper，但此时 rank=0

    torch.cuda.set_device(args.local_rank)
    
    # 创建 flat config 供 Trainer 使用 (因为 Trainer 预期 config.lr, config.epochs 等在顶层)
    flat_config = flatten_config(config)
    # 手动补充 Trainer 需要的额外 flag
    flat_config.local_rank = args.local_rank
    flat_config.use_ddp = config.use_ddp
    
    # 确保输出目录存在
    if flat_config.local_rank == 0:
        os.makedirs(flat_config.results_dir, exist_ok=True)
        print(f"Loading config from {args.config}")
        print(f"Results will be saved to {flat_config.results_dir}")
        print(f"Training on {torch.cuda.device_count()} GPUs")

    # 4. 构建 DataLoaders
    # build_dit_dataloaders 接受一个对象，我们传入 flat_config 即可，它包含 data_path, batch_size 等
    loaders = build_dit_dataloaders(flat_config)
    train_loader = loaders['train']

    # 5. 初始化 VAE (冻结)
    if flat_config.local_rank == 0:
        print("Loading VAE...")
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
        learn_sigma=config.model.learn_sigma,
        num_classes=config.model.num_classes
    )
    
    # 7. 初始化扩散过程
    diffusion = GaussianDiffusion(
        num_timesteps=config.diffusion.num_timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        device=f"cuda:{args.local_rank}"
    )
    
    # 8. 开始训练
    if flat_config.local_rank == 0:
        print("Start Training...")
        
    trainer = DiTImangenetTrainer(model, diffusion, vae, train_loader, flat_config)
    
    for epoch in range(flat_config.epochs):
        if config.use_ddp:
            train_loader.sampler.set_epoch(epoch)
        trainer.train_one_epoch(epoch)
        trainer.save_checkpoint(epoch)

    cleanup()

if __name__ == "__main__":
    main()