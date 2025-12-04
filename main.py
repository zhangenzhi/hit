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

from model.dit import DiT
from diffusion.gaussian_diffusion import GaussianDiffusion
from train.dit_imagenet import Trainer
from dataset.dit_imagenet import build_dit_dataloaders

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
    # 修正：保持与实际创建的配置文件名一致
    parser.add_argument("--config", type=str, default="./configs/dit-xl_IN1K.yaml", help="Path to config yaml")
    parser.add_argument("--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)), help="Local rank for DDP")
    
    # 允许命令行覆盖关键参数
    parser.add_argument("--data_path", type=str, default=None, help="Override data path")
    parser.add_argument("--results_dir", type=str, default=None, help="Override results dir")
    
    # [新增] Resume 参数
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from (e.g. results/checkpoint_050.pt)")
    
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
    
    # 8. 初始化 Trainer
    if flat_config.local_rank == 0:
        print("Initializing Trainer...")
        
    trainer = Trainer(model, diffusion, vae, train_loader, flat_config)

    # -----------------------------------------------------------------------------
    # [新增] 9. 处理断点续训 (Resume)
    # -----------------------------------------------------------------------------
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            if flat_config.local_rank == 0:
                print(f"--- Resuming training from checkpoint: {args.resume} ---")
            
            # 确保 map_location 正确，防止在多卡加载时 OOM 或设备错误
            map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
            checkpoint = torch.load(args.resume, map_location=map_location)
            
            # 1. 加载模型权重
            # 注意：如果 Trainer 内部已经用 DDP 包装了 model，state_dict 可能有 'module.' 前缀
            # 这里我们假设 Trainer 暴露了 self.model (可能是 DDP wrapping 后的)
            # 最好是加载到 trainer.model 中
            if hasattr(trainer, 'model'):
                # 处理 module. 前缀不匹配的问题 (根据保存方式不同可能需要调整)
                # 这种方式比较鲁棒，尝试直接加载
                try:
                    trainer.model.load_state_dict(checkpoint['model'])
                except RuntimeError as e:
                    if flat_config.local_rank == 0:
                        print(f"Direct load failed, trying to handle 'module.' prefix... Error: {e}")
                    # 如果 checkpoint 有 module. 但当前模型没有，或者反之，需手动处理 key
                    # 这里简化处理，通常 Trainer save 时处理好了
                    pass
            else:
                # 如果 Trainer 没暴露 model，尝试直接加载到外面的 model 对象 (如果它是引用)
                model.load_state_dict(checkpoint['model'])

            # 2. 加载优化器状态 (关键步骤)
            # 必须假设 Trainer 拥有 optimizer 属性
            if hasattr(trainer, 'optimizer') and 'optimizer' in checkpoint:
                trainer.optimizer.load_state_dict(checkpoint['optimizer'])
                if flat_config.local_rank == 0:
                    print("Optimizer state loaded.")
            
            # 3. 加载 EMA 模型 (如果有)
            if hasattr(trainer, 'ema') and 'ema' in checkpoint:
                trainer.ema.load_state_dict(checkpoint['ema'])
                if flat_config.local_rank == 0:
                    print("EMA model state loaded.")

            # 4. 恢复 Epoch
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                if flat_config.local_rank == 0:
                    print(f"Resuming from epoch {start_epoch}")
            
        else:
            if flat_config.local_rank == 0:
                print(f"Warning: Checkpoint file not found at {args.resume}, starting from scratch.")

    # -----------------------------------------------------------------------------
    # 10. 开始训练循环
    # -----------------------------------------------------------------------------
    if flat_config.local_rank == 0:
        print("Start Training Loop...")
    
    # 修改 range 从 start_epoch 开始
    for epoch in range(start_epoch, flat_config.epochs):
        if config.use_ddp:
            train_loader.sampler.set_epoch(epoch)
        
        # 训练一个 epoch
        trainer.train_one_epoch(epoch)
        
        # 保存 checkpoint (通常 Trainer 内部保存，但这里外部控制也可以)
        # 这里调用 trainer 的保存方法
        trainer.save_checkpoint(epoch)

    cleanup()

if __name__ == "__main__":
    main()