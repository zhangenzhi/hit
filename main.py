import argparse
import os
import sys
import yaml
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from types import SimpleNamespace
import re # [新增] 用于解析文件名中的 epoch

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
            
            # 兼容处理：检查是 纯StateDict 还是 包含元数据的字典
            state_dict = checkpoint
            has_metadata = False
            
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                # 这是一个包含元数据的完整 Checkpoint (如果将来你改了保存逻辑)
                state_dict = checkpoint["model"]
                has_metadata = True
            
            # 1. 加载模型权重
            # 注意：保存时使用了 model.module.state_dict() (无 module. 前缀)
            # 加载时如果 trainer.model 是 DDP，需要加载到 trainer.model.module
            model_to_load = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
            
            try:
                model_to_load.load_state_dict(state_dict, strict=True)
                if flat_config.local_rank == 0:
                    print("Model weights loaded successfully.")
            except Exception as e:
                if flat_config.local_rank == 0:
                    print(f"Strict load failed (keys mismatch?), trying strict=False. Error: {e}")
                missing, unexpected = model_to_load.load_state_dict(state_dict, strict=False)
                if flat_config.local_rank == 0:
                    print(f"Loaded with strict=False. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

            # 2. 恢复 Epoch (尝试从文件名解析)
            if has_metadata and 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            else:
                # 尝试从文件名解析 epoch (例如 checkpoint_050.pt -> 50)
                try:
                    filename = os.path.basename(args.resume)
                    # 匹配 _数字.pt 结尾
                    match = re.search(r'_(\d+)\.pt$', filename)
                    if match:
                        epoch_num = int(match.group(1))
                        start_epoch = epoch_num + 1 # 从下一轮开始
                        if flat_config.local_rank == 0:
                            print(f"Inferred resume epoch {start_epoch} from filename '{filename}'.")
                    else:
                        if flat_config.local_rank == 0:
                            print("Could not infer epoch from filename. Starting from epoch 0.")
                except Exception:
                    pass
            
            # 3. 恢复优化器 (仅当存在元数据时)
            if has_metadata and hasattr(trainer, 'optimizer') and 'optimizer' in checkpoint:
                try:
                    trainer.optimizer.load_state_dict(checkpoint['optimizer'])
                    if flat_config.local_rank == 0:
                        print("Optimizer state loaded.")
                except Exception:
                    pass
            elif flat_config.local_rank == 0:
                print("Note: Optimizer state not found in checkpoint (training with fresh optimizer).")
            
        else:
            if flat_config.local_rank == 0:
                print(f"Warning: Checkpoint file not found at {args.resume}, starting from scratch.")

    # -----------------------------------------------------------------------------
    # 10. 开始训练循环
    # -----------------------------------------------------------------------------
    if flat_config.local_rank == 0:
        print(f"Start Training Loop from epoch {start_epoch}...")
    
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