import torch
import torch.distributed as dist
import argparse
import os
from diffusers.models import AutoencoderKL

# 导入你的模块
from model.dit import DiT
from diffusion.ddim import GaussianDiffusion
from train.dit_imagenet import Trainer
# 关键：导入你提供的 dataloader 构建函数
from dataset.dit_imagenet import build_dit_dataloaders

def cleanup():
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    # 基础训练参数
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    
    # Dataloader 需要的参数 (与 data/dataloader.py 对应)
    parser.add_argument("--data_path", type=str, required=True, help="Path to ImageNet")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    # 1. 初始化分布式环境
    dist.init_process_group("nccl")
    torch.cuda.set_device(args.local_rank)
    args.use_ddp = True
    
    if args.local_rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        print(f"Training DiT on {torch.cuda.device_count()} GPUs")

    # 2. 构建 DataLoaders (使用你的 data/dataloader.py)
    loaders = build_dit_dataloaders(args)
    train_loader = loaders['train']

    # 3. 初始化 VAE (用于 Latent 压缩)
    # H100 环境建议预先缓存 Latent，但这里为了流程完整进行实时编码
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    
    # 4. 初始化 DiT 模型 (DiT-XL/2 Config)
    # image_size 256 -> latent_size 32
    model = DiT(
        input_size=args.image_size // 8, 
        depth=28,
        hidden_size=1152,
        patch_size=2,
        num_heads=16
    )
    
    # 5. 初始化扩散过程
    diffusion = GaussianDiffusion(device=f"cuda:{args.local_rank}")
    
    # 6. 开始训练
    trainer = Trainer(model, diffusion, vae, train_loader, args)
    
    for epoch in range(args.epochs):
        if args.use_ddp:
            train_loader.sampler.set_epoch(epoch)
        trainer.train_one_epoch(epoch)
        trainer.save_checkpoint(epoch)

    cleanup()

if __name__ == "__main__":
    main()