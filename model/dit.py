import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from time import time

class Trainer:
    def __init__(self, model, diffusion, vae, loader, config):
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        
        self.model = model.to(self.device)
        # VAE 用于将 Pixel 压缩为 Latent，训练时冻结
        self.vae = vae.to(self.device).eval() 
        self.diffusion = diffusion
        
        if config.use_ddp:
            self.model = DDP(self.model, device_ids=[config.local_rank])
            
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.lr, 
            weight_decay=0.0
        )
        self.loader = loader
        # H100 推荐使用 BF16
        self.dtype = torch.bfloat16

    def train_one_epoch(self, epoch):
        self.model.train()
        start_time = time()
        
        # dataloader 产生的是 Pixel 图像 (N, 3, H, W)，值域 [-1, 1]
        for step, (images, labels) in enumerate(self.loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 1. VAE Encoding: Pixel -> Latent
            # 使用 torch.no_grad 避免 VAE 梯度计算
            with torch.no_grad():
                # SD VAE 需要输入归一化后的图像
                dist = self.vae.encode(images).latent_dist
                latents = dist.sample() * 0.18215 # SD 缩放因子
                
            # 2. 采样时间步
            t = torch.randint(0, self.diffusion.num_timesteps, (images.shape[0],), device=self.device)
            
            # 3. 混合精度训练 (BF16)
            with torch.amp.autocast('cuda', dtype=self.dtype):
                loss = self.diffusion.p_losses(self.model, latents, t, labels)
                
            # 4. 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 日志
            if step % 100 == 0 and self.config.local_rank == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | Time: {time() - start_time:.2f}s")
                start_time = time()
                
    def save_checkpoint(self, epoch):
        if self.config.local_rank == 0:
            checkpoint_path = os.path.join(self.config.results_dir, f"checkpoint_{epoch}.pt")
            to_save = self.model.module.state_dict() if self.config.use_ddp else self.model.state_dict()
            torch.save(to_save, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")