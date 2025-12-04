import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
from time import time

class Trainer:
    """
    负责训练循环、混合精度、分布式、日志
    """
    def __init__(self, model, diffusion, vae, dataset, config):
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        
        # 1. 准备模型
        self.model = model.to(self.device)
        self.vae = vae.to(self.device).eval() # VAE 冻结
        self.diffusion = diffusion
        
        # 2. 分布式设置
        if config.use_ddp:
            self.model = DDP(self.model, device_ids=[config.local_rank])
            
        # 3. 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.lr, 
            weight_decay=0.0
        )
        
        # 4. 数据加载
        sampler = DistributedSampler(dataset) if config.use_ddp else None
        self.loader = DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            sampler=sampler, 
            num_workers=config.num_workers,
            pin_memory=True
        )

        # 5. H100 关键优化: BF16
        # H100 上不需要 GradScaler (BF16 范围足够大)
        self.dtype = torch.bfloat16

    def train_one_epoch(self, epoch):
        self.model.train()
        start_time = time()
        
        for step, (images, labels) in enumerate(self.loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 1. VAE Encode (No Grad, FP16 or FP32)
            # 实际中通常建议预先处理好 Latent 存硬盘，否则 VAE 会占用 GPU 显存并拖慢训练
            with torch.no_grad():
                # 假设 images 已经是 tensor
                # map to [-1, 1] if needed
                latents = self.vae.encode(images).latent_dist.sample()
                latents = latents * 0.18215 # Scale factor specifically for SD VAE
                
            # 2. 采样时间步 t
            t = torch.randint(0, self.diffusion.num_timesteps, (images.shape[0],), device=self.device)
            
            # 3. 前向传播 & Loss (BF16 Context)
            with torch.amp.autocast('cuda', dtype=self.dtype):
                loss = self.diffusion.p_losses(self.model, latents, t, labels)
                
            # 4. 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Log
            if step % 100 == 0 and self.config.local_rank == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | Time: {time() - start_time:.2f}s")
                start_time = time()
                
    def save_checkpoint(self, epoch):
        if self.config.local_rank == 0:
            checkpoint_path = os.path.join(self.config.results_dir, f"checkpoint_{epoch}.pt")
            to_save = self.model.module.state_dict() if self.config.use_ddp else self.model.state_dict()
            torch.save(to_save, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
