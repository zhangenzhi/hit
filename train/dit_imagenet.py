import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import numpy as np
from time import time
from torchvision.utils import save_image, make_grid
from contextlib import nullcontext

class DiTImangenetTrainer:
    def __init__(self, model, diffusion, vae, loader, config):
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        
        self.model = model.to(self.device)
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
        
        # 混合精度开关
        self.use_amp = getattr(config, 'use_amp', True)
        self.dtype = torch.bfloat16 if self.use_amp else torch.float32
        
        if config.local_rank == 0:
            print(f"Training with AMP: {self.use_amp}, Dtype: {self.dtype}")
        
        # 编译模型以加速 (H100 建议开启)
        try:
            self.model = torch.compile(self.model, mode="default")
            if config.local_rank == 0:
                print("Model compiled with torch.compile")
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")

        self.fid_metric = None
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            # 【修复 1】移除 reset_real_features=False，或设为 True
            # 这样每次调用 .reset() 都会清空 Real 和 Fake 的统计量，防止无限累积
            self.fid_metric = FrechetInceptionDistance(feature=2048, reset_real_features=True).to(self.device)
            if self.config.local_rank == 0:
                print("FID metric initialized successfully (Distributed Mode).")
        except ImportError:
            if self.config.local_rank == 0:
                print("Warning: torchmetrics not found. FID evaluation will be skipped.")

    @torch.no_grad()
    def sample_ddim(self, n, labels, size, num_inference_steps=50, eta=0.0):
        """
        DDIM 快速采样 (强制 FP32 计算)
        """
        self.model.eval()
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        
        # 1. 构造时间步
        step_ratio = self.diffusion.num_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(int)
        
        x = torch.randn(n, *size).to(self.device)
        
        for i, t_step in enumerate(timesteps):
            t = torch.full((n,), t_step, device=self.device, dtype=torch.long)
            
            # 2.1 预测噪声
            with torch.amp.autocast('cuda', dtype=self.dtype) if self.use_amp else nullcontext():
                model_output = raw_model(x, t, labels)
            
            # 计算必须用 float32
            model_output = model_output.float()
            x = x.float()
            
            if model_output.shape[1] == 2 * x.shape[1]:
                model_output, _ = model_output.chunk(2, dim=1)
            
            # 2.2 参数获取
            alpha_bar_t = self.diffusion.alphas_cumprod[t_step].to(self.device).float()
            
            if i == len(timesteps) - 1:
                alpha_bar_t_prev = torch.tensor(1.0).to(self.device).float()
            else:
                prev_t = timesteps[i+1]
                alpha_bar_t_prev = self.diffusion.alphas_cumprod[prev_t].to(self.device).float()
            
            # 2.3 更新公式
            sigma_t = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev))
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * model_output) / torch.sqrt(alpha_bar_t)
            dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * model_output
            noise = sigma_t * torch.randn_like(x)
            
            x = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + noise
            
        return x

    @torch.no_grad()
    def visualize(self, epoch):
        if self.config.local_rank != 0:
            return

        self.model.eval()
        n_samples = 4
        labels = torch.randint(0, 1000, (n_samples,), device=self.device)
        
        in_channels = getattr(self.config, 'in_channels', 4)
        input_size = getattr(self.config, 'input_size', 32)
        latent_size = (in_channels, input_size, input_size)

        z = self.sample_ddim(n_samples, labels, latent_size, num_inference_steps=50)
        
        # 移除 .to(self.dtype)
        x_recon = self.vae.decode(z / 0.18215).sample.float()
        x_recon = torch.clamp((x_recon + 1.0) / 2.0, 0.0, 1.0)
        
        save_path = os.path.join(self.config.results_dir, f"sample_epoch_{epoch}.png")
        save_image(x_recon, save_path, nrow=2)
        print(f"[Visual] Saved visualization to {save_path}")
        
        self.model.train()

    @torch.no_grad()
    def evaluate_fid(self, epoch, num_gen_batches=10):
        if self.fid_metric is None:
            return

        if self.config.use_ddp:
            dist.barrier()

        if self.config.local_rank == 0:
            print(f"[FID] Starting distributed evaluation for epoch {epoch}...")
        
        self.model.eval()
        self.fid_metric.reset() # 现在 reset 会正确清空 Real 和 Fake

        # 1. Real Images
        for i, (real_imgs, _) in enumerate(self.loader):
            if i >= num_gen_batches: break
            real_imgs = real_imgs.to(self.device)
            
            if real_imgs.shape[1] == 4:
                # Latent -> Pixel
                real_imgs = self.vae.decode(real_imgs / 0.18215).sample
                real_imgs = real_imgs.clamp(-1, 1)

            real_imgs = ((real_imgs + 1.0) / 2.0).clamp(0.0, 1.0)
            real_imgs_uint8 = (real_imgs * 255.0).to(torch.uint8)
            self.fid_metric.update(real_imgs_uint8, real=True)
        
        # 2. Fake Images
        for i in range(num_gen_batches):
            n_samples = self.config.batch_size
            labels = torch.randint(0, 1000, (n_samples,), device=self.device)
            
            in_channels = getattr(self.config, 'in_channels', 4)
            input_size = getattr(self.config, 'input_size', 32)
            latent_size = (in_channels, input_size, input_size)
            
            z = self.sample_ddim(n_samples, labels, latent_size, num_inference_steps=50)
            
            fake_imgs = self.vae.decode(z / 0.18215).sample.float()
            fake_imgs = ((fake_imgs + 1.0) / 2.0).clamp(0.0, 1.0)
            
            if i == 0 and self.config.local_rank == 0:
                save_path = os.path.join(self.config.results_dir, f"fid_samples_epoch_{epoch}.png")
                save_image(fake_imgs[:8], save_path, nrow=4)
                print(f"[Visual] Saved FID evaluation samples to {save_path}")

            fake_imgs_uint8 = (fake_imgs * 255.0).to(torch.uint8)
            self.fid_metric.update(fake_imgs_uint8, real=False)
            
        fid_score = self.fid_metric.compute()
        
        if self.config.local_rank == 0:
            print(f"[FID] Epoch {epoch} | Distributed FID Score: {fid_score.item():.4f}")
            with open(os.path.join(self.config.results_dir, "fid_log.txt"), "a") as f:
                f.write(f"Epoch {epoch}, FID: {fid_score.item():.4f}\n")

        self.model.train()
        
        # 【修复 2】清理显存，防止 VAE/Inception 残留占用训练内存
        torch.cuda.empty_cache()
        
        if self.config.use_ddp:
            dist.barrier()

    def train_one_epoch(self, epoch):
        self.model.train()
        start_time = time()
        max_steps = getattr(self.config, 'max_train_steps', None)
        
        for step, (images, labels) in enumerate(self.loader):
            if max_steps is not None and step >= max_steps:
                if self.config.local_rank == 0:
                    print(f"Debug: Max steps {max_steps} reached, breaking epoch.")
                break

            images = images.to(self.device)
            labels = labels.to(self.device)
            
            if images.shape[1] == 3:
                with torch.no_grad():
                    posterior = self.vae.encode(images).latent_dist
                    latents = posterior.sample().mul_(0.18215)
            else:
                latents = images * 0.18215
                
            t = torch.randint(0, self.diffusion.num_timesteps, (images.shape[0],), device=self.device)
            
            with torch.amp.autocast('cuda', dtype=self.dtype) if self.use_amp else nullcontext():
                loss = self.diffusion.p_losses(self.model, latents, t, labels)
                
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            if step % 100 == 0 and self.config.local_rank == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | Time: {time() - start_time:.2f}s")
                start_time = time()

        if self.config.local_rank == 0:
            viz_interval = getattr(self.config, 'log_interval', 1)
            if epoch % viz_interval == 0 and not (epoch > 0 and epoch % 5 == 0):
                self.visualize(epoch)
            
        if epoch > 0 and epoch % 5 == 0:
            self.evaluate_fid(epoch, num_gen_batches=15) 

    def save_checkpoint(self, epoch):
        if self.config.local_rank == 0:
            checkpoint_path = os.path.join(self.config.results_dir, f"checkpoint_{epoch}.pt")
            checkpoint = {
                "model": self.model.module.state_dict() if self.config.use_ddp else self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
                "config": self.config,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            latest_path = os.path.join(self.config.results_dir, "latest.pt")
            torch.save(checkpoint, latest_path)

    def resume_checkpoint(self, checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        model_state_dict = checkpoint["model"]
        if self.config.use_ddp:
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)
        print("Model weights loaded.")

        if "optimizer" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("Optimizer state loaded.")
            
        start_epoch = checkpoint.get("epoch", -1) + 1
        print(f"Resuming training from epoch {start_epoch}")
        return start_epoch