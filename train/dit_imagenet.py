import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
import os
import numpy as np
from time import time
from copy import deepcopy
import math

# 假设 train.utilz 中包含了这些工具函数
# 移除了 visualize, evaluate_fid 的导入，因为我们将它们集成到了类中
from train.utilz import EMA, get_cosine_schedule_with_warmup

class DiTImangenetTrainer:
    def __init__(self, model, diffusion, vae, loader, val_loader, config):
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        
        self.model = model.to(self.device)
        # 如果提供了 VAE，确保它是 eval 模式且不需要梯度
        self.vae = vae.to(self.device).eval() if vae is not None else None
        if self.vae is not None:
            for param in self.vae.parameters():
                param.requires_grad = False
                
        self.diffusion = diffusion
        
        # DDP setup
        if config.use_ddp:
            self.model = DDP(
                self.model, 
                device_ids=[config.local_rank],
                gradient_as_bucket_view=True, 
                static_graph=getattr(config, 'static_graph', True) 
            )
            
        # [Critical Fix] 建议将 ema_update_every 设为 1 以避免 EMA 滞后，或者根据 Batch Size 调整
        self.ema_update_every = getattr(config, 'ema_update_every', 1)
        self.ema = EMA(self.model, decay=0.9999)
            
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.lr, 
            weight_decay=0.0
        )
        self.loader = loader
        self.val_loader = val_loader

        self.use_amp = getattr(config, 'use_amp', True)
        self.dtype = torch.bfloat16 if self.use_amp else torch.float32

        # BF16 dynamic range is usually large enough, so Scaler is often not needed
        # But if using float16, scaler is required.
        use_scaler = self.use_amp and (self.dtype == torch.float16)
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
        
        self.train_loss_history = []
        
        self.num_classes = getattr(config, 'num_classes', 1000)
        self.label_dropout_prob = getattr(config, 'label_dropout_prob', 0.1)

        if config.local_rank == 0:
            print(f"Training with AMP: {self.use_amp}, Dtype: {self.dtype}")
            print(f"GradScaler Enabled: {use_scaler} (Disabled for BF16 recommended)")
            print(f"EMA initialized with decay: {self.ema.decay}, Update Every: {self.ema_update_every} steps")
            print(f"CFG Label Dropout Prob: {self.label_dropout_prob}")
            print(f"Mode: {'Latent Diffusion' if self.vae else 'Pixel Diffusion'}")
        
        try:
            compile_mode = getattr(config, 'compile_mode', 'default') 
            self.model = torch.compile(self.model, mode=compile_mode)
            if config.local_rank == 0:
                print(f"Model compiled with torch.compile (mode={compile_mode})")
        except Exception as e:
            if config.local_rank == 0:
                print(f"Warning: torch.compile failed: {e}. Fallback to Eager mode.")

        self.fid_metric = None
        # Initialize FID metric
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            self.fid_metric = FrechetInceptionDistance(feature=2048, reset_real_features=True).to(self.device)
            if self.config.local_rank == 0:
                print("FID metric initialized successfully.")
        except ImportError:
            if self.config.local_rank == 0:
                print("Warning: torchmetrics not found. FID evaluation will be skipped.")

    def decode_image_or_latent(self, x):
        """
        统一处理 Pixel 空间和 Latent 空间的解码逻辑。
        如果存在 VAE，则视为 Latent 空间，进行解码。
        否则视为 Pixel 空间，直接处理。
        """
        x = x.float()
        if self.vae is not None:
            # Latent Space: Unscale -> Decode
            # 0.18215 is the standard scale factor for KL-f8 VAE (Stable Diffusion)
            x = x / 0.18215 
            with torch.no_grad():
                x = self.vae.decode(x).sample
        
        # Normalize from [-1, 1] to [0, 1] for visualization/FID
        x = (x / 2 + 0.5).clamp(0, 1)
        return x

    @torch.no_grad()
    def sample_ddim(self, n, labels, size, num_inference_steps=50, eta=0.0, cfg_scale=4.0, model=None):
        use_model = model if model is not None else self.model
        use_model.eval()
        # Handle DDP wrapper
        raw_model = use_model.module if hasattr(use_model, 'module') else use_model
        
        # [修改] 使用 linspace 生成均匀分布的时间步
        timesteps = np.linspace(0, self.diffusion.num_timesteps - 1, num_inference_steps, dtype=int)[::-1].copy()
        
        # Init Noise
        x = torch.randn(n, *size).to(self.device)
        
        C = x.shape[1]
        null_idx = getattr(self.config, 'num_classes', 1000)
        null_labels = torch.full_like(labels, null_idx, device=self.device)

        for i, t_step in enumerate(timesteps):
            t = torch.full((n,), t_step, device=self.device, dtype=torch.long)
            
            # Classifier-Free Guidance Input Prep
            if cfg_scale > 1.0:
                x_in = torch.cat([x, x])
                t_in = torch.cat([t, t])
                y_in = torch.cat([labels, null_labels])
            else:
                x_in, t_in, y_in = x, t, labels

            # Model Forward
            with torch.amp.autocast('cuda', dtype=self.dtype, enabled=self.use_amp):
                model_output = raw_model(x_in, t_in, y_in)
            
            # Always process sampling in float32 for precision
            model_output = model_output.float()
            
            # 如果模型输出了方差通道 (Learned Variance)，只取前半部分 epsilon
            if model_output.shape[1] == 2 * C:
                model_output, _ = torch.split(model_output, C, dim=1)
            
            eps = model_output
            
            # Apply CFG
            if cfg_scale > 1.0:
                cond_eps, uncond_eps = eps.chunk(2)
                eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            
            # --- DDIM Update Steps ---
            alpha_bar_t = self.diffusion.alphas_cumprod[t_step].to(self.device).float()
            
            # Calculate alpha_bar_t_prev
            if i == len(timesteps) - 1:
                alpha_bar_t_prev = torch.tensor(1.0, device=self.device).float()
            else:
                prev_t = timesteps[i+1]
                alpha_bar_t_prev = self.diffusion.alphas_cumprod[prev_t].to(self.device).float()
            
            sigma_t = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev))
            
            # 预测 x0 (Predicted x_start)
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            
            # [Latent vs Pixel Logic]
            # Pixel Space 通常需要 clip 到 [-1, 1] 以保持数值稳定
            # Latent Space 通常不需要 clip，或者范围不同
            if self.vae is None: 
                pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # 指向 xt 的方向
            dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * eps
            
            # Noise term (only if eta > 0)
            noise = sigma_t * torch.randn_like(x) if eta > 0 else 0.
            
            x = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + noise
        
        # [Final Step] Decode Latents to Pixels if using VAE
        final_image = self.decode_image_or_latent(x)
        return final_image

    @torch.no_grad()
    def sample(self, n, labels, size, cfg_scale=4.0, model=None):
        """
        标准 DDPM 采样 (Full steps). 
        """
        use_model = model if model is not None else self.model
        use_model.eval()
        raw_model = use_model.module if hasattr(use_model, 'module') else use_model

        x = torch.randn(n, *size).to(self.device)
        C = x.shape[1]
        null_idx = getattr(self.config, 'num_classes', 1000)
        null_labels = torch.full_like(labels, null_idx, device=self.device)

        # Iterate 999 -> 0
        for t_step in reversed(range(self.diffusion.num_timesteps)):
            t = torch.full((n,), t_step, device=self.device, dtype=torch.long)
            
            if cfg_scale > 1.0:
                x_in = torch.cat([x, x])
                t_in = torch.cat([t, t])
                y_in = torch.cat([labels, null_labels])
            else:
                x_in, t_in, y_in = x, t, labels

            with torch.amp.autocast('cuda', dtype=self.dtype, enabled=self.use_amp):
                model_output = raw_model(x_in, t_in, y_in)
            
            model_output = model_output.float()
            
            if model_output.shape[1] == 2 * C:
                model_output, _ = torch.split(model_output, C, dim=1)
                
            eps = model_output
            
            if cfg_scale > 1.0:
                cond_eps, uncond_eps = eps.chunk(2)
                eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            
            # Get alpha, beta terms for this timestep
            beta_t = self.diffusion.betas[t_step].to(self.device).float()
            alpha_t = 1.0 - beta_t
            alpha_bar_t = self.diffusion.alphas_cumprod[t_step].to(self.device).float()
            
            # Calculate posterior mean
            pred_mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps)
            
            if t_step > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t) 
                x = pred_mean + sigma_t * noise
            else:
                x = pred_mean
                
        # Decode
        final_image = self.decode_image_or_latent(x)
        return final_image

    @torch.no_grad()
    def visualize(self, epoch):
        """
        可视化采样函数
        """
        if self.config.local_rank != 0: return
        
        self.ema.apply_shadow()
        self.model.eval()
        try:
            print(f"[Visual] Generating samples for Epoch {epoch}...")
            n_samples = 4
            labels = torch.randint(0, self.num_classes, (n_samples,), device=self.device)
            in_channels = getattr(self.config, 'in_channels', 4)
            input_size = getattr(self.config, 'input_size', 32)
            # 这里指模型输入的 shape，可能是 latent 也可能是 pixel
            size = (in_channels, input_size, input_size)

            # 调用 sample_ddim，它内部会自动处理 decode_image_or_latent，返回 [0, 1] 的 Tensor
            x_vis = self.sample_ddim(n_samples, labels, size, cfg_scale=4.0, model=self.model)
            
            save_path = os.path.join(self.config.results_dir, f"sample_epoch_{epoch}.png")
            save_image(x_vis, save_path, nrow=2)
            print(f"[Visual] Saved to {save_path}")
        finally:
            self.ema.restore()
            self.model.train()

    @torch.no_grad()
    def evaluate_fid(self, epoch, num_gen_batches=10):
        """
        FID 评估函数
        """
        if self.fid_metric is None:
            return
        if self.config.use_ddp:
            dist.barrier()
        
        print(f"[FID] Starting evaluation for Epoch {epoch}...")
        self.ema.apply_shadow()
        self.model.eval()
        self.fid_metric.reset()
        
        torch.cuda.empty_cache()
        
        # 优先使用验证集
        loader_to_use = self.val_loader if self.val_loader is not None else self.loader
        loader_iter = iter(loader_to_use)

        try:
            # 1. 计算真实图片特征 (Real Statistics)
            for i in range(num_gen_batches):
                try:
                    real_imgs, _ = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader_to_use)
                    real_imgs, _ = next(loader_iter)
                    
                real_imgs = real_imgs.to(self.device)
                
                # 兼容性处理：如果输入是 Latent (4通道)，需要解码；如果是 Pixel (3通道)，需要反归一化
                # 使用 decode_image_or_latent 的逻辑，但要注意它假设输入是 VAE 编码后的 latent
                
                if self.vae is not None and real_imgs.shape[1] == getattr(self.config, 'in_channels', 4):
                     # 如果有 VAE 且通道数匹配模型输入（通常是 Latent），则解码
                     real_imgs = self.decode_image_or_latent(real_imgs)
                else:
                     # 否则假设是 Pixel Space [-1, 1]，直接转 [0, 1]
                     real_imgs = (real_imgs.float() + 1.0) / 2.0
                     real_imgs = real_imgs.clamp(0, 1)

                real_imgs_uint8 = (real_imgs * 255.0).to(torch.uint8)
                self.fid_metric.update(real_imgs_uint8, real=True)
            
            # 2. 计算生成图片特征 (Fake Statistics)
            for i in range(num_gen_batches):
                n_samples = self.config.batch_size
                labels = torch.randint(0, self.num_classes, (n_samples,), device=self.device)
                in_channels = getattr(self.config, 'in_channels', 4)
                input_size = getattr(self.config, 'input_size', 32)
                size = (in_channels, input_size, input_size)
                
                # sample_ddim 已经返回解码后的 [0, 1] 图像
                fake_imgs = self.sample_ddim(n_samples, labels, size, num_inference_steps=50, cfg_scale=4.0, model=self.model)
                
                fake_imgs_uint8 = (fake_imgs * 255.0).to(torch.uint8)
                self.fid_metric.update(fake_imgs_uint8, real=False)
            
            fid_score = self.fid_metric.compute()
            if self.config.local_rank == 0:
                print(f"[FID] Epoch {epoch} | FID Score: {fid_score.item():.4f}")
                with open(os.path.join(self.config.results_dir, "fid_log.txt"), "a") as f:
                    f.write(f"Epoch {epoch}, FID: {fid_score.item():.4f}\n")
        finally:
            self.ema.restore()
            self.model.train()
            torch.cuda.empty_cache()
        if self.config.use_ddp:
            dist.barrier()

    def train_one_epoch(self, epoch):
        self.model.train()
        # VAE Always Eval
        if self.vae: self.vae.eval()
        
        start_time = time()
        max_steps = getattr(self.config, 'max_train_steps', None)
        
        running_loss = torch.tensor(0.0, device=self.device)
        log_steps = 0
        
        for step, (images, labels) in enumerate(self.loader):
            if max_steps is not None and step >= max_steps: break

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # --- Pixel vs Latent Handling ---
            with torch.no_grad(): 
                # Case 1: VAE exists and input is raw pixels (3 channels)
                if self.vae is not None and images.shape[1] == 3:
                    posterior = self.vae.encode(images).latent_dist
                    # Latent Sampling with scaling factor
                    latents = posterior.sample() * 0.18215
                # Case 2: No VAE (Pixel Diffusion) or Data Loader yields latents
                else:
                    latents = images
                    # If dealing with pixel diffusion, ensure scaling is [-1, 1] if not already
                    # 如果数据加载器没有做 normalize，这里可能需要做
            
            t = torch.randint(0, self.diffusion.num_timesteps, (latents.shape[0],), device=self.device)
            
            # Label Dropout (CFG Training)
            if self.label_dropout_prob > 0:
                mask = torch.rand(labels.shape, device=self.device) < self.label_dropout_prob
                labels = torch.where(mask, torch.tensor(self.num_classes, device=self.device), labels)

            # Forward & Backward
            with torch.amp.autocast('cuda', dtype=self.dtype, enabled=self.use_amp):
                # Ensure model inputs match latent shape
                loss_dict = self.diffusion.training_losses(self.model, latents, t, model_kwargs=dict(y=labels))
                loss = loss_dict["loss"].mean()
            
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if step % self.ema_update_every == 0:
                self.ema.update()
            
            # Async loss recording
            running_loss += loss.detach()
            log_steps += 1
            
            if step % 100 == 0:
                avg_loss = running_loss.item() / log_steps
                
                if self.config.local_rank == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch} | Step {step} | Loss: {avg_loss:.4f} | GNorm: {grad_norm:.2f} | LR: {current_lr:.2e} | Time: {time() - start_time:.2f}s")
                
                running_loss = torch.tensor(0.0, device=self.device)
                log_steps = 0
                start_time = time()

        if self.config.local_rank == 0:
            viz_interval = getattr(self.config, 'log_interval', 1)
            # Visualize / Checkpoints
            if epoch % viz_interval == 0:
                self.visualize(epoch)
            
            # FID Evaluation
            if epoch > 0 and epoch % 10 == 0:
                 self.evaluate_fid(epoch)

        return None