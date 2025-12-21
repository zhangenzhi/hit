import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import numpy as np
from time import time
from copy import deepcopy
import math

from train.utilz import EMA, visualize, evaluate_fid, get_cosine_schedule_with_warmup

class DiTImangenetTrainer:
    def __init__(self, model, diffusion, vae, loader, val_loader, config):
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        
        self.model = model.to(self.device)
        self.vae = vae.to(self.device).eval() 
        self.diffusion = diffusion
        
        # DDP setup
        if config.use_ddp:
            self.model = DDP(
                self.model, 
                device_ids=[config.local_rank],
                gradient_as_bucket_view=True, 
                static_graph=getattr(config, 'static_graph', True) 
            )
            
        # [Critical Fix] 建议将 ema_update_every 设为 1 以避免 EMA 滞后
        self.ema_update_every = getattr(config, 'ema_update_every', 1)
        self.ema = EMA(self.model, decay=0.9999)
            
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.lr, 
            weight_decay=0.0
        )
        self.loader = loader
        self.val_loader = val_loader
        
        # --- 初始化 Learning Rate Scheduler (Cosine + Warmup) ---
        # 计算总步数
        total_epochs = getattr(config, 'epochs', 100)
        steps_per_epoch = len(self.loader)
        num_training_steps = total_epochs * steps_per_epoch
        
        # 默认 Warmup 为总 Epoch 的 10% (例如 400 epochs -> 40 epochs warmup)
        # 也可以在 config 中指定 warmup_epochs
        warmup_epochs = getattr(config, 'warmup_epochs', int(total_epochs * 0.1))
        num_warmup_steps = warmup_epochs * steps_per_epoch
        
        if config.local_rank == 0:
            print(f"[Scheduler] Initializing Cosine Decay with Warmup")
            print(f"  - Total Steps: {num_training_steps}")
            print(f"  - Warmup Steps: {num_warmup_steps} ({warmup_epochs} epochs)")
            
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=num_training_steps
        )
        # --------------------------------------------------------

        self.use_amp = getattr(config, 'use_amp', True)
        self.dtype = torch.bfloat16 if self.use_amp else torch.float32

        # BF16 dynamic range is usually large enough, so Scaler is often not needed
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
        
        try:
            compile_mode = getattr(config, 'compile_mode', 'max-autotune') 
            self.model = torch.compile(self.model, mode=compile_mode)
            if config.local_rank == 0:
                print(f"Model compiled with torch.compile (mode={compile_mode})")
        except Exception as e:
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

    @torch.no_grad()
    def sample_ddim(self, n, labels, size, num_inference_steps=50, eta=0.0, cfg_scale=4.0, model=None):
        use_model = model if model is not None else self.model
        use_model.eval()
        raw_model = use_model.module if hasattr(use_model, 'module') else use_model
        
        step_ratio = self.diffusion.num_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(int)
        
        x = torch.randn(n, *size).to(self.device)
        C = x.shape[1]
        null_idx = getattr(self.config, 'num_classes', 1000)
        null_labels = torch.full_like(labels, null_idx, device=self.device)

        for i, t_step in enumerate(timesteps):
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
            eps = model_output[:, :C]
            
            if cfg_scale > 1.0:
                cond_eps, uncond_eps = eps.chunk(2)
                eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            
            alpha_bar_t = self.diffusion.alphas_cumprod[t_step].to(self.device).float()
            if i == len(timesteps) - 1:
                alpha_bar_t_prev = torch.tensor(1.0, device=self.device).float()
            else:
                prev_t = timesteps[i+1]
                alpha_bar_t_prev = self.diffusion.alphas_cumprod[prev_t].to(self.device).float()
            
            sigma_t = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev))
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * eps
            noise = sigma_t * torch.randn_like(x)
            x = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + noise
            
        return x

    def train_one_epoch(self, epoch):
        self.model.train()
        self.vae.eval()
        
        start_time = time()
        max_steps = getattr(self.config, 'max_train_steps', None)
        
        running_loss = torch.tensor(0.0, device=self.device)
        log_steps = 0
        
        for step, (images, labels) in enumerate(self.loader):
            if max_steps is not None and step >= max_steps: break

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # VAE Encode (No Grads)
            with torch.no_grad(): 
                if images.shape[1] == 3:
                    posterior = self.vae.encode(images).latent_dist
                    latents = posterior.sample() * 0.18215
                else:
                    latents = images * 0.18215
                
                # latents = latents.clamp(-4.0, 4.0)
            
            t = torch.randint(0, self.diffusion.num_timesteps, (images.shape[0],), device=self.device)
            
            # Label Dropout (CFG Training)
            if self.label_dropout_prob > 0:
                mask = torch.rand(labels.shape, device=self.device) < self.label_dropout_prob
                labels = torch.where(mask, torch.tensor(self.num_classes, device=self.device), labels)

            # Forward & Backward
            with torch.amp.autocast('cuda', dtype=self.dtype, enabled=self.use_amp):
                loss_dict = self.diffusion.training_losses(self.model, latents, t, model_kwargs=dict(y=labels))
                loss = loss_dict["loss"].mean()
            
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # [新增] 更新 Learning Rate Scheduler
            self.scheduler.step()
            
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
            if epoch % viz_interval == 0:
                visualize(self, epoch)
            
        if epoch > 0 and epoch % 10 == 0:
             evaluate_fid(self, epoch)

        return None