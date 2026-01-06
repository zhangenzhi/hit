import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
import os
import numpy as np
from time import time

# from train.utilz import EMA

class DiTImangenetTrainer:
    def __init__(self, model, diffusion, vae, loader, val_loader, config):
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        
        self.model = model.to(self.device)
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
            
        # self.ema_update_every = getattr(config, 'ema_update_every', 1)
        # self.ema = EMA(self.model, decay=0.9999)
            
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.lr, 
            weight_decay=0.0
        )
        self.loader = loader
        self.val_loader = val_loader

        self.use_amp = getattr(config, 'use_amp', True)
        self.dtype = torch.bfloat16 if self.use_amp else torch.float32

        use_scaler = self.use_amp and (self.dtype == torch.float16)
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
        
        self.train_loss_history = []
        
        self.num_classes = getattr(config, 'num_classes', 1000)
        self.label_dropout_prob = getattr(config, 'label_dropout_prob', 0.1)

        if config.local_rank == 0:
            print(f"Training with AMP: {self.use_amp}, Dtype: {self.dtype}")
            print(f"GradScaler Enabled: {use_scaler} (Disabled for BF16 recommended)")
            # print(f"EMA initialized with decay: {self.ema.decay}, Update Every: {self.ema_update_every} steps")
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
    def visualize(self, epoch):
        """
        可视化采样函数
        """
        if self.config.local_rank != 0: return
        
        # self.ema.apply_shadow()
        self.model.eval()
        try:
            print(f"[Visual] Generating samples for Epoch {epoch}...")
            n_samples = 4
            labels = torch.randint(0, self.num_classes, (n_samples,), device=self.device)
            in_channels = getattr(self.config, 'in_channels', 4)
            input_size = getattr(self.config, 'input_size', 32)
            size = (in_channels, input_size, input_size)

            # [Update] 可视化通常使用 DDIM 加速，但也支持配置
            viz_method = getattr(self.config, 'sample_method', 'ddim') 
            
            if viz_method == 'ddpm':
                z = self.diffusion.sample_ddpm(
                    model=self.model,
                    labels=labels,
                    size=size,
                    num_classes=self.num_classes,
                    cfg_scale=4.0,
                    use_amp=self.use_amp,
                    dtype=self.dtype,
                    is_latent=(self.vae is not None)
                )
            else:
                 z = self.diffusion.sample_ddim(
                    model=self.model, 
                    labels=labels, 
                    size=size,
                    num_classes=self.num_classes,
                    cfg_scale=4.0, 
                    use_amp=self.use_amp, 
                    dtype=self.dtype,
                    is_latent=(self.vae is not None)
                )
            
            x_vis = self.decode_image_or_latent(z)
            
            save_path = os.path.join(self.config.results_dir, f"sample_epoch_{epoch}.png")
            save_image(x_vis, save_path, nrow=2)
            print(f"[Visual] Saved to {save_path}")
        finally:
            # self.ema.restore()
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
        
        # [Config] 强制使用 DDPM (和 DiT 论文一致)
        sample_method = 'ddpm'
        
        print(f"[FID] Starting evaluation for Epoch {epoch} using {sample_method.upper()} sampling...")
        # self.ema.apply_shadow()
        self.model.eval()
        self.fid_metric.reset()
        
        torch.cuda.empty_cache()
        
        loader_to_use = self.val_loader if self.val_loader is not None else self.loader
        loader_iter = iter(loader_to_use)

        try:
            # 1. Real Images
            for i in range(num_gen_batches):
                try:
                    real_imgs, _ = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader_to_use)
                    real_imgs, _ = next(loader_iter)
                    
                real_imgs = real_imgs.to(self.device)
                
                if self.vae is not None and real_imgs.shape[1] == getattr(self.config, 'in_channels', 4):
                     real_imgs = self.decode_image_or_latent(real_imgs)
                else:
                     real_imgs = (real_imgs.float() + 1.0) / 2.0
                     real_imgs = real_imgs.clamp(0, 1)

                real_imgs_uint8 = (real_imgs * 255.0).to(torch.uint8)
                self.fid_metric.update(real_imgs_uint8, real=True)
            
            # 2. Fake Images
            for i in range(num_gen_batches):
                n_samples = self.config.batch_size
                labels = torch.randint(0, self.num_classes, (n_samples,), device=self.device)
                in_channels = getattr(self.config, 'in_channels', 4)
                input_size = getattr(self.config, 'input_size', 32)
                size = (in_channels, input_size, input_size)
                
                # [Update] Force DDPM
                z = self.diffusion.sample_ddpm(
                    model=self.model,
                    labels=labels,
                    size=size,
                    num_classes=self.num_classes,
                    cfg_scale=4.0,
                    use_amp=self.use_amp,
                    dtype=self.dtype,
                    is_latent=(self.vae is not None)
                )
                
                fake_imgs = self.decode_image_or_latent(z)
                
                fake_imgs_uint8 = (fake_imgs * 255.0).to(torch.uint8)
                self.fid_metric.update(fake_imgs_uint8, real=False)
            
            fid_score = self.fid_metric.compute()
            
            if self.config.local_rank == 0:
                print(f"[FID] Epoch {epoch} | FID Score: {fid_score.item():.4f} | Method: {sample_method.upper()}")
                with open(os.path.join(self.config.results_dir, "fid_log.txt"), "a") as f:
                    f.write(f"Epoch {epoch}, FID: {fid_score.item():.4f}, Method: {sample_method.upper()}\n")
        finally:
            # self.ema.restore()
            self.model.train()
            torch.cuda.empty_cache()


    def train_one_epoch(self, epoch):
        self.model.train()
        if self.vae: self.vae.eval()
        
        start_time = time()
        max_steps = getattr(self.config, 'max_train_steps', None)
        
        running_loss = torch.tensor(0.0, device=self.device)
        log_steps = 0
        
        for step, (images, labels) in enumerate(self.loader):
            if max_steps is not None and step >= max_steps: break

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            with torch.no_grad(): 
                if self.vae is not None and images.shape[1] == 3:
                    posterior = self.vae.encode(images).latent_dist
                    latents = posterior.sample() * 0.18215
                else:
                    latents = images * 0.18215
            
            t = torch.randint(0, self.diffusion.num_timesteps, (latents.shape[0],), device=self.device)
            
            # if self.label_dropout_prob > 0:
            #     mask = torch.rand(labels.shape, device=self.device) < self.label_dropout_prob
            #     labels = torch.where(mask, torch.tensor(self.num_classes, device=self.device), labels)

            with torch.amp.autocast('cuda', dtype=self.dtype, enabled=self.use_amp):
                loss_dict = self.diffusion.training_losses(self.model, latents, t, model_kwargs=dict(y=labels))
                loss = loss_dict["loss"].mean()
            
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # if step % self.ema_update_every == 0:
            #     self.ema.update()
            
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
                self.visualize(epoch)
            
            # if epoch > 0 and epoch % 10 == 0:
            #      self.evaluate_fid(epoch)

        return None