import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import numpy as np
from time import time
from torchvision.utils import save_image, make_grid
from copy import deepcopy

# --- 1. EMA 类 (保持不变) ---
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        model_to_track = self.model.module if isinstance(self.model, DDP) else self.model
        for name, param in model_to_track.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self):
        model_to_track = self.model.module if isinstance(self.model, DDP) else self.model
        for name, param in model_to_track.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        model_to_track = self.model.module if isinstance(self.model, DDP) else self.model
        for name, param in model_to_track.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        model_to_track = self.model.module if isinstance(self.model, DDP) else self.model
        for name, param in model_to_track.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class DiTImangenetTrainer:
    def __init__(self, model, diffusion, vae, loader, config):
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        
        self.model = model.to(self.device)
        self.vae = vae.to(self.device).eval() 
        self.diffusion = diffusion
        
        if config.use_ddp:
            self.model = DDP(self.model, device_ids=[config.local_rank])
            
        self.ema = EMA(self.model, decay=0.9999)
            
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.lr, 
            weight_decay=0.0
        )
        self.loader = loader
        
        self.use_amp = getattr(config, 'use_amp', True)
        self.dtype = torch.bfloat16 if self.use_amp else torch.float32

        # GradScaler: 推荐保留，增强鲁棒性
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        if config.local_rank == 0:
            print(f"Training with AMP: {self.use_amp}, Dtype: {self.dtype}")
            print(f"EMA initialized with decay: {self.ema.decay}")
        
        # 编译优化 (可根据需要开启)
        # try:
        #     self.model = torch.compile(self.model, mode="default")
        #     if config.local_rank == 0:
        #         print("Model compiled with torch.compile")
        # except Exception as e:
        #     print(f"Warning: torch.compile failed: {e}")

        self.fid_metric = None
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            self.fid_metric = FrechetInceptionDistance(feature=2048, reset_real_features=True).to(self.device)
            if self.config.local_rank == 0:
                print("FID metric initialized successfully.")
        except ImportError:
            if self.config.local_rank == 0:
                print("Warning: torchmetrics not found. FID evaluation will be skipped.")

    def _normalize_images(self, images):
        images = (images + 1.0) / 2.0
        images = images.clamp(0.0, 1.0)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        return (images * 255.0).to(torch.uint8)

    @torch.no_grad()
    def sample_ddim(self, n, labels, size, num_inference_steps=50, eta=0.0, cfg_scale=4.0, model=None):
        use_model = model if model is not None else self.model
        use_model.eval()
        raw_model = use_model.module if hasattr(use_model, 'module') else use_model
        
        timesteps = torch.linspace(0, self.diffusion.num_timesteps - 1, num_inference_steps, dtype=torch.long)
        timesteps = timesteps.flip(0).tolist()
        
        # Init noise (std=1)
        x = torch.randn(n, *size).to(self.device)
        C = x.shape[1]
        null_idx = getattr(self.config, 'num_classes', 1000)
        null_labels = torch.full_like(labels, null_idx, device=self.device)

        for i, t_step in enumerate(timesteps):
            t = torch.full((n,), t_step, device=self.device, dtype=torch.long)
            
            # CFG
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
                alpha_bar_t_prev = torch.tensor(1.0, device=self.device)
            else:
                prev_t = timesteps[i+1]
                alpha_bar_t_prev = self.diffusion.alphas_cumprod[prev_t].to(self.device).float()
            
            sigma_t = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev))
            
            # Predict x0 (in SCALED space)
            # [WARNING] 当 t 很大(alpha_bar_t很小)时，此处除法会导致数值爆炸，尤其是在模型未训练好时
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            
            # [FIX] 对 pred_x0 进行截断。Latent (std=1) 的正常范围在 [-3, 3] 左右。
            # 这能有效防止 Epoch 0 时的数值爆炸 (你的 Std: 19708 就是这里导致的)。
            pred_x0 = pred_x0.clamp(-3.0, 3.0)
            
            dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * eps
            noise = sigma_t * torch.randn_like(x)
            x = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + noise
            
        return x

    @torch.no_grad()
    def visualize(self, epoch):
        if self.config.local_rank != 0:
            return

        self.ema.apply_shadow()
        self.model.eval()
        
        try:
            # --- 1. Sanity Check: Visualize REAL Reconstruction ---
            # 这步非常关键：如果不通过，说明是 VAE 或者是 Scaling 问题
            try:
                real_batch, _ = next(iter(self.loader))
                real_batch = real_batch[:4].to(self.device)
                
                # 如果是 Latent (4通道)，直接解码
                if real_batch.shape[1] == 4:
                    # Real Latents (std=5.5) -> Image
                    # 注意：这里不需要 / 0.18215，因为 Dataset 返回的就是 Unscaled
                    recon_real = self.vae.decode(real_batch.float()).sample.float()
                else:
                    # 如果是 Pixel，先 Encode 再 Decode
                    post = self.vae.encode(real_batch.float()).latent_dist
                    recon_real = self.vae.decode(post.sample()).sample.float()
                
                recon_real = recon_real.clamp(-1, 1)
                recon_vis = (recon_real + 1.0) / 2.0
                
                recon_path = os.path.join(self.config.results_dir, f"recon_sanity_epoch_{epoch}.png")
                save_image(recon_vis, recon_path, nrow=2)
                print(f"[Visual] Saved REAL reconstruction to {recon_path} (Check this if generated images are wrong!)")
            except Exception as e:
                print(f"[Visual] Reconstruction Sanity Check Failed: {e}")

            # --- 2. Generate Samples ---
            n_samples = 4
            labels = torch.randint(0, 1000, (n_samples,), device=self.device)
            in_channels = getattr(self.config, 'in_channels', 4)
            input_size = getattr(self.config, 'input_size', 32)
            latent_size = (in_channels, input_size, input_size)

            z = self.sample_ddim(n_samples, labels, latent_size, num_inference_steps=50, cfg_scale=1.5, model=self.model)
            z = z.float()
            
            # 打印生成 Latent 的统计信息，确保它不是全是 0 或者全是 NaN
            print(f"[Visual] Generated Latent Stats | Mean: {z.mean():.4f} | Std: {z.std():.4f} | Min: {z.min():.4f} | Max: {z.max():.4f}")
            
            # Correct Unscaling: std=1 -> std=5.5
            x_recon = self.vae.decode(z / 0.18215).sample.float()
            
            x_recon = x_recon.clamp(-1, 1)
            x_vis = (x_recon + 1.0) / 2.0
            x_vis = x_vis.clamp(0.0, 1.0)
            
            save_path = os.path.join(self.config.results_dir, f"sample_epoch_{epoch}.png")
            save_image(x_vis, save_path, nrow=2)
            print(f"[Visual] Saved visualization to {save_path}")
        finally:
            self.ema.restore()
            self.model.train()

    @torch.no_grad()
    def evaluate_fid(self, epoch, num_gen_batches=10):
        if self.fid_metric is None:
            return
        if self.config.use_ddp:
            dist.barrier()
        
        self.ema.apply_shadow()
        self.model.eval()
        self.fid_metric.reset()

        try:
            loader_iter = iter(self.loader)
            for i in range(num_gen_batches):
                try:
                    real_imgs, _ = next(loader_iter)
                except StopIteration:
                    break
                real_imgs = real_imgs.to(self.device)
                if real_imgs.shape[1] == 4:
                    real_imgs = real_imgs.float()
                    # Real Latents (std=5.5) -> Image
                    real_imgs = self.vae.decode(real_imgs).sample.float()
                real_imgs = real_imgs.clamp(-1, 1)
                real_imgs_uint8 = self._normalize_images(real_imgs)
                self.fid_metric.update(real_imgs_uint8, real=True)
            
            for i in range(num_gen_batches):
                n_samples = self.config.batch_size
                labels = torch.randint(0, 1000, (n_samples,), device=self.device)
                in_channels = getattr(self.config, 'in_channels', 4)
                input_size = getattr(self.config, 'input_size', 32)
                latent_size = (in_channels, input_size, input_size)
                
                z = self.sample_ddim(n_samples, labels, latent_size, num_inference_steps=50, cfg_scale=1.5, model=self.model)
                z = z.float()
                
                # Fake Latents (std=1) -> Unscaled (std=5.5) -> Image
                fake_imgs = self.vae.decode(z / 0.18215).sample.float()
                fake_imgs = fake_imgs.clamp(-1, 1)
                
                fake_imgs_uint8 = self._normalize_images(fake_imgs)
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
        start_time = time()
        max_steps = getattr(self.config, 'max_train_steps', None)
        
        def mp_model_wrapper(*args, **kwargs):
            with torch.amp.autocast('cuda', dtype=self.dtype, enabled=self.use_amp):
                return self.model(*args, **kwargs)
        
        for step, (images, labels) in enumerate(self.loader):
            if max_steps is not None and step >= max_steps:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # --- Latent Scaling Logic ---
            if images.shape[1] == 3:
                with torch.no_grad():
                    posterior = self.vae.encode(images).latent_dist
                    latents = posterior.sample() * 0.18215
            else:
                # Disk latents (Unscaled std~5.5) -> Scaled (std~1.0)
                latents = images * 0.18215
            
            t = torch.randint(0, self.diffusion.num_timesteps, (images.shape[0],), device=self.device)
            
            loss_dict = self.diffusion.training_losses(mp_model_wrapper, latents, t, model_kwargs=dict(y=labels))
            loss = loss_dict["loss"].mean()
            
            self.optimizer.zero_grad()
            
            # Scaler Logic
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.ema.update()
            
            if step % 100 == 0 and self.config.local_rank == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | Time: {time() - start_time:.2f}s")
                start_time = time()

        if self.config.local_rank == 0:
            viz_interval = getattr(self.config, 'log_interval', 1)
            if epoch % viz_interval == 0:
                self.visualize(epoch)
            
        if epoch > 0 and epoch % 10 == 0:
             self.evaluate_fid(epoch, num_gen_batches=10)

    def save_checkpoint(self, epoch):
        if self.config.local_rank == 0:
            checkpoint_path = os.path.join(self.config.results_dir, f"checkpoint_{epoch}.pt")
            
            checkpoint = {
                "model": self.model.module.state_dict() if self.config.use_ddp else self.model.state_dict(),
                "ema": self.ema.shadow, 
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model_state_dict = checkpoint["model"]
        if self.config.use_ddp:
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)
            
        if "ema" in checkpoint:
            self.ema.shadow = checkpoint["ema"]
            print("EMA state loaded.")
        else:
            print("Warning: No EMA state found in checkpoint. Initializing from current model.")
            self.ema.register()

        if "optimizer" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            
        start_epoch = checkpoint.get("epoch", -1) + 1
        print(f"Resuming training from epoch {start_epoch}")
        return start_epoch