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
    def __init__(self, model, diffusion, vae, loader, val_loader, config):
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
        self.val_loader = val_loader
        
        self.use_amp = getattr(config, 'use_amp', True)
        self.dtype = torch.bfloat16 if self.use_amp else torch.float32

        # GradScaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # [Added] 用于记录训练过程中的 loss
        self.train_loss_history = []

        if config.local_rank == 0:
            print(f"Training with AMP: {self.use_amp}, Dtype: {self.dtype}")
            print(f"EMA initialized with decay: {self.ema.decay}")
            if self.val_loader is not None:
                print("Validation Loader provided. FID will be calculated on VALIDATION set.")
            else:
                print("WARNING: No Validation Loader provided. FID will be calculated on TRAINING set.")
        
        try:
            self.model = torch.compile(self.model, mode="default")
            if config.local_rank == 0:
                print("Model compiled with torch.compile")
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")

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
        
        # --- [关键回归] 恢复旧版本的时间步策略 (从 980 开始，避开 999) ---
        step_ratio = self.diffusion.num_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(int)
        
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
                x_in = x
                t_in = t
                y_in = labels

            with torch.amp.autocast('cuda', dtype=self.dtype, enabled=self.use_amp):
                model_output = raw_model(x_in, t_in, y_in)
            
            # --- [关键回归] 严格 FP32 计算 ---
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
            
            # --- [关键回归] 移除截断 (No Clamp) ---
            # 因为采样避开了 t=999，这里数值稳定，无需截断，保留高频细节
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            
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
            # --- 1. Sanity Check (从 val_loader 取) ---
            try:
                loader_to_use = self.val_loader if self.val_loader is not None else self.loader
                real_batch, _ = next(iter(loader_to_use))
                real_batch = real_batch[:4].to(self.device)
                
                if real_batch.shape[1] == 4:
                    recon_real = self.vae.decode(real_batch.float()).sample.float()
                else:
                    post = self.vae.encode(real_batch.float()).latent_dist
                    recon_real = self.vae.decode(post.sample()).sample.float()
                
                recon_real = recon_real.clamp(-1, 1)
                recon_vis = (recon_real + 1.0) / 2.0
                
                recon_path = os.path.join(self.config.results_dir, f"recon_sanity_epoch_{epoch}.png")
                save_image(recon_vis, recon_path, nrow=2)
            except Exception as e:
                print(f"[Visual] Reconstruction Sanity Check Failed: {e}")

            # --- 2. Generate Samples ---
            n_samples = 4
            labels = torch.randint(0, 1000, (n_samples,), device=self.device)
            in_channels = getattr(self.config, 'in_channels', 4)
            input_size = getattr(self.config, 'input_size', 32)
            latent_size = (in_channels, input_size, input_size)

            # 使用 cfg_scale=4.0
            z = self.sample_ddim(n_samples, labels, latent_size, num_inference_steps=50, cfg_scale=4.0, model=self.model)
            z = z.float()
            
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
        
        torch.cuda.empty_cache()
        vae_batch_size = 32

        # 优先使用验证集
        loader_to_use = self.val_loader if self.val_loader is not None else self.loader
        loader_iter = iter(loader_to_use)

        try:
            for i in range(num_gen_batches):
                try:
                    real_imgs, _ = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader_to_use)
                    real_imgs, _ = next(loader_iter)
                    
                real_imgs = real_imgs.to(self.device)
                
                # --- Real Images Decoding (Sliced) ---
                if real_imgs.shape[1] == 4:
                    real_imgs = real_imgs.float()
                    decoded_list = []
                    for k in range(0, real_imgs.shape[0], vae_batch_size):
                        batch_slice = real_imgs[k : k + vae_batch_size]
                        decoded_slice = self.vae.decode(batch_slice).sample.float()
                        decoded_list.append(decoded_slice)
                    real_imgs = torch.cat(decoded_list, dim=0)
                
                real_imgs = real_imgs.clamp(-1, 1)
                real_imgs_uint8 = self._normalize_images(real_imgs)
                self.fid_metric.update(real_imgs_uint8, real=True)
            
            for i in range(num_gen_batches):
                n_samples = self.config.batch_size
                labels = torch.randint(0, 1000, (n_samples,), device=self.device)
                in_channels = getattr(self.config, 'in_channels', 4)
                input_size = getattr(self.config, 'input_size', 32)
                latent_size = (in_channels, input_size, input_size)
                
                # 采样使用 cfg=4.0 和 无截断策略
                z = self.sample_ddim(n_samples, labels, latent_size, num_inference_steps=50, cfg_scale=4.0, model=self.model)
                z = z.float()
                
                decoded_list = []
                for k in range(0, z.shape[0], vae_batch_size):
                    z_slice = z[k : k + vae_batch_size]
                    decoded_slice = self.vae.decode(z_slice / 0.18215).sample.float()
                    decoded_list.append(decoded_slice)
                
                fake_imgs = torch.cat(decoded_list, dim=0)
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
        
        # [Added] 用于累计 epoch 平均 loss
        epoch_loss = 0.0
        steps_count = 0
        
        for step, (images, labels) in enumerate(self.loader):
            if max_steps is not None and step >= max_steps:
                break

            images = images.to(self.device)
            labels = labels.to(self.device)
            
            if images.shape[1] == 3:
                with torch.no_grad():
                    posterior = self.vae.encode(images).latent_dist
                    latents = posterior.sample() * 0.18215
            else:
                latents = images * 0.18215
            
            # --- [关键修改] 防止 Loss 震荡：不采样最后的 10 个时间步 ---
            # 避开 t >= 990 (假设 num_timesteps=1000)
            # 这些步全是噪声，容易导致梯度不稳
            t = torch.randint(0, self.diffusion.num_timesteps - 10, (images.shape[0],), device=self.device)
            
            loss_dict = self.diffusion.training_losses(self.model, latents, t, model_kwargs=dict(y=labels))
            loss = loss_dict["loss"].mean()
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.ema.update()
            
            # [Added] 记录 step loss
            epoch_loss += loss.item()
            steps_count += 1
            
            if step % 100 == 0 and self.config.local_rank == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | Time: {time() - start_time:.2f}s")
                start_time = time()

        if self.config.local_rank == 0:
            viz_interval = getattr(self.config, 'log_interval', 1)
            if epoch % viz_interval == 0:
                self.visualize(epoch)
            
        if epoch > 0 and epoch % 10 == 0:
             self.evaluate_fid(epoch, num_gen_batches=10)
             
        # [Added] 返回平均 loss
        avg_loss = epoch_loss / steps_count if steps_count > 0 else 0.0
        return avg_loss

    def save_checkpoint(self, epoch, avg_loss=None):
        # [Modified] 接收 avg_loss 参数
        if self.config.local_rank == 0:
            checkpoint_path = os.path.join(self.config.results_dir, f"checkpoint_{epoch}.pt")
            
            # 更新 loss 历史
            if avg_loss is not None:
                self.train_loss_history.append((epoch, avg_loss))
            
            checkpoint = {
                "model": self.model.module.state_dict() if self.config.use_ddp else self.model.state_dict(),
                "ema": self.ema.shadow, 
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
                "config": self.config,
                # [Added] 保存 loss 信息
                "train_loss": avg_loss,
                "loss_history": self.train_loss_history,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path} (Loss: {avg_loss:.4f})")
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
            
        # [Added] 恢复 loss 历史
        if "loss_history" in checkpoint:
            self.train_loss_history = checkpoint["loss_history"]
            print(f"Restored loss history: {len(self.train_loss_history)} epochs")
            
        start_epoch = checkpoint.get("epoch", -1) + 1
        print(f"Resuming training from epoch {start_epoch}")
        return start_epoch