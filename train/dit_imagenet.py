import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from time import time
from torchvision.utils import save_image, make_grid

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
        self.dtype = torch.bfloat16
        
        # --- FID 评估指标初始化 ---
        self.fid_metric = None
        if self.config.local_rank == 0:
            try:
                from torchmetrics.image.fid import FrechetInceptionDistance
                # feature=2048 是标准 InceptionV3 特征层
                self.fid_metric = FrechetInceptionDistance(feature=2048).to(self.device)
                print("FID metric initialized successfully.")
            except ImportError:
                print("Warning: torchmetrics not found. FID evaluation will be skipped.")

    @torch.no_grad()
    def sample_ddpm(self, n, labels, size):
        self.model.eval()
        
        # 关键修正：推理时如果是 DDP 模型，解包取出原始 module
        # 避免 DDP wrapper 在单卡推理时产生的额外开销或同步问题
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        
        x = torch.randn(n, *size).to(self.device)
        
        # 你的 Diffusion loop
        for i in reversed(range(self.diffusion.num_timesteps)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            
            # 使用 raw_model 进行前向
            model_output = raw_model(x, t, labels)
            
            if model_output.shape[1] == 2 * x.shape[1]:
                model_output, _ = model_output.chunk(2, dim=1)
            
            beta = self.diffusion.betas[i]
            alpha = self.diffusion.alphas[i]
            sqrt_one_minus_alpha_cumprod = self.diffusion.sqrt_one_minus_alphas_cumprod[i]
            
            coeff = beta / sqrt_one_minus_alpha_cumprod
            mean = (1 / torch.sqrt(alpha)) * (x - coeff * model_output)
            
            if i > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta)
                x = mean + sigma * noise
            else:
                x = mean
                
        return x

    @torch.no_grad()
    def visualize(self, epoch):
        self.model.eval()
        n_samples = 4
        labels = torch.randint(0, 1000, (n_samples,), device=self.device)
        
        # 获取 latent 尺寸
        in_channels = getattr(self.config, 'in_channels', 4)
        input_size = getattr(self.config, 'input_size', 32)
        latent_size = (in_channels, input_size, input_size)

        z = self.sample_ddpm(n_samples, labels, latent_size)
        x_recon = self.vae.decode(z / 0.18215).sample
        x_recon = torch.clamp((x_recon + 1.0) / 2.0, 0.0, 1.0)
        
        save_path = os.path.join(self.config.results_dir, f"sample_epoch_{epoch}.png")
        save_image(x_recon, save_path, nrow=2)
        print(f"[Visual] Saved visualization to {save_path}")
        
        self.model.train()

    @torch.no_grad()
    def evaluate_fid(self, epoch, num_batches=5):
        if self.fid_metric is None:
            return

        print(f"[FID] Starting evaluation for epoch {epoch}...")
        self.model.eval()
        self.fid_metric.reset()

        # 1. 收集真实图片
        for i, (real_imgs, _) in enumerate(self.loader):
            if i >= num_batches: break
            real_imgs = real_imgs.to(self.device)
            real_imgs_uint8 = ((real_imgs + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            self.fid_metric.update(real_imgs_uint8, real=True)
        
        # 2. 收集生成图片
        # 注意：num_batches * batch_size * 1000步 采样非常耗时，可能导致超时
        # 建议 num_batches 设小一点，或者在 init_process_group 时调大 timeout
        for _ in range(num_batches):
            n_samples = self.config.batch_size
            labels = torch.randint(0, 1000, (n_samples,), device=self.device)
            
            in_channels = getattr(self.config, 'in_channels', 4)
            input_size = getattr(self.config, 'input_size', 32)
            latent_size = (in_channels, input_size, input_size)
            
            z = self.sample_ddpm(n_samples, labels, latent_size)
            fake_imgs = self.vae.decode(z / 0.18215).sample
            fake_imgs_uint8 = ((fake_imgs + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            self.fid_metric.update(fake_imgs_uint8, real=False)
            
        fid_score = self.fid_metric.compute()
        print(f"[FID] Epoch {epoch} | FID Score: {fid_score.item():.4f}")
        
        with open(os.path.join(self.config.results_dir, "fid_log.txt"), "a") as f:
            f.write(f"Epoch {epoch}, FID: {fid_score.item():.4f}\n")

        self.model.train()

    def train_one_epoch(self, epoch):
        self.model.train()
        start_time = time()
        
        # 获取最大步数限制 (用于快速测试)
        max_steps = getattr(self.config, 'max_train_steps', None)
        
        for step, (images, labels) in enumerate(self.loader):
            # 1. 快速测试截断逻辑
            if max_steps is not None and step >= max_steps:
                if self.config.local_rank == 0:
                    print(f"Debug: Max steps {max_steps} reached, breaking epoch.")
                break

            images = images.to(self.device)
            labels = labels.to(self.device)
            
            with torch.no_grad():
                # 修复: 变量名 dist -> posterior，避免覆盖 torch.distributed
                posterior = self.vae.encode(images).latent_dist
                latents = posterior.sample() * 0.18215
                
            t = torch.randint(0, self.diffusion.num_timesteps, (images.shape[0],), device=self.device)
            
            with torch.amp.autocast('cuda', dtype=self.dtype):
                loss = self.diffusion.p_losses(self.model, latents, t, labels)
                
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            if step % 100 == 0 and self.config.local_rank == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | Time: {time() - start_time:.2f}s")
                start_time = time()

        # --- Epoch 结束逻辑 ---
        
        # 1. Rank 0 执行评估
        if self.config.local_rank == 0:
            viz_interval = getattr(self.config, 'log_interval', 1)
            if epoch % viz_interval == 0:
                self.visualize(epoch)
            
            # FID 评估频率
            if epoch > 0 and epoch % 5 == 0:
                self.evaluate_fid(epoch, num_batches=2) # 减少 batch 数防止超时
                
        # 2. 关键修正：同步屏障
        # 强制 Rank 1-3 等待 Rank 0 完成评估工作，防止它们提前进入下一个 Epoch 导致死锁
        if self.config.use_ddp:
            dist.barrier()

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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
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