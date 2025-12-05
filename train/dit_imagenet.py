import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import numpy as np  # 新增: 用于 DDIM 时间步计算
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
        
        # --- FID 评估指标初始化 (所有 Rank 都初始化) ---
        self.fid_metric = None
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            # feature=2048 是标准 InceptionV3 特征层
            # reset_real_features=False 允许在一个 Epoch 内累积真实图片统计量
            self.fid_metric = FrechetInceptionDistance(feature=2048, reset_real_features=False).to(self.device)
            if self.config.local_rank == 0:
                print("FID metric initialized successfully (Distributed Mode).")
        except ImportError:
            if self.config.local_rank == 0:
                print("Warning: torchmetrics not found. FID evaluation will be skipped.")

    @torch.no_grad()
    def sample_ddim(self, n, labels, size, num_inference_steps=50, eta=0.0):
        """
        DDIM 快速采样 (50步 vs DDPM 1000步)
        """
        self.model.eval()
        raw_model = self.model.module if isinstance(self.model, DDP) else self.model
        
        # 1. 构造时间步序列 (e.g., [980, 960, ... 0])
        # 使用简单的均匀间隔采样
        step_ratio = self.diffusion.num_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(int)
        
        # 初始噪声
        x = torch.randn(n, *size).to(self.device)
        
        # 2. DDIM 循环
        for i, t_step in enumerate(timesteps):
            # 构造当前时间步 tensor
            t = torch.full((n,), t_step, device=self.device, dtype=torch.long)
            
            # 2.1 预测噪声 epsilon_theta
            model_output = raw_model(x, t, labels)
            if model_output.shape[1] == 2 * x.shape[1]:
                model_output, _ = model_output.chunk(2, dim=1)
            
            # 2.2 获取 alpha 参数
            # 注意: self.diffusion.alphas_cumprod 是在 __init__ 里定义的张量
            alpha_bar_t = self.diffusion.alphas_cumprod[t_step]
            
            # 获取上一时刻 (prev_t) 的 alpha_bar
            prev_t = timesteps[i+1] if i < len(timesteps) - 1 else 0
            alpha_bar_t_prev = self.diffusion.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0).to(self.device)
            
            # 2.3 DDIM 更新公式
            sigma_t = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev))
            
            # 预测 x0 (pred_x0)
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * model_output) / torch.sqrt(alpha_bar_t)
            
            # 指向 x_{t-1} 的方向
            dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * model_output
            
            # 噪声项
            noise = sigma_t * torch.randn_like(x)
            
            # 更新 x
            x = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + noise
            
        return x

    @torch.no_grad()
    def visualize(self, epoch):
        # 只在 Rank 0 进行可视化保存，但这不影响其他 Rank 参与 FID 计算
        if self.config.local_rank != 0:
            return

        self.model.eval()
        n_samples = 4
        labels = torch.randint(0, 1000, (n_samples,), device=self.device)
        
        in_channels = getattr(self.config, 'in_channels', 4)
        input_size = getattr(self.config, 'input_size', 32)
        latent_size = (in_channels, input_size, input_size)

        # 使用 DDIM 快速采样
        z = self.sample_ddim(n_samples, labels, latent_size, num_inference_steps=50)
        x_recon = self.vae.decode(z / 0.18215).sample
        x_recon = torch.clamp((x_recon + 1.0) / 2.0, 0.0, 1.0)
        
        save_path = os.path.join(self.config.results_dir, f"sample_epoch_{epoch}.png")
        save_image(x_recon, save_path, nrow=2)
        print(f"[Visual] Saved visualization to {save_path}")
        
        self.model.train()

    @torch.no_grad()
    def evaluate_fid(self, epoch, num_gen_batches=10):
        """
        分布式 FID 评估 (所有 Rank 共同参与)
        """
        if self.fid_metric is None:
            return

        # 同步屏障：确保所有进程都准备好进入评估阶段
        if self.config.use_ddp:
            dist.barrier()

        if self.config.local_rank == 0:
            print(f"[FID] Starting distributed evaluation for epoch {epoch}...")
        
        self.model.eval()
        self.fid_metric.reset()

        # 1. 更新真实图片分布 (Real)
        # 每个 Rank 读取自己的 loader 数据，torchmetrics 会自动 sync
        # 为了速度，我们只取 loader 的前 N 个 batch
        num_real_batches = num_gen_batches # 保持数量一致
        for i, (real_imgs, _) in enumerate(self.loader):
            if i >= num_real_batches: break
            real_imgs = real_imgs.to(self.device)
            # 修正: 严格转换为 uint8 [0, 255]
            # real_imgs 原始范围 [-1, 1]
            real_imgs = ((real_imgs + 1.0) / 2.0 * 255.0).clamp(0, 255).to(torch.uint8)
            self.fid_metric.update(real_imgs, real=True)
        
        # 2. 更新生成图片分布 (Fake) - 并行生成
        # 每个 Rank 生成一部分图片，大大加快速度
        for _ in range(num_gen_batches):
            n_samples = self.config.batch_size # 每个 GPU 生成 batch_size 张
            labels = torch.randint(0, 1000, (n_samples,), device=self.device)
            
            in_channels = getattr(self.config, 'in_channels', 4)
            input_size = getattr(self.config, 'input_size', 32)
            latent_size = (in_channels, input_size, input_size)
            
            # 使用 DDIM 快速采样 (50步)
            z = self.sample_ddim(n_samples, labels, latent_size, num_inference_steps=50)
            
            fake_imgs = self.vae.decode(z / 0.18215).sample
            # 修正: 严格转换为 uint8 [0, 255]
            fake_imgs = ((fake_imgs + 1.0) / 2.0 * 255.0).clamp(0, 255).to(torch.uint8)
            
            self.fid_metric.update(fake_imgs, real=False)
            
        # 3. 计算全局 FID (内部会做 AllReduce)
        # 这一步是同步点
        fid_score = self.fid_metric.compute()
        
        if self.config.local_rank == 0:
            print(f"[FID] Epoch {epoch} | Distributed FID Score: {fid_score.item():.4f}")
            with open(os.path.join(self.config.results_dir, "fid_log.txt"), "a") as f:
                f.write(f"Epoch {epoch}, FID: {fid_score.item():.4f}\n")

        self.model.train()
        
        # 再次同步，防止某些 Rank 跑得太快回到 Training Loop
        if self.config.use_ddp:
            dist.barrier()

    def train_one_epoch(self, epoch):
        self.model.train()
        start_time = time()
        
        # 获取最大步数限制 (用于快速测试)
        max_steps = getattr(self.config, 'max_train_steps', None)
        
        for step, (images, labels) in enumerate(self.loader):
            if max_steps is not None and step >= max_steps:
                if self.config.local_rank == 0:
                    print(f"Debug: Max steps {max_steps} reached, breaking epoch.")
                break

            images = images.to(self.device)
            labels = labels.to(self.device)
            
            with torch.no_grad():
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

        # --- Epoch 结束逻辑 (所有 Rank 都参与) ---
        
        # 1. 简单的可视化 (内部有 rank0 check)
        viz_interval = getattr(self.config, 'log_interval', 1)
        if epoch % viz_interval == 0:
            self.visualize(epoch)
        
        # 2. 分布式 FID 评估 (内部处理同步，不再需要外部 barrier)
        # 默认每 5 个 epoch 测一次
        if epoch > 0 and epoch % 5 == 0:
            # num_gen_batches 可以设大一点，因为是分布式生成
            # 例如 4卡 x 32 batch x 15 batches = 1920 张图片，足够看大概趋势
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
        # 修复：设置 weights_only=False 以允许加载 SimpleNamespace (config)
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