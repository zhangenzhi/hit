import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from time import time
from torchvision.utils import save_image, make_grid

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
        
        # --- 新增: FID 评估指标初始化 ---
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
        """
        在 Trainer 内部实现的简单 DDPM 采样循环
        解决 AttributeError: 'GaussianDiffusion' object has no attribute 'sample'
        """
        self.model.eval()
        x = torch.randn(n, *size).to(self.device)
        
        for i in reversed(range(self.diffusion.num_timesteps)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            
            # 1. 预测噪声
            model_output = self.model(x, t, labels)
            # 如果模型输出包含方差 (2*C)，只取前半部分作为噪声预测
            if model_output.shape[1] == 2 * x.shape[1]:
                model_output, _ = model_output.chunk(2, dim=1)
            
            # 2. 获取当前时间步的参数
            beta = self.diffusion.betas[i]
            alpha = self.diffusion.alphas[i]
            # alpha_cumprod = self.diffusion.alphas_cumprod[i]
            sqrt_one_minus_alpha_cumprod = self.diffusion.sqrt_one_minus_alphas_cumprod[i]
            
            # 3. 计算均值: 1/sqrt(alpha) * (x - beta/sqrt(1-alpha_bar) * eps)
            coeff = beta / sqrt_one_minus_alpha_cumprod
            mean = (1 / torch.sqrt(alpha)) * (x - coeff * model_output)
            
            # 4. 加噪声 (t > 0)
            if i > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta) # 简单的 sigma = sqrt(beta)
                x = mean + sigma * noise
            else:
                x = mean
                
        return x

    @torch.no_grad()
    def visualize(self, epoch):
        """
        生成采样图片进行可视化验证
        """
        self.model.eval()
        # 采样数量
        n_samples = 4
        # 随机类别 (0-999)
        labels = torch.randint(0, 1000, (n_samples,), device=self.device)
        
        # 获取 latent 尺寸 (C, H, W)
        # 修改：从 config 获取参数，因为模型实例可能未保存这些属性
        # flat_config 中包含了 model.in_channels 和 model.input_size
        in_channels = getattr(self.config, 'in_channels', 4)
        input_size = getattr(self.config, 'input_size', 32)
        latent_size = (in_channels, input_size, input_size)

        # 1. 扩散模型采样 Latent
        # 修改：使用内部实现的 sample_ddpm 替代 self.diffusion.sample
        z = self.sample_ddpm(n_samples, labels, latent_size)
        
        # 2. VAE 解码: Latent -> Pixel
        # 注意: SD VAE 训练时乘了 0.18215，解码时需要除回去
        x_recon = self.vae.decode(z / 0.18215).sample
        
        # 3. 归一化还原 [-1, 1] -> [0, 1]
        x_recon = torch.clamp((x_recon + 1.0) / 2.0, 0.0, 1.0)
        
        # 4. 保存图片
        save_path = os.path.join(self.config.results_dir, f"sample_epoch_{epoch}.png")
        save_image(x_recon, save_path, nrow=2)
        print(f"[Visual] Saved visualization to {save_path}")
        
        self.model.train()

    @torch.no_grad()
    def evaluate_fid(self, epoch, num_batches=5):
        """
        计算 FID 分数 (需要 torchmetrics)
        """
        if self.fid_metric is None:
            return

        print(f"[FID] Starting evaluation for epoch {epoch}...")
        self.model.eval()
        self.fid_metric.reset()

        # 1. 收集真实图片分布 (从 dataloader)
        # 注意: 这里简单复用 train loader 的一部分，理想情况下应使用 val loader
        for i, (real_imgs, _) in enumerate(self.loader):
            if i >= num_batches: break
            real_imgs = real_imgs.to(self.device)
            # map [-1, 1] -> [0, 255] uint8
            real_imgs_uint8 = ((real_imgs + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            self.fid_metric.update(real_imgs_uint8, real=True)
        
        # 2. 收集生成图片分布
        for _ in range(num_batches):
            n_samples = self.config.batch_size
            labels = torch.randint(0, 1000, (n_samples,), device=self.device)
            
            # 这里也同样修改为从 config 获取 latent_size
            in_channels = getattr(self.config, 'in_channels', 4)
            input_size = getattr(self.config, 'input_size', 32)
            latent_size = (in_channels, input_size, input_size)
            
            # 修改：使用内部实现的 sample_ddpm
            z = self.sample_ddpm(n_samples, labels, latent_size)
            fake_imgs = self.vae.decode(z / 0.18215).sample
            fake_imgs_uint8 = ((fake_imgs + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            self.fid_metric.update(fake_imgs_uint8, real=False)
            
        # 3. 计算 FID
        fid_score = self.fid_metric.compute()
        print(f"[FID] Epoch {epoch} | FID Score: {fid_score.item():.4f}")
        
        # 这里可以将 FID 写入日志文件
        with open(os.path.join(self.config.results_dir, "fid_log.txt"), "a") as f:
            f.write(f"Epoch {epoch}, FID: {fid_score.item():.4f}\n")

        self.model.train()

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

        # --- Epoch 结束后的可视化与评估 (仅在 Rank 0 进行) ---
        if self.config.local_rank == 0:
            # 每隔一定的 epoch 可视化一次 (假设 config 中有 save_interval，如果没有则默认每 5 epoch)
            viz_interval = getattr(self.config, 'log_interval', 1) # 这里简单复用 log_interval 或设为 1
            if epoch % viz_interval == 0:
                self.visualize(epoch)
            
            # FID 计算通常比较慢，建议频率低一点，例如每 5 个 epoch
            # if epoch > 0 and epoch % 5 == 0:
            #     self.evaluate_fid(epoch)
                
    def save_checkpoint(self, epoch):
        if self.config.local_rank == 0:
            checkpoint_path = os.path.join(self.config.results_dir, f"checkpoint_{epoch}.pt")
            
            # 保存完整状态：模型权重、优化器状态、Epoch、配置
            checkpoint = {
                "model": self.model.module.state_dict() if self.config.use_ddp else self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
                "config": self.config,
            }
            
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
            # 同时保存一个 latest.pt 方便自动恢复
            latest_path = os.path.join(self.config.results_dir, "latest.pt")
            torch.save(checkpoint, latest_path)

    def resume_checkpoint(self, checkpoint_path):
        """
        从 Checkpoint 恢复模型和优化器状态
        返回: start_epoch (int)
        """
        print(f"Loading checkpoint from {checkpoint_path}...")
        # 确保加载到正确的设备
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 1. 加载模型权重
        model_state_dict = checkpoint["model"]
        if self.config.use_ddp:
            self.model.module.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(model_state_dict)
        print("Model weights loaded.")

        # 2. 加载优化器状态
        if "optimizer" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("Optimizer state loaded.")
            
        # 3. 获取开始 Epoch
        start_epoch = checkpoint.get("epoch", -1) + 1
        print(f"Resuming training from epoch {start_epoch}")
        
        return start_epoch