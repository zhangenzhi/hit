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
        if isinstance(self.model, DDP):
            latent_size = (self.model.module.in_channels, self.model.module.input_size, self.model.module.input_size)
        else:
            latent_size = (self.model.in_channels, self.model.input_size, self.model.input_size)

        # 1. 扩散模型采样 Latent
        # 假设 diffusion.sample 实现了 DDIM/DDPM 采样循环
        z = self.diffusion.sample(self.model, n_samples, labels, latent_size)
        
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
            
            if isinstance(self.model, DDP):
                latent_size = (self.model.module.in_channels, self.model.module.input_size, self.model.module.input_size)
            else:
                latent_size = (self.model.in_channels, self.model.input_size, self.model.input_size)
            
            z = self.diffusion.sample(self.model, n_samples, labels, latent_size)
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
            if epoch > 0 and epoch % 5 == 0:
                self.evaluate_fid(epoch)
                
    def save_checkpoint(self, epoch):
        if self.config.local_rank == 0:
            checkpoint_path = os.path.join(self.config.results_dir, f"checkpoint_{epoch}.pt")
            to_save = self.model.module.state_dict() if self.config.use_ddp else self.model.state_dict()
            torch.save(to_save, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")