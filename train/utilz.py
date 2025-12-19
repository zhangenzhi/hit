import torch
import os
import torch.distributed as dist
from torchvision.utils import save_image

class EMA:
    """
    指数移动平均 (Exponential Moving Average) 用于模型参数平滑。
    通常能显著提升生成模型的 FID 分数。
    """
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        # 处理 DDP 包裹的情况
        model_to_track = self.model.module if hasattr(self.model, 'module') else self.model
        for name, param in model_to_track.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self):
        model_to_track = self.model.module if hasattr(self.model, 'module') else self.model
        for name, param in model_to_track.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                # 这是一个内存带宽密集型操作
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        model_to_track = self.model.module if hasattr(self.model, 'module') else self.model
        for name, param in model_to_track.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        model_to_track = self.model.module if hasattr(self.model, 'module') else self.model
        for name, param in model_to_track.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def normalize_images(images):
    """
    将 [-1, 1] 的 tensor 转换为 [0, 255] 的 uint8 tensor
    """
    images = (images + 1.0) / 2.0
    images = images.clamp(0.0, 1.0)
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    return (images * 255.0).to(torch.uint8)

@torch.no_grad()
def visualize(trainer, epoch):
    """
    可视化采样函数，接收 trainer 实例
    """
    if trainer.config.local_rank != 0: return
    
    trainer.ema.apply_shadow()
    trainer.model.eval()
    try:
        print(f"[Visual] Generating samples for Epoch {epoch}...")
        n_samples = 4
        labels = torch.randint(0, 1000, (n_samples,), device=trainer.device)
        in_channels = getattr(trainer.config, 'in_channels', 4)
        input_size = getattr(trainer.config, 'input_size', 32)
        latent_size = (in_channels, input_size, input_size)

        # 调用 trainer 的 sample_ddim 方法
        z = trainer.sample_ddim(n_samples, labels, latent_size, cfg_scale=1.0, model=trainer.model)
        x_recon = trainer.vae.decode(z.float() / 0.18215).sample.float()
        x_vis = ((x_recon.clamp(-1, 1) + 1.0) / 2.0).clamp(0.0, 1.0)
        
        save_path = os.path.join(trainer.config.results_dir, f"sample_epoch_{epoch}.png")
        save_image(x_vis, save_path, nrow=2)
        print(f"[Visual] Saved to {save_path}")
    finally:
        trainer.ema.restore()
        trainer.model.train()

@torch.no_grad()
def evaluate_fid(trainer, epoch, num_gen_batches=10):
    """
    FID 评估函数，接收 trainer 实例
    """
    if trainer.fid_metric is None:
        return
    if trainer.config.use_ddp:
        dist.barrier()
    
    print(f"[FID] Starting evaluation for Epoch {epoch}...")
    trainer.ema.apply_shadow()
    trainer.model.eval()
    trainer.fid_metric.reset()
    
    # OOM Fix
    torch.cuda.empty_cache()
    vae_batch_size = 32

    # 优先使用验证集
    loader_to_use = trainer.val_loader if trainer.val_loader is not None else trainer.loader
    loader_iter = iter(loader_to_use)

    try:
        # 1. 计算真实图片特征 (Real Statistics)
        for i in range(num_gen_batches):
            try:
                real_imgs, _ = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader_to_use)
                real_imgs, _ = next(loader_iter)
                
            real_imgs = real_imgs.to(trainer.device)
            
            # 如果是 Latent [B, 4, H, W]，需要解码成 Pixel
            if real_imgs.shape[1] == 4:
                real_imgs = real_imgs.float()
                decoded_list = []
                # 分块解码防止 OOM
                for k in range(0, real_imgs.shape[0], vae_batch_size):
                    batch_slice = real_imgs[k : k + vae_batch_size]
                    decoded_slice = trainer.vae.decode(batch_slice).sample.float()
                    decoded_list.append(decoded_slice)
                real_imgs = torch.cat(decoded_list, dim=0)
            
            real_imgs = real_imgs.clamp(-1, 1)
            real_imgs_uint8 = normalize_images(real_imgs)
            trainer.fid_metric.update(real_imgs_uint8, real=True)
        
        # 2. 计算生成图片特征 (Fake Statistics)
        for i in range(num_gen_batches):
            n_samples = trainer.config.batch_size
            labels = torch.randint(0, 1000, (n_samples,), device=trainer.device)
            in_channels = getattr(trainer.config, 'in_channels', 4)
            input_size = getattr(trainer.config, 'input_size', 32)
            latent_size = (in_channels, input_size, input_size)
            
            # 采样使用 cfg=4.0 和 无截断策略
            z = trainer.sample_ddim(n_samples, labels, latent_size, num_inference_steps=50, cfg_scale=1.0, model=trainer.model)
            z = z.float()
            
            # 分块解码
            decoded_list = []
            for k in range(0, z.shape[0], vae_batch_size):
                z_slice = z[k : k + vae_batch_size]
                decoded_slice = trainer.vae.decode(z_slice / 0.18215).sample.float()
                decoded_list.append(decoded_slice)
            
            fake_imgs = torch.cat(decoded_list, dim=0)
            fake_imgs = fake_imgs.clamp(-1, 1)
            
            fake_imgs_uint8 = normalize_images(fake_imgs)
            trainer.fid_metric.update(fake_imgs_uint8, real=False)
        
        fid_score = trainer.fid_metric.compute()
        if trainer.config.local_rank == 0:
            print(f"[FID] Epoch {epoch} | FID Score: {fid_score.item():.4f}")
            with open(os.path.join(trainer.config.results_dir, "fid_log.txt"), "a") as f:
                f.write(f"Epoch {epoch}, FID: {fid_score.item():.4f}\n")
    finally:
        trainer.ema.restore()
        trainer.model.train()
        torch.cuda.empty_cache()
    if trainer.config.use_ddp:
        dist.barrier()

def save_checkpoint(trainer, epoch, avg_loss=None):
    """
    保存 Checkpoint
    """
    if trainer.config.local_rank == 0:
        checkpoint_path = os.path.join(trainer.config.results_dir, f"checkpoint_{epoch}.pt")
        checkpoint = {
            "model": trainer.model.module.state_dict() if trainer.config.use_ddp else trainer.model.state_dict(),
            "ema": trainer.ema.shadow, 
            "optimizer": trainer.optimizer.state_dict(),
            "epoch": epoch,
            "config": trainer.config,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        latest_path = os.path.join(trainer.config.results_dir, "latest.pt")
        torch.save(checkpoint, latest_path)

def resume_checkpoint(trainer, checkpoint_path):
    """
    恢复 Checkpoint
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
    
    model_state_dict = checkpoint["model"]
    # 处理可能的 DDP 前缀不匹配
    if trainer.config.use_ddp:
            trainer.model.module.load_state_dict(model_state_dict)
    else:
            # 如果 checkpoint 是 DDP 存的但当前是单卡，去掉 'module.' 前缀
            new_state_dict = {}
            for k, v in model_state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            trainer.model.load_state_dict(new_state_dict)
        
    if "ema" in checkpoint:
        trainer.ema.shadow = checkpoint["ema"]
    else:
        trainer.ema.register()

    if "optimizer" in checkpoint and trainer.optimizer is not None:
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        
    start_epoch = checkpoint.get("epoch", -1) + 1
    return start_epoch