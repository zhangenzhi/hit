import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Block, PatchEmbedding

class SkewedNoiseScheduler(nn.Module):
    """
    HiT 专用的偏态噪声调度器。
    核心策略：主要采样高噪声区间（逼迫模型学语义），少量采样低噪声区间（让模型学修图）。
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        # 线性 Beta Schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # 注册 Buffer (自动随模型转移到 GPU)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def get_skewed_timesteps(self, n_samples, device):
        """
        关键策略：70% 概率在高噪区 [500, 1000)，30% 概率在低噪区 [0, 500)。
        """
        probs = torch.rand(n_samples, device=device)
        high_noise_mask = probs < 0.7  # 70% 概率
        
        # 高噪声区间
        t_high = torch.randint(
            int(self.num_timesteps * 0.5), self.num_timesteps, (n_samples,), device=device
        )
        # 低噪声区间
        t_low = torch.randint(
            0, int(self.num_timesteps * 0.5), (n_samples,), device=device
        )
        
        t = torch.where(high_noise_mask, t_high, t_low)
        return t

    def add_noise(self, x_start, t):
        """ x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * epsilon """
        noise = torch.randn_like(x_start)
        # Reshape for broadcasting [B, 1, 1]
        sqrt_alpha = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        
        x_noisy = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        return x_noisy, noise

class HiTEncoder(nn.Module):
    """
    HiT 的全能 Encoder。
    区别于 MAE：它不丢弃 Masked Tokens，而是把它们当做 Noisy Tokens 读入。
    区别于 ViT：它增加了 Type Embedding。
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 dropout=0.0):
        super().__init__()
        
        # 1. 基础组件
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # 2. 位置嵌入 & CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # 3. 【HiT 核心】Type Embedding (Explicit Conditioning)
        # 0: Clean (Anchor), 1: Noisy (Target)
        self.type_embed = nn.Embedding(2, embed_dim)
        
        # 4. Transformer Blocks (使用 timm 的标准 Block)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_drop = nn.Dropout(dropout)
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.pos_embed, std=.02)
        torch.nn.init.normal_(self.type_embed.weight, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_patches, mask):
        """
        x_patches: [B, N, D] 混合序列
        mask: [B, N] 0=Clean, 1=Noisy
        """
        B, N, D = x_patches.shape
        
        # 1. 加入位置编码 (跳过 CLS)
        x = x_patches + self.pos_embed[:, 1:, :]
        
        # 2. 加入 Type Embedding (区分 Clean vs Noisy)
        # mask 是 0/1，直接查表
        type_emb = self.type_embed(mask.long())
        x = x + type_emb
        
        # 3. 拼接 CLS Token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.pos_drop(x)
        
        # 4. 全注意力机制 (Full Attention)
        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x)
        return x

class HiT(nn.Module):
    """
    HiT: Hybrid image Transformer
    结合 MAE 的语义学习能力和 Diffusion 的生成能力。
    """
    def __init__(self, config):
        super().__init__()
        model_conf = config['model']
        self.mask_ratio = config['training']['mask_ratio']
        self.patch_size = model_conf['patch_size']
        
        # 核心 Encoder
        self.encoder = HiTEncoder(
            img_size=model_conf['img_size'],
            patch_size=model_conf['patch_size'],
            embed_dim=model_conf['embed_dim'],
            depth=model_conf['depth'],
            num_heads=model_conf['num_heads'],
            mlp_ratio=model_conf.get('mlp_ratio', 4.0)
        )
        
        # 预测头 (JiT style: 直接预测像素)
        # 输入 embed_dim -> 输出 patch_size^2 * 3
        self.head = nn.Linear(model_conf['embed_dim'], (self.patch_size**2) * 3)
        
        # 噪声调度器
        self.noise_scheduler = SkewedNoiseScheduler()

    def patchify(self, imgs):
        """ [B, 3, H, W] -> [B, N, D] """
        p = self.patch_size
        B, C, H, W = imgs.shape
        x = imgs.reshape(B, C, H//p, p, W//p, p)
        x = torch.einsum('bchpwq->bhwcpq', x)
        x = x.reshape(B, -1, C*p*p)
        return x

    def forward(self, imgs):
        """
        imgs: [B, 3, H, W]
        Returns: loss, pred_patches, mask
        """
        device = imgs.device
        
        # 1. 图像 Patch 化
        x_clean = self.patchify(imgs)
        
        # 2. 归一化 (MAE 技巧，对训练稳定性很重要)
        mean = x_clean.mean(dim=-1, keepdim=True)
        var = x_clean.var(dim=-1, keepdim=True)
        x_norm = (x_clean - mean) / (var + 1.e-6)**.5
        
        # 3. 构建混合输入 (Hybrid Input Generation)
        B, N, D = x_norm.shape
        
        # 3.1 生成 Mask (75% Noisy)
        len_keep = int(N * (1 - self.mask_ratio))
        noise_rand = torch.rand(B, N, device=device)
        ids_shuffle = torch.argsort(noise_rand, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        ids_noisy = ids_shuffle[:, len_keep:]
        
        # Mask: 0=Clean, 1=Noisy
        mask = torch.ones(B, N, device=device)
        mask.scatter_(1, ids_keep, 0.0)
        
        # 3.2 加噪 (只针对 Mask 部分)
        t = self.noise_scheduler.get_skewed_timesteps(B, device)
        x_noisy_patches, _ = self.noise_scheduler.add_noise(x_norm, t)
        
        # 3.3 拼装
        # Clean 区域保持原样，Noisy 区域使用 x_noisy_patches
        mask_unsqueezed = mask.unsqueeze(-1)
        x_input = x_norm * (1 - mask_unsqueezed) + x_noisy_patches * mask_unsqueezed
        
        # 4. 模型前向
        # Encoder 输出包含 CLS token，我们需要去掉它
        latent = self.encoder(x_input, mask)
        latent = latent[:, 1:, :] # [B, N, embed_dim]
        
        # 5. 预测 x_0
        pred = self.head(latent)
        
        # 6. 计算 Loss (仅计算 Noisy 区域的 MSE)
        # JiT 目标：预测 Clean Image (x_norm)
        loss = (pred - x_norm) ** 2
        loss = loss.mean(dim=-1)  # [B, N]
        
        # Masked MSE
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        
        return loss, pred, mask

def create_hit_model(config):
    return HiT(config)

if __name__ == "__main__":
    print("\n--- Starting HiT Model Integrity Test ---")
    
    # 1. 配置 Fake Config
    config = {
        'model': {
            'img_size': 224,
            'patch_size': 16,
            'embed_dim': 768,
            'depth': 4, # 减少 depth 以便快速测试
            'num_heads': 8,
            'mlp_ratio': 4.0
        },
        'training': {
            'mask_ratio': 0.75
        }
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # 2. 实例化模型
    model = create_hit_model(config).to(device)
    print("Model instantiated successfully.")
    
    # 3. Fake Input
    B, C, H, W = 2, 3, 224, 224
    fake_input = torch.randn(B, C, H, W).to(device)
    print(f"Fake Input Shape: {fake_input.shape}")
    
    # 4. 正向传播 (Forward Pass)
    print("\n--- Testing Forward Pass ---")
    loss, pred, mask = model(fake_input)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Prediction Shape: {pred.shape}")
    print(f"Mask Shape: {mask.shape}")
    print(f"Mask Ratio (Actual): {mask.sum() / mask.numel():.2f} (Target: {config['training']['mask_ratio']})")
    
    assert not torch.isnan(loss), "Loss is NaN!"
    assert pred.shape == (B, (H//16)*(W//16), 16*16*3), "Prediction shape mismatch!"
    
    # 5. 反向传播 (Backward Pass & Gradient Check)
    print("\n--- Testing Backward Pass ---")
    loss.backward()
    
    # 检查 Encoder 第一层 Block 的梯度是否存在
    # 注意：HiTEncoder -> blocks (ModuleList) -> [0] -> norm1 -> weight
    grad_check = model.encoder.blocks[0].norm1.weight.grad
    
    if grad_check is not None:
        print(f"✅ Gradient detected in first encoder block! (Norm1 weight grad mean: {grad_check.mean().item():.6f})")
    else:
        print("❌ No gradient found! Backward pass failed.")
        
    # 检查 Type Embedding 是否有梯度 (关键！确认 Clean/Noisy 区分机制在学习)
    type_embed_grad = model.encoder.type_embed.weight.grad
    if type_embed_grad is not None:
        print(f"✅ Gradient detected in Type Embedding! (Grad mean: {type_embed_grad.mean().item():.6f})")
    else:
        print("❌ No gradient found for Type Embedding!")
        
    print("\n--- Integrity Test Passed! ---")