import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from diffusers import AutoencoderKL
from torchvision import transforms

import torch

class GaussianDiffusion:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear Beta Schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).float()

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def p_losses(self, model, x_start, t, y, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        model_output = model(x_t, t, y)
        
        # 如果模型学习方差，输出通道是 2*C，前 C 个是噪声预测
        if model_output.shape[1] == 2 * x_start.shape[1]:
            model_output, _ = model_output.chunk(2, dim=1)
            
        return torch.nn.functional.mse_loss(model_output, noise)
    
# -----------------------------------------------------------------------------
# 1. 简易的扩散调度器 (与 diffusion/gaussian_diffusion.py 逻辑一致)
# -----------------------------------------------------------------------------
class SimpleDiffusion:
    def __init__(self, num_timesteps=1000, device="cuda"):
        # 核心定义：虽然 DDIM 推理只用 50 步，但模型权重是基于 1000 步的物理过程训练的。
        # DDIM 只是在这个 1000 步的稠密网格上进行"跳跃"采样。
        self.num_timesteps = num_timesteps
        self.device = device
        
        # 定义 Beta Schedule (Linear)
        # 注意：SD v1 使用的 Linear Schedule 在 t>300 时信噪比极低，导致肉眼看去全是噪声。
        # 这是 Linear Schedule 的特性（缺陷），后续模型如 DiT/SDXL 常改用 Cosine Schedule。
        beta_start = 1e-4
        beta_end = 0.02
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t):
        """
        前向加噪: x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * epsilon
        """
        noise = torch.randn_like(x_start)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        x_t = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        return x_t, noise

# -----------------------------------------------------------------------------
# 2. 辅助函数
# -----------------------------------------------------------------------------
def load_image(path, size=256):
    # 修改：直接从本地路径加载图片
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # map to [-1, 1]
    ])
    return transform(img).unsqueeze(0)

def decode_img(vae, latents):
    # VAE 解码: Latent -> Pixel
    # SD VAE 需要 rescale latents
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    return image[0]

# -----------------------------------------------------------------------------
# 3. 主可视化逻辑
# -----------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}...")

    # A. 加载模型 (使用 HuggingFace 的 SD VAE)
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    # 虽然推理只用 50 步，但必须载入 1000 步的 scheduler 才能对齐
    diffusion = SimpleDiffusion(num_timesteps=1000, device=device)

    # B. 准备图片
    # 请将下面的 "your_image.jpg" 替换为您想要可视化的本地图片路径
    img_path = "./diffusion/test.jpg" 
    
    # 简单的错误处理，防止路径不存在报错
    import os
    if not os.path.exists(img_path):
        print(f"警告: 找不到文件 '{img_path}'")
        print("请在脚本中修改 'img_path' 变量为有效的本地图片路径。")
        # 创建一个随机噪声图作为 fallback，以便代码能跑通演示
        print("正在使用随机噪声图作为演示...")
        x_pixel = torch.randn(1, 3, 256, 256).to(device).clamp(-1, 1)
    else:
        x_pixel = load_image(img_path, size=256).to(device) # (1, 3, 256, 256)

    # C. 像素 -> Latent (Encoding)
    print("Encoding to Latent...")
    with torch.no_grad():
        # VAE 编码得到分布，取均值作为 latent
        # 乘以缩放因子 0.18215 (SD 官方参数)
        z_0 = vae.encode(x_pixel).latent_dist.sample() * 0.18215

    print(f"Pixel shape: {x_pixel.shape}")   # (1, 3, 256, 256)
    print(f"Latent shape: {z_0.shape}")      # (1, 4, 32, 32) -> 压缩了 8 倍

    # -------------------------------------------------------------------------
    # D. 可视化绘图
    # -------------------------------------------------------------------------
    plt.figure(figsize=(20, 10))

    # 1. 原始图片
    plt.subplot(3, 5, 1)
    plt.imshow((x_pixel[0].cpu().permute(1, 2, 0) + 1) / 2)
    plt.title("Original Image (Pixel Space)")
    plt.axis('off')

    # 2. Latent Vector 可视化 (4个通道)
    # Latent 是 4x32x32，我们把4个通道画出来
    z_0_cpu = z_0[0].cpu().numpy()
    for i in range(4):
        plt.subplot(3, 5, 2 + i)
        plt.imshow(z_0_cpu[i], cmap='viridis')
        plt.title(f"Latent Channel {i+1}\n(32x32)")
        plt.axis('off')

    # 3. 扩散过程可视化 (模拟 DDIM 50 Steps)
    
    # 构造 DDIM 采样时间步：从 0 到 1000 均匀选取 50 个点
    # 步长 = 1000 / 50 = 20
    # DDIM step 0  -> Original step 0
    # DDIM step 10 -> Original step 200
    # DDIM step 49 -> Original step 980
    ddim_num_steps = 50
    ddim_timesteps = np.linspace(0, 999, ddim_num_steps, dtype=int)
    
    # 我们从中选 5 个代表性的点来画图：第 0, 10, 20, 30, 49 步
    selection_indices = [0, 3, 6, 9, 12]
    selected_timesteps = ddim_timesteps[selection_indices]
    
    print(f"Visualizing DDIM steps: {selection_indices}")
    print(f"Corresponding original timesteps: {selected_timesteps}")
    
    for idx, t in enumerate(selected_timesteps):
        t_tensor = torch.tensor([t], device=device).long()
        
        # 加噪: z_t
        z_t, noise = diffusion.q_sample(z_0, t_tensor)
        
        # 解码: z_t -> x_t (为了让人眼能看懂)
        img_recon = decode_img(vae, z_t)
        
        # 计算当前的 alpha_bar 以展示信噪比衰减
        alpha_bar = diffusion.alphas_cumprod[t].item()

        # Plot Noisy Image
        plt.subplot(3, 5, 6 + idx)
        plt.imshow(img_recon)
        ddim_idx = selection_indices[idx]
        # 使用 fr 字符串避免 LaTeX 转义问题，并显示更多信息
        plt.title(fr"DDIM Step {ddim_idx + 1}/50" + "\n" + fr"($t={t}, \bar{{\alpha}}_t={alpha_bar:.4f}$)")
        plt.axis('off')

        # Plot Noisy Latent (Channel 0)
        plt.subplot(3, 5, 11 + idx)
        plt.imshow(z_t[0, 0].cpu().numpy(), cmap='viridis') # 只看第0个通道
        plt.title(f"Latent (Ch0)\nDDIM Step {ddim_idx + 1}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("dit_noise_process.png")
    print("Visualization saved to dit_noise_process.png")
    plt.show()

if __name__ == "__main__":
    main()