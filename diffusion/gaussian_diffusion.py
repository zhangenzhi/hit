import torch
import numpy as np
import math

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    创建一个 beta schedule，该 schedule 离散化了给定的 alpha_t_bar 函数，
    该函数定义了从 t = [0,1] 随时间变化的 (1-beta) 的累积乘积。
    来源于 OpenAI Improved Diffusion。
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class GaussianDiffusion:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule="linear", device="cuda"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # 1. 计算 Betas (使用 float64 避免精度损失)
        if schedule == "linear":
            # 经典的 Linear Schedule
            betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
        elif schedule == "cosine":
            # OpenAI 改进的 Cosine Schedule
            # 解决 "300步后全是噪声" 的问题，信噪比下降更平滑
            betas = betas_for_alpha_bar(
                num_timesteps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )
        else:
            raise NotImplementedError(f"Unknown schedule: {schedule}")

        self.betas = torch.from_numpy(betas).to(device=device, dtype=torch.float32)
        
        # 2. 计算 Alphas 及其累乘 (使用 numpy float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        
        # 3. 转回 Tensor (Float32)
        self.alphas = torch.from_numpy(alphas).to(device=device, dtype=torch.float32)
        self.alphas_cumprod = torch.from_numpy(alphas_cumprod).to(device=device, dtype=torch.float32)
        
        # 4. 预计算常用系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        # 确保系数正确 reshape
        sqrt_alpha = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def p_losses(self, model, x_start, t, y, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        model_output = model(x_t, t, y)
        
        if model_output.shape[1] == 2 * x_start.shape[1]:
            model_output, _ = model_output.chunk(2, dim=1)
            
        return torch.nn.functional.mse_loss(model_output, noise)