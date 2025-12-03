import torch
import torch.nn as nn
import numpy as np

class GaussianDiffusion:
    """
    管理扩散过程的数学逻辑 (Schedule, q_sample, loss calculation)
    与模型架构完全解耦
    """
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # 定义 Beta Schedule (Linear)
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 预计算常用变量 (转回 float32 节省显存)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).float()

    def q_sample(self, x_start, t, noise=None):
        """
        前向加噪过程: x_0 -> x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alpha = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def p_losses(self, model, x_start, t, y, noise=None):
        """
        计算训练损失
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # 1. 生成带噪图片 x_t
        x_t = self.q_sample(x_start, t, noise)
        
        # 2. 模型预测噪声 (或同时预测方差)
        model_output = model(x_t, t, y)
        
        # DiT 通常学习预测 variance (learn_sigma=True)，输出通道是 2*C
        # 这里为了演示简化为只预测噪声
        # 如果 output 是 chunk 过的，取前半部分作为噪声预测
        if model_output.shape[1] == 2 * x_start.shape[1]:
            model_output, _ = model_output.chunk(2, dim=1)
            
        # 3. 计算 MSE Loss
        loss = torch.nn.functional.mse_loss(model_output, noise)
        return loss

    def sample(self, model, n, labels, size, cfg_scale=4.0):
        """
        推理采样 (简单的 DDPM 采样，实际应使用 DDIM)
        """
        model.eval()
        with torch.no_grad():
            x = torch.randn(n, *size).to(self.device)
            for i in reversed(range(self.num_timesteps)):
                t = torch.full((n,), i, device=self.device, dtype=torch.long)
                
                # Classifier-Free Guidance Logic would go here
                # ...
                
                # 简单示意
                pred_noise = model(x, t, labels)
                if pred_noise.shape[1] == 2 * x.shape[1]:
                    pred_noise, _ = pred_noise.chunk(2, dim=1)
                
                # 更新 x (公式省略，使用 DDPM 或 DDIM update)
                # x = ... 
        model.train()
        return x