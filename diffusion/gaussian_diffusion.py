import math
import numpy as np
import torch as th
import enum

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

class ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()

class ModelVarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()

class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
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
        
        self.model_mean_type = ModelMeanType.EPSILON
        self.model_var_type = ModelVarType.LEARNED_RANGE
        self.loss_type = LossType.MSE

        if schedule == "linear":
            betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
        elif schedule == "cosine":
            betas = betas_for_alpha_bar(
                num_timesteps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )
        else:
            raise NotImplementedError(f"Unknown schedule: {schedule}")

        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else np.array([])

        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )
        
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                setattr(self, k, th.from_numpy(v).to(device=device, dtype=th.float32))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, clip_denoised=False, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        model_output = model(x, t, **model_kwargs)

        if model_output.shape[1] == 2 * C:
            model_output, model_var_values = th.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = _extract_into_tensor(th.log(self.betas), t, x.shape)
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = (
                self.posterior_variance,
                self.posterior_log_variance_clipped,
            )
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        pred_xstart = process_xstart(
            self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        )
        
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    # --- Sampling Methods ---

    def p_sample(self, model, x, t, clip_denoised=False, model_kwargs=None):
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(self, model, shape, noise=None, clip_denoised=False, model_kwargs=None, progress=False):
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(self, model, shape, noise=None, clip_denoised=False, model_kwargs=None, progress=False):
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=self.device)
            
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=self.device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    # --- Wrapper for Trainer compatibility ---
    
    def sample_ddpm(self, model, labels, size, num_classes, cfg_scale=1.0, use_amp=False, dtype=th.float32, is_latent=False):
        """
        Adapter for DiT Trainer to call standard DDPM sampling
        """
        n = labels.shape[0]
        C, H, W = size
        shape = (n, C, H, W)
        
        # Pass labels and cfg_scale to model via kwargs
        model_kwargs = dict(y=labels, cfg_scale=cfg_scale, num_classes=num_classes)
        
        # Use AMP if requested
        with th.amp.autocast('cuda', enabled=use_amp, dtype=dtype):
             return self.p_sample_loop(
                model, 
                shape, 
                noise=None, 
                clip_denoised=False, 
                model_kwargs=model_kwargs, 
                progress=True
            )

    def sample_ddim(self, model, labels, size, num_classes, num_inference_steps=50, cfg_scale=1.0, use_amp=False, dtype=th.float32, is_latent=False):
        """
        Adapter for DiT Trainer to call DDIM sampling
        """
        n = labels.shape[0]
        C, H, W = size
        shape = (n, C, H, W)
        
        model_kwargs = dict(y=labels, cfg_scale=cfg_scale, num_classes=num_classes)
        
        with th.amp.autocast('cuda', enabled=use_amp, dtype=dtype):
            return self.ddim_sample_loop(
                model,
                shape,
                num_inference_steps=num_inference_steps,
                noise=None,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True
            )

    # --- DDIM Methods ---

    def ddim_sample(self, model, x, t, next_t, eta=0.0, clip_denoised=False, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        
        # 1. Get prediction of x_start and epsilon
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
        )
        
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        
        # 2. Compute alphas for t and next_t
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        
        # Handle next_t being -1 (end of chain)
        if next_t[0] < 0:
            alpha_bar_prev = th.tensor(1.0, device=self.device)
        else:
            alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod, next_t, x.shape)
            
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        
        # 3. Equation for x_{t-1}
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        
        noise = th.randn_like(x)
        sample = mean_pred + sigma * noise
        
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(self, model, shape, num_inference_steps=50, noise=None, clip_denoised=False, model_kwargs=None, progress=False, eta=0.0):
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            num_inference_steps=num_inference_steps,
            noise=noise,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(self, model, shape, num_inference_steps=50, noise=None, clip_denoised=False, model_kwargs=None, progress=False, eta=0.0):
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=self.device)
            
        # Create a subsequence of timesteps
        step_ratio = self.num_timesteps // num_inference_steps
        indices = list(range(0, self.num_timesteps, step_ratio))[::-1]
        
        # Ensure we have exactly num_inference_steps (sometimes integer division cuts one off)
        # Or just use the stride logic.
        
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i, step in enumerate(indices):
            t = th.tensor([step] * shape[0], device=self.device)
            
            # Determine next timestep
            if i == len(indices) - 1:
                next_step = -1
            else:
                next_step = indices[i + 1]
                
            next_t = th.tensor([next_step] * shape[0], device=self.device)
            
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    next_t,
                    eta=eta,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(self, model, x_start, x_t, t, clip_denoised=False, model_kwargs=None):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}
        model_output = model(x_t, t, **model_kwargs)

        B, C = x_t.shape[:2]
        if model_output.shape[1] == 2 * C:
            model_output, model_var_values = th.split(model_output, C, dim=1)
            frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
            
            # [FIX] Force clip_denoised=False for Latent Diffusion VLB calculation
            terms["vb"] = self._vb_terms_bpd(
                model=lambda *args, r=frozen_out: r,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,  # Changed from True to False
            )["output"]
            
            terms["vb"] *= 1.0

        target = noise
        terms["mse"] = mean_flat((target - model_output) ** 2)
        
        if "vb" in terms:
            terms["loss"] = terms["mse"] + terms["vb"]
        else:
            terms["loss"] = terms["mse"]

        return terms

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = arr[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (
        -1.0 + logvar2 - logvar1 + th.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = th.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = th.sigmoid(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    return log_probs