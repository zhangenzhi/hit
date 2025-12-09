import argparse
import os
import torch
import random
from diffusers import AutoencoderKL
from torchvision.utils import save_image
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Sanity check: Load latents and VAE decode them directly.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the directory containing .pt files")
    parser.add_argument("--save_path", type=str, default="./reconstruction_check", help="Where to save reconstructed images")
    parser.add_argument("--vae_model", type=str, default="stabilityai/sd-vae-ft-mse", help="VAE model name")
    parser.add_argument("--num_samples", type=int, default=10, help="How many images to check")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    device = torch.device(args.device)

    print(f"--- Loading VAE: {args.vae_model} ---")
    vae = AutoencoderKL.from_pretrained(args.vae_model).to(device)
    vae.eval()

    print(f"--- Scanning for .pt files in {args.data_path} ---")
    pt_files = []
    for root, _, files in os.walk(args.data_path):
        for f in files:
            if f.endswith(".pt"):
                pt_files.append(os.path.join(root, f))
                if len(pt_files) >= args.num_samples:
                    break
        if len(pt_files) >= args.num_samples:
            break

    if not pt_files:
        print("Error: No .pt files found!")
        return

    print(f"Found {len(pt_files)} samples. Processing...")

    with torch.no_grad():
        for i, pt_path in enumerate(pt_files):
            # 1. Load Latent
            try:
                # 加载 Latent
                latent = torch.load(pt_path, map_location=device)
            except Exception as e:
                print(f"Failed to load {pt_path}: {e}")
                continue
            
            # 2. Handle Dimensions (Crops)
            # 你的数据可能是 [Num_Crops, 4, 32, 32] 或 [4, 32, 32]
            if latent.dim() == 4:
                print(f"[{i}] Shape {latent.shape} -> Selecting random crop")
                idx = random.randint(0, latent.shape[0] - 1)
                latent = latent[idx] # 取出一个 crop: [4, 32, 32]
            
            # 增加 Batch 维度 -> [1, 4, 32, 32]
            latent = latent.unsqueeze(0)

            # 3. Statistical Analysis & Scaling Check (关键步骤)
            mean = latent.mean().item()
            std = latent.std().item()
            min_val = latent.min().item()
            max_val = latent.max().item()

            print(f"Sample {i} Stats | Mean: {mean:.2f} | Std: {std:.2f} | Range: [{min_val:.2f}, {max_val:.2f}]")

            # 智能判断逻辑：
            # SD VAE 的 Unscaled Latent 标准差通常在 4.0 ~ 6.0 之间
            # 如果经过了 0.18215 缩放，标准差通常在 0.8 ~ 1.2 之间
            
            is_scaled = False
            if std < 2.0:
                print(f"  -> Detect SCALED latent (Std ~ 1.0). Dividing by 0.18215 before decoding.")
                latent = latent / 0.18215
                is_scaled = True
            else:
                print(f"  -> Detect UNSCALED latent (Std > 2.0). Decoding directly.")

            if std < 0.01:
                print(f"  -> WARNING: Latent is mostly zeros or flat! Image will be gray.")

            # 4. Decode
            # image = vae.decode(latent).sample
            # 为了保险，显式转为 float32
            image = vae.decode(latent.float()).sample 

            # 5. Post-process
            image = (image / 2 + 0.5).clamp(0, 1)

            # 6. Save
            file_name = f"recon_{i}_scaled.png" if is_scaled else f"recon_{i}_unscaled.png"
            save_full_path = os.path.join(args.save_path, file_name)
            save_image(image, save_full_path)
            print(f"  -> Saved to {save_full_path}")

    print(f"\nDone! Check images in {args.save_path}")

if __name__ == "__main__":
    main()