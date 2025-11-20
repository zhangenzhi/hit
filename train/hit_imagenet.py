import os
import sys
import yaml
import logging
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# 引入我们定义的 HiT 模型
from model_hit import create_hit_model

# 假设你有一个现成的 ImageNet dataloader (Mock for demonstration)
# from dataset.imagenet import imagenet_distribute 
import torchvision
import torchvision.transforms as transforms

def setup_logging(args):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def get_dataloaders(args, config):
    """ 
    这里使用 torchvision 的 FakeData 方便你直接跑通代码逻辑。
    实际使用请替换为你原本的 imagenet_distribute
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config['model']['img_size']),
        transforms.ToTensor(),
    ])
    # 使用 CIFAR10 或 FakeData 做演示
    dataset = torchvision.datasets.FakeData(
        size=1000, 
        image_size=(3, config['model']['img_size'], config['model']['img_size']),
        num_classes=1000,
        transform=transform
    )
    sampler = torch.utils.data.DistributedSampler(dataset) if dist.is_initialized() else None
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config['training']['batch_size'], 
        sampler=sampler, num_workers=4, pin_memory=True
    )
    return {'train': loader, 'val': loader}

def train_one_epoch(model, loader, optimizer, device, epoch, config):
    model.train()
    optimizer.zero_grad()
    
    total_loss = 0
    steps = 0
    
    for i, (imgs, _) in enumerate(loader): # HiT 不需要 label
        imgs = imgs.to(device, non_blocking=True)
        
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            # HiT forward 直接返回 loss
            loss, _, mask = model(imgs)
            loss = loss / config['training'].get('accumulation_steps', 1)
        
        loss.backward()
        
        if (i + 1) % config['training'].get('accumulation_steps', 1) == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪很关键
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()
        steps += 1
        
        if i % 20 == 0 and dist.get_rank() == 0:
             logging.info(f"Epoch {epoch} [{i}/{len(loader)}] Loss: {loss.item():.4f}")
             
    return total_loss / steps

def main(args):
    # 1. DDP Init
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    # 2. Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    if dist.get_rank() == 0:
        setup_logging(args)
        logging.info(f"Initializing HiT Model: {config['model']['embed_dim']} dim")

    # 3. Model
    model = create_hit_model(config).to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    # 4. Optimizer (ViT 标准配置)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        betas=(0.9, 0.95),
        weight_decay=0.05
    )
    
    # 5. Data
    loaders = get_dataloaders(args, config)
    
    # 6. Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['num_epochs'])
    
    # 7. Loop
    for epoch in range(config['training']['num_epochs']):
        if hasattr(loaders['train'].sampler, 'set_epoch'):
            loaders['train'].sampler.set_epoch(epoch)
            
        avg_loss = train_one_epoch(model, loaders['train'], optimizer, local_rank, epoch, config)
        scheduler.step()
        
        if dist.get_rank() == 0:
            logging.info(f"Epoch {epoch} Finished. Avg Loss: {avg_loss:.4f}")
            # Save logic...
            torch.save(model.module.state_dict(), os.path.join(args.output, "hit_latest.pth"))

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config.yaml')
    parser.add_argument('--output', default='./output')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    main(args)