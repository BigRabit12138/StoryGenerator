import json
from tqdm import tqdm
from pathlib import Path

import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

from model.configuration_qwen2 import Qwen2Config
from model.modeling_qwen2 import Qwen2ForStoryGenerator
from dataset.pretrain_weight import PretrainWeightDataset

# 超参数配置
RUN_CONFIG = {
    "batch_size": 4,
    "lr": 1e-4,
    "epochs": 20,
    "warmup_steps": 200000,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def masked_mse_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor = None,
    ignore_value: float = -100.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """支持 Mask 的 MSE 损失函数"""
    if input.shape[-1] == 1:
        input = input.squeeze(-1)
    if mask is None:
        mask = (target != ignore_value).float()
    squared_error = (input - target) ** 2
    masked_error = squared_error * mask
    
    if reduction == 'mean':
        valid_count = mask.sum()
        return masked_error.sum() / (valid_count + 1e-6)
    elif reduction == 'sum':
        return masked_error.sum()
    elif reduction == 'none':
        return masked_error
    else:
        raise ValueError(f"无效的 reduction 模式: {reduction}")

def prepare_dataloaders():
    train_dataset = PretrainWeightDataset(mode='train')
    val_dataset = PretrainWeightDataset(mode='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=RUN_CONFIG['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=PretrainWeightDataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=RUN_CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=PretrainWeightDataset.collate_fn
    )
    return train_loader, val_loader

def train_epoch(model, loader, optimizer, scheduler, scaler, device, writer, epoch):
    model.train()
    total_loss = 0
    
    for step, batch in tqdm(enumerate(loader)):
        input_ids = batch['input_ids'].to(device)
        locations = batch['locations'].to(device)
        labels = batch['labels'].to(device)
        masks = batch['masks'].to(device)
        
        optimizer.zero_grad()
        
        # 混合精度训练
        with torch.amp.autocast(device):
            outputs = model(
                input_ids=input_ids, 
                locations=locations, 
                labels=labels,
                masks=masks,
            )
            loss = outputs.loss
        
        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        
        # 记录每一步的学习率
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch * len(loader) + step)
        # 记录每一步的学习率
        writer.add_scalar('Loss', loss.item(), epoch * len(loader) + step)
        if torch.isnan(loss):
            continue
    
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(device)
            locations = batch['locations'].to(device)
            labels = batch['labels'].to(device)
            masks = batch['masks'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                locations=locations,
                labels=labels,
                masks=masks,
            )
            total_loss += outputs.loss.item()
    
    return total_loss / len(loader)

def main():
    # 初始化TensorBoard
    writer = SummaryWriter(log_dir='/root/tf-logs')
    
    # 准备数据
    train_loader, val_loader = prepare_dataloaders()
    
    # 加载配置
    config = Qwen2Config.from_json_file('./config.json')
    STATUS_FILE = Path(__file__).parent / 'dataset' / "status.json"
    if Path(STATUS_FILE).exists():
        with open(STATUS_FILE) as f:
            status = json.load(f)
        config.vocab_size = status["max_dimension"]
    
    # 初始化模型
    model = Qwen2ForStoryGenerator(config).to(RUN_CONFIG['device'])
    model.loss_function = masked_mse_loss
    
    # 优化器和学习率调度
    optimizer = AdamW(model.parameters(), lr=RUN_CONFIG['lr'])
    total_steps = len(train_loader) * RUN_CONFIG['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=RUN_CONFIG['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # 混合精度训练
    scaler = torch.amp.GradScaler(enabled=RUN_CONFIG['device'] == 'cuda')
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(RUN_CONFIG['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, RUN_CONFIG['device'], writer, epoch)
        val_loss = validate(model, val_loader, RUN_CONFIG['device'])
        
        # 记录指标
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained("best_model/")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, "best_model/checkpoint.pth")
        
        print(f"Epoch {epoch+1}/{RUN_CONFIG['epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Current LR: {scheduler.get_last_lr()[0]:.2e}\n")
    
    writer.close()

if __name__ == "__main__":
    main()