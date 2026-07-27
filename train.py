"""
Training script for U-Net segmentation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from model import UNet
from dataset import get_data_loaders
from utils import dice_coefficient, iou_score, EarlyStopping


class DiceBCELoss(nn.Module):
    """Combined Dice Loss + Binary Cross Entropy for better segmentation."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice = dice_coefficient(pred, target)
        return bce_loss + (1 - dice)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    epoch_iou = 0
    
    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_dice += dice_coefficient(outputs, masks).item()
        epoch_iou += iou_score(outputs, masks).item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice_coefficient(outputs, masks).item():.4f}'
        })
    
    return epoch_loss / len(loader), epoch_dice / len(loader), epoch_iou / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_dice = 0
    epoch_iou = 0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            epoch_loss += loss.item()
            epoch_dice += dice_coefficient(outputs, masks).item()
            epoch_iou += iou_score(outputs, masks).item()
    
    return epoch_loss / len(loader), epoch_dice / len(loader), epoch_iou / len(loader)


def train_model(epochs=20, batch_size=4, lr=1e-4, device='cuda'):
    # Setup
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Initialize
    model = UNet(n_channels=1, n_classes=1).to(device)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    early_stopping = EarlyStopping(patience=7)
    
    # Data
    train_loader, val_loader = get_data_loaders(batch_size=batch_size)
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': []}
    
    best_dice = 0.0
    
    print(f"\n{'='*50}")
    print(f"Training U-Net for Medical Image Segmentation")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print(f"{'='*50}\n")
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 30)
        
        train_loss, train_dice, train_iou = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, device
        )
        
        scheduler.step(val_loss)
        early_stopping(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        
        print(f"Train Loss: {train_loss:.4f} | Dice: {train_dice:.4f} | IoU: {train_iou:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, 'checkpoints/best_model.pth')
            print(f"✓ Saved best model (Dice: {best_dice:.4f})")
        
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
    
    print(f"\n{'='*50}")
    print(f"Training Complete! Best Val Dice: {best_dice:.4f}")
    print(f"{'='*50}")
    
    return model, history


if __name__ == "__main__":
    model, history = train_model(epochs=20, batch_size=4, lr=1e-4)