"""
Evaluation and visualization of segmentation results.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

from model import UNet
from dataset import SyntheticMedicalDataset
from utils import visualize_prediction, dice_coefficient, iou_score


def evaluate_model(model_path='checkpoints/best_model.pth', num_samples=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = UNet(n_channels=1, n_classes=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test dataset
    test_dataset = SyntheticMedicalDataset(num_samples=200, img_size=256)
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with Dice: {checkpoint['best_dice']:.4f}")
    print(f"\nEvaluating on {num_samples} test samples...\n")
    
    total_dice = 0
    total_iou = 0
    
    with torch.no_grad():
        for i in range(num_samples):
            image, mask = test_dataset[i]
            image = image.unsqueeze(0).to(device)  # Add batch dimension
            mask = mask.unsqueeze(0).to(device)
            
            pred = model(image)
            
            dice = dice_coefficient(pred, mask).item()
            iou = iou_score(pred, mask).item()
            total_dice += dice
            total_iou += iou
            
            print(f"Sample {i+1}: Dice = {dice:.4f}, IoU = {iou:.4f}")
            visualize_prediction(image[0], mask[0], pred[0], 
                               save_path=f'results/sample_{i+1}.png')
    
    print(f"\nAverage Dice: {total_dice/num_samples:.4f}")
    print(f"Average IoU:  {total_iou/num_samples:.4f}")


def plot_training_history(history):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Dice curve
    axes[1].plot(history['train_dice'], label='Train Dice', linewidth=2)
    axes[1].plot(history['val_dice'], label='Val Dice', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Dice Coefficient', fontsize=12)
    axes[1].set_title('Training & Validation Dice Score', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    evaluate_model(num_samples=5)