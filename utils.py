"""
Utility functions for training and evaluation.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt


def dice_coefficient(pred, target, smooth=1e-5):
    """
    Dice Similarity Coefficient (F1-score for segmentation).
    Range: 0 to 1 (higher is better)
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()


def iou_score(pred, target, smooth=1e-5):
    """
    Intersection over Union (Jaccard Index).
    Range: 0 to 1 (higher is better)
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum(dim=(2, 3))
    total = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    union = total - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def visualize_prediction(image, mask, pred, save_path=None):
    """
    Visualize: Original Image | Ground Truth | Prediction | Overlay
    """
    pred = torch.sigmoid(pred)
    pred_mask = (pred > 0.5).float()
    
    image = image.cpu().numpy().squeeze()
    mask = mask.cpu().numpy().squeeze()
    pred_mask = pred_mask.cpu().numpy().squeeze()
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original Image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Ground Truth
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Prediction', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Overlay
    overlay = np.zeros((*image.shape, 3))
    overlay[..., 0] = image  # Red channel = image
    overlay[..., 1] = image  # Green channel = image
    overlay[..., 2] = image  # Blue channel = image
    overlay[..., 0] = np.maximum(overlay[..., 0], pred_mask * 0.8)  # Red overlay
    axes[3].imshow(np.clip(overlay, 0, 1))
    axes[3].set_title('Overlay (Red=Pred)', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    # plt.show()
    plt.savefig("results/training_history.png", dpi=300)
    plt.close()


class EarlyStopping:
    """Stop training when validation loss stops improving."""
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0