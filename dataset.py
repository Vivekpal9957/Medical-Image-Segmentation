"""
Dataset loader with synthetic medical image generation.
For real data, replace SyntheticMedicalDataset with your actual dataset.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms


class SyntheticMedicalDataset(Dataset):
    """
    Generates synthetic grayscale medical images with circular tumors.
    Perfect for demonstration without downloading large datasets.
    """
    def __init__(self, num_samples=500, img_size=256, transform=None):
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate synthetic image (grayscale medical scan-like)
        np.random.seed(idx)
        image = np.random.normal(0.3, 0.05, (self.img_size, self.img_size)).astype(np.float32)
        
        # Create a mask with random circles (simulating tumors/organs)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        num_shapes = np.random.randint(1, 4)
        
        for _ in range(num_shapes):
            center_y = np.random.randint(50, self.img_size - 50)
            center_x = np.random.randint(50, self.img_size - 50)
            radius = np.random.randint(15, 40)
            
            y, x = np.ogrid[:self.img_size, :self.img_size]
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Add shape to mask
            mask[dist <= radius] = 1.0
            
            # Add corresponding intensity to image
            image[dist <= radius] += np.random.uniform(0.2, 0.4)
        
        # Add noise for realism
        image += np.random.normal(0, 0.02, image.shape)
        image = np.clip(image, 0, 1)
        
        # Convert to tensors
        image = torch.from_numpy(image).unsqueeze(0)  # [1, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0)    # [1, H, W]
        
        return image, mask


def get_data_loaders(batch_size=4, num_workers=0):
    """Create train and validation dataloaders."""
    train_dataset = SyntheticMedicalDataset(num_samples=800, img_size=256)
    val_dataset = SyntheticMedicalDataset(num_samples=200, img_size=256)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader