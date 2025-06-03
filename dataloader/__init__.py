from torch.utils.data import DataLoader
from .kitti_dataset import KITTIDataset

def create_dataloaders(root_path, batch_size=32, num_workers=4):
    """
    Create training and validation dataloaders for KITTI dataset.
    
    Args:
        root_path (str): Path to the KITTI dataset root directory
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    # Create datasets
    train_dataset = KITTIDataset(root_path=root_path, split='train')
    val_dataset = KITTIDataset(root_path=root_path, split='val')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
