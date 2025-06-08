import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# --- Import your custom modules ---
try:
    from kitti_dataset_module import KITTIDataset
    from model import AVmodel
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Please ensure kitti_dataset_module.py and model.py are in the current directory or PYTHONPATH.")
    exit()

try:
    import my_pipeline_transforms
except ImportError as e:
    print(f"Error: my_pipeline_transforms.py not found or accessible: {e}")
    print("This file is required by KITTIDataset.")
    exit()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, print_freq=50):
    model.train()
    running_loss = 0.0
    processed_samples = 0

    for i, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        pseudo_images = batch['pseudo_image'].to(device)
        labels = batch['label'].to(device)
        images_t = images.unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(pc=pseudo_images, rgb=images_t)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        processed_samples += images.size(0)

        if (i + 1) % print_freq == 0:
            avg_loss = running_loss / processed_samples
            print(f"Epoch [{epoch+1}], Batch [{i+1}/{len(dataloader)}], Loss: {avg_loss:.4f}")

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions_exact_match = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            pseudo_images = batch['pseudo_image'].to(device)
            labels = batch['label'].to(device)
            images_t = images.unsqueeze(1)

            outputs = model(pc=pseudo_images, rgb=images_t)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct_predictions_exact_match += (preds == labels).all(dim=1).sum().item()
            total_samples += labels.size(0)

    val_loss = running_loss / len(dataloader.dataset)
    val_accuracy_emr = correct_predictions_exact_match / total_samples if total_samples > 0 else 0.0
    return val_loss, val_accuracy_emr

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"Using device: {device}")

    # --- Datasets and DataLoaders ---
    train_dataset = KITTIDataset(root_path=args.dataset_root, split='train')
    val_dataset = KITTIDataset(root_path=args.dataset_root, split='val')

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    num_classes = train_dataset.num_classes
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {train_dataset.get_class_names()}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # --- Model, Loss, Optimizer ---
    model = AVmodel(num_classes=num_classes, num_latents=args.num_latents, dim=args.dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- NEW: Logic to resume from checkpoint ---
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'")
            # Load checkpoint to the current device
            checkpoint = torch.load(args.resume, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            # Load best_val_loss to correctly save the next best model
            best_val_loss = checkpoint.get('val_loss', float('inf')) 
            
            print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            print(f"   Resuming training from epoch {start_epoch + 1}")
        else:
            print(f"=> ERROR: No checkpoint found at '{args.resume}'. Starting from scratch.")
    # --- End of new logic ---


    # --- Training Loop ---
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"\nStarting training from epoch {start_epoch + 1} up to {args.epochs}...")

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, print_freq=args.print_freq)
        val_loss, val_accuracy_emr = validate(model, val_loader, criterion, device)

        epoch_duration = time.time() - start_time
        print(f"-"*50)
        print(f"Epoch [{epoch+1}/{args.epochs}] - Duration: {epoch_duration:.2f}s")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val EMR Accuracy: {val_accuracy_emr:.4f}")
        print(f"-"*50)

        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1, # Save the *next* epoch number to start from
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy_emr': val_accuracy_emr
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model to {best_model_path} (Val Loss: {best_val_loss:.4f})")

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for AVmodel on KITTI.")
    parser.add_argument('--dataset_root', type=str, required=True, help="Path to the root of the KITTI dataset.")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help="Directory to save model checkpoints.")
    parser.add_argument('--resume', type=str, default=None, help="Path to the checkpoint to resume training from.") # NEW
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay for optimizer.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training and validation.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_tument('--num_latents', type=int, default=128, help="Number of latents for AdaptFormer.")
    parser.add_argument('--dim', type=int, default=768, help="Dimension for features (should match ViT and PointCloudEncoder output).")
    parser.add_argument('--print_freq', type=int, default=20, help="Frequency of printing training loss (batches).")
    parser.add_argument('--force_cpu', action='store_true', help="Force training on CPU even if CUDA is available.")
    
    args = parser.parse_args()
    main(args)