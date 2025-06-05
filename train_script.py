import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize # For default dataset transform

# --- Import your custom modules ---
# Ensure these files are in the same directory or your PYTHONPATH
try:
    from dataloader.kitti_dataset import KITTIDataset
    from models.visual_model import AVmodel # This file should contain AVmodel, PointCloudEncoder, AdaptFormer, QuickGELU
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Please ensure kitti_dataset_module.py and model.py are in the current directory or PYTHONPATH.")
    exit()

try:
    import dataloader.my_pipeline_transforms # Check if accessible, PointsToPseudoImage is imported within KITTIDataset
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
        # sample_indices = batch['sample_idx'] # Not used in training directly

        # AVmodel expects RGB input as [B, T, C, H, W]. Our Dataloader gives [B, C, H, W].
        # Add a time dimension T=1.
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

            # For multi-label accuracy (exact match ratio)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct_predictions_exact_match += (preds == labels).all(dim=1).sum().item()
            total_samples += labels.size(0)

    val_loss = running_loss / len(dataloader.dataset)
    val_accuracy_emr = correct_predictions_exact_match / total_samples if total_samples > 0 else 0.0
    return val_loss, val_accuracy_emr

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")

    # --- Datasets and DataLoaders ---
    # Using default transforms from KITTIDataset if not overridden
    # You might want to define specific transforms for train and val
    train_dataset = KITTIDataset(root_path=args.dataset_root, split='train')
    val_dataset = KITTIDataset(root_path=args.dataset_root, split='val') # Assuming a 'val' split exists

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: One or both datasets are empty. Please check your dataset path and .pkl files.")
        return

    num_classes = train_dataset.num_classes
    class_names = train_dataset.get_class_names()
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # --- Model ---
    # AVmodel(num_classes, num_latents, dim=768, vit_model_name='vit_base_patch16_224', adaptformer_blocks=12)
    # dim, vit_model_name, adaptformer_blocks are defaults in your model.py
    model = AVmodel(num_classes=num_classes, num_latents=args.num_latents, dim=args.dim).to(device)
    print(f"Model AVmodel initialized with num_classes={num_classes}, num_latents={args.num_latents}, dim={args.dim}.")
    # You can print model summary here if desired: print(model)

    # --- Loss Function ---
    # For multi-label classification with logits output and multi-hot labels
    criterion = nn.BCEWithLogitsLoss()

    # --- Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- Learning Rate Scheduler (Optional) ---
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # --- Training Loop ---
    best_val_loss = float('inf')
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, print_freq=args.print_freq)
        val_loss, val_accuracy_emr = validate(model, val_loader, criterion, device)

        # if scheduler:
        #     scheduler.step()

        epoch_duration = time.time() - start_time
        print(f"-"*50)
        print(f"Epoch [{epoch+1}/{args.epochs}] - Duration: {epoch_duration:.2f}s")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val EMR Accuracy: {val_accuracy_emr:.4f}")
        print(f"-"*50)

        # Save checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
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
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay for optimizer.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training and validation.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument('--num_latents', type=int, default=128, help="Number of latents for AdaptFormer.")
    parser.add_argument('--dim', type=int, default=768, help="Dimension for features (should match ViT and PointCloudEncoder output).")
    parser.add_argument('--print_freq', type=int, default=20, help="Frequency of printing training loss (batches).")
    parser.add_argument('--use_cuda', action='store_true', help="Use CUDA if available (default is to use if available).")
    # If you don't pass --use_cuda, it will default to False (unless you change the logic),
    # so the device line `torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")`
    # is a bit redundant with just checking torch.cuda.is_available().
    # A simpler approach is to let the script auto-detect CUDA and only add a --no_cuda flag if needed.
    # For now, this setup works: if CUDA is available AND --use_cuda is passed, it uses CUDA.
    # To make it simpler: remove --use_cuda and just rely on torch.cuda.is_available().
    # I'll modify the device line to be simpler if --use_cuda is not explicitly used.

    args = parser.parse_args()

    # Simplified device selection (auto-use CUDA if available, unless user specifically wants CPU later via a --cpu flag)
    if args.use_cuda and not torch.cuda.is_available():
        print("Warning: --use_cuda specified, but CUDA is not available. Using CPU.")
        args.use_cuda = False # Fallback to CPU
    elif not args.use_cuda and torch.cuda.is_available():
        print("CUDA is available, but --use_cuda was not specified. Using CPU. To use CUDA, pass the --use_cuda flag.")


    main(args)