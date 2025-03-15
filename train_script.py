import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from dataloader.av_data import KITTIMultiDriveDataset
from models.visual_model import AVmodel
import os
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm
import torchvision.transforms.functional as TF



def parse_options():
    parser = argparse.ArgumentParser(description="Multimodal Bottleneck Attention with KITTI Dataset")

    ##### TRAINING DYNAMICS
    parser.add_argument('--gpu_id', type=str, default="cpu", help='the GPU id')
    parser.add_argument('--lr', type=float, default=3e-4, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=3, help='batch size')  # Lower default batch size
    parser.add_argument('--num_epochs', type=int, default=15, help='total training epochs')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')

    ##### ADAPTER AND LATENT PARAMETERS
    parser.add_argument('--adapter_dim', type=int, default=768, help='dimension of the low-rank adapter')
    parser.add_argument('--num_latent', type=int, default=4, help='number of latent tokens')
    parser.add_argument('--num_classes', type=int, default=7, help='number of output classes')

    ##### DATA
    parser.add_argument('--data_root', type=str, default='/Users/abhiramannaluru/Documents/data/raw_data_downloader/2011_09_26', help='path to KITTI dataset')

    opts = parser.parse_args()
    torch.manual_seed(opts.seed)

    opts.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # if opts.gpu_id.lower() == "cpu" or not torch.cuda.is_available():
    #     opts.device = torch.device("cpu")
    # else:
    #     opts.device = torch.device(f"cuda:{opts.gpu_id}")  # Updated GPU ID handling
    return opts

def lidar_to_histogram_features(lidar, crop=256):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """

    def splat_points(point_cloud):
        # 256 x 256 grid
        pixels_per_meter = 16
        hist_max_per_pixel = 5
        x_meters_max = 28
        y_meters_max = 56

        # # Increase the range and shift it to the right
        # x_start = 0           # start at 0 meters (or a positive offset)
        # x_end = 100            # extend further to the right
        # pixels_per_meter = 8

        # xbins = np.linspace(
        #     x_start,
        #     x_end,
        #     (x_end - x_start) * pixels_per_meter + 1,
        # )


        xbins = np.linspace(
            -4 * x_meters_max,
            4 * x_meters_max + 1,
            2 * x_meters_max * pixels_per_meter + 1,
        )
        # ybins = np.linspace(-y_meters_max, 0, y_meters_max * pixels_per_meter + 1)
        ybins = np.linspace(-y_meters_max, y_meters_max, y_meters_max * pixels_per_meter + 1)
        
        hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
        hist[hist > hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist / hist_max_per_pixel
        return overhead_splat

    below = lidar[lidar[..., 2] <= -2.0]
    above = lidar[lidar[..., 2] > -2.0]
    below_features = splat_points(below)
    above_features = splat_points(above)
    total_features = below_features + above_features
    features = np.stack([below_features, above_features, total_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)  # Shape: (3, H, W)
    return features


'''
def collate_fn(batch):
    point_clouds, rgb_frames, timestamps, oxts_data = [], [], [], []

    output_dir = "pseudo_images"
    
    i = 0

    
    for point_cloud, rgb_frame, timestamp, oxts in batch:
        #print(f"Point cloud shape: {point_cloud.shape}")
        
        # Transform the LiDAR point cloud to a pseudo-image
        pseudo_image = lidar_to_histogram_features(point_cloud)
        
        # Print the shape of the pseudo-image before and after adding batch dimension
        #print(f"Pseudo-image shape (before unsqueeze): {pseudo_image.shape}")
        
        # Ensure the pseudo-image has 3 channels (for compatibility with the encoder)
        # Here we don't use unsqueeze(0) since we're going to stack the images later.
        pseudo_image = torch.tensor(pseudo_image)  # Shape should be [3, 224, 224]
        
        #print(f"Pseudo-image shape (after converting to tensor): {pseudo_image.shape}")
        
        point_clouds.append(pseudo_image)
        rgb_frames.append(rgb_frame)
        timestamps.append(timestamp)
        oxts_data.append(oxts)

        # print(pseudo_image)

        output_path = os.path.join(output_dir, f"pseudo_image_{i}.png")
        torchvision.utils.save_image(pseudo_image, output_path)

        i += 1

    
    # Stack the point clouds and other data to create a batch
    # Now point_clouds is a list of tensors of shape [3, 224, 224]
    point_clouds = torch.stack(point_clouds)  # Shape: [B, 3, 224, 224]
    rgb_frames = torch.stack(rgb_frames)
    oxts_data = torch.stack(oxts_data)

    # Print final shapes to ensure correctness
    #print(f"Final point clouds batch shape: {point_clouds.shape}")
    
    return point_clouds, rgb_frames, timestamps, oxts_data
'''
def collate_fn(batch):

    filtered_batch = [sample for sample in batch if sample[-1][-1].item() != -1]
    if len(filtered_batch) == 0:
        return None

    point_clouds, rgb_frames, timestamps, oxts_data = [], [], [], []
    output_dir = "pseudo_images"
    i = 0

    for point_cloud, rgb_frame, timestamp, oxts in filtered_batch:
        #print("OXTS sample:", oxts)
        pseudo_image = lidar_to_histogram_features(point_cloud)
        # Debug: print the min/max values and shape of the pseudo_image
        #print(f"Sample {i} pseudo_image: min={pseudo_image.min()}, max={pseudo_image.max()}, shape={pseudo_image.shape}")
        
        # Optionally, if you expect a certain size (e.g., [3,224,224]) but the histogram is larger,
        # you can resize the image. For example:
        # import torchvision.transforms.functional as TF
        # pseudo_image = TF.resize(torch.tensor(pseudo_image), [224, 224]).numpy()
        
        pseudo_image = torch.tensor(pseudo_image)  # Convert to tensor
        pseudo_image = TF.resize(pseudo_image.unsqueeze(0), size=[224, 224], interpolation=TF.InterpolationMode.BILINEAR).squeeze(0)

        point_clouds.append(pseudo_image)
        rgb_frames.append(rgb_frame)
        timestamps.append(timestamp)
        oxts_data.append(oxts)

        output_path = os.path.join(output_dir, f"pseudo_image_{i}.png")
        torchvision.utils.save_image(pseudo_image, output_path)
        i += 1

    # Stack tensors
    point_clouds = torch.stack(point_clouds)  # Expected shape: [B, 3, H, W]
    rgb_frames = torch.stack(rgb_frames)
    oxts_data = torch.stack(oxts_data)
    return point_clouds, rgb_frames, timestamps, oxts_data


def train_one_epoch(train_data_loader, model, optimizer, loss_fn, device, args):
    epoch_loss = []
    sum_correct_pred = 0
    total_samples = 0

    model.train()
    scaler = GradScaler()  # Gradient scaler for mixed precision

    # Wrap the data loader with tqdm for progress bar
    progress_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc="Training", leave=False)
    
    
    for batch_idx, (point_clouds, rgb_frames, _, oxts_data) in progress_bar:
        # Move data to device
        point_clouds = point_clouds.to(device)
        rgb_frames = rgb_frames.to(device)
        oxts_data = oxts_data.to(device)
        
        optimizer.zero_grad()

        # Run the model in full precision (FP32) by NOT using autocast
        preds = model(point_clouds, rgb_frames)
        labels = oxts_data[:, -1].long().to(device)
        if labels.min().item() < 0 or labels.max().item() >= args.num_classes:
            tqdm.write(f"Invalid labels detected: min={labels.min().item()}, max={labels.max().item()}")
        _loss = loss_fn(preds, labels)
        
        # Optionally print if loss is NaN
        if torch.isnan(_loss):
            print("Loss is NaN!")
        
        _loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss.append(_loss.item())
        sum_correct_pred += (torch.argmax(preds, dim=1) == labels).sum().item()
        total_samples += len(labels)

        progress_bar.set_postfix(loss=np.mean(epoch_loss))

    

    '''
    for batch_idx, (point_clouds, rgb_frames, _, oxts_data) in progress_bar:
        # Move data to device
        point_clouds = point_clouds.to(device)
        rgb_frames = rgb_frames.to(device)
        oxts_data = oxts_data.to(device)
        
        optimizer.zero_grad()


        # with autocast():
        #     preds = model(point_clouds, rgb_frames)
        #     labels = oxts_data[:, -5].long().to(device)  # Ensure labels are on the correct device
        #     if labels.min().item() < 0 or labels.max().item() >= 28:
        #         tqdm.write(f"Invalid labels detected: min={labels.min().item()}, max={labels.max().item()}")

        #     # _loss = loss_fn(preds, labels).to(device)
        #     _loss = loss_fn(preds, labels)

        with autocast():
            preds = model(point_clouds, rgb_frames)
            # Debug: check if preds contain NaNs or are extremely large
            if torch.isnan(preds).any():
                print("NaN detected in model predictions!")
            max_val = preds.abs().max().item()
            if max_val > 1e4:
                print("Model predictions are very large:", max_val)
            labels = oxts_data[:, -1].long().to(device)
            if labels.min().item() < 0 or labels.max().item() >= args.num_classes:
                tqdm.write(f"Invalid labels detected: min={labels.min().item()}, max={labels.max().item()}")
            _loss = loss_fn(preds, labels)


        scaler.scale(_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss.append(_loss.item())
        sum_correct_pred += (torch.argmax(preds, dim=1) == labels).sum().item()
        total_samples += len(labels)

        # Update tqdm's progress bar with current metrics
        progress_bar.set_postfix(loss=np.mean(epoch_loss))
    '''
    
    acc = round(sum_correct_pred / total_samples, 5) * 100
    return np.mean(epoch_loss), acc


def val_one_epoch(val_data_loader, model, loss_fn, device):
    epoch_loss = []
    sum_correct_pred = 0
    total_samples = 0

    model.eval()

    # Wrap the validation data loader with tqdm for progress bar
    progress_bar = tqdm(val_data_loader, total=len(val_data_loader), desc="Validation", leave=False)
    
    with torch.no_grad():
        for point_clouds, rgb_frames, _, oxts_data in progress_bar:
            point_clouds = point_clouds.to(device)
            rgb_frames = rgb_frames.to(device)
            oxts_data = oxts_data.to(device)

            preds = model(point_clouds, rgb_frames)
            labels = oxts_data[:, -1].long()
            _loss = loss_fn(preds, labels)

            if torch.isnan(_loss):
                print("Loss is NaN!")

            epoch_loss.append(_loss.item())
            sum_correct_pred += (torch.argmax(preds, dim=1) == labels).sum().item()
            total_samples += len(labels)

            progress_bar.set_postfix(loss=np.mean(epoch_loss))
            progress_bar.refresh()
    
    acc = round(sum_correct_pred / total_samples, 5) * 100
    return np.mean(epoch_loss), acc


def train_test(args):

    dataset = KITTIMultiDriveDataset(root_dir=args.data_root)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    trainloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=0  # Avoid multiprocessing issues
    )

    valloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0  # Avoid multiprocessing issues
    )

    model = AVmodel(num_classes=args.num_classes, num_latents=args.num_latent, dim=args.adapter_dim)
    model.to(args.device)
    print("\t Model Loaded")
    print('\t Trainable params = ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    best_val_acc = []
    num_epochs = args.num_epochs

    # Wrap the epoch loop with tqdm
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        torch.cuda.empty_cache()  # Clear memory before each epoch
        loss, acc = train_one_epoch(trainloader, model, optimizer, loss_fn, args.device, args)
        val_loss, val_acc = val_one_epoch(valloader, model, loss_fn, args.device)

        print('\nEpoch:', epoch + 1)
        print("Training loss & accuracy:", round(loss, 4), round(acc, 3))
        print("Validation loss & accuracy:", round(val_loss, 4), round(val_acc, 3))
        best_val_acc.append(val_acc)

    print("\n\t Completed Training \n")  
    print("\t Best Results:", np.max(np.asarray(best_val_acc)))

if __name__ == "__main__":
    opts = parse_options()
    train_test(args=opts)

