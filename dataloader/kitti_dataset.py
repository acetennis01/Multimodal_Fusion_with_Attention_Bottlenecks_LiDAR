import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from dataloader.my_pipeline_transforms import PointsToPseudoImage 
class KITTIDataset(Dataset):
    def __init__(self, root_path, split='train', transform=None):
        """
        Args:
            root_path (str): Path to the KITTI dataset root directory
            split (str): 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on images
        """
        self.root_path = root_path
        self.split = split
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load the appropriate info file
        info_file = os.path.join(root_path, f'kitti_infos_{split}.pkl')
        with open(info_file, 'rb') as f:
            self.infos = pickle.load(f)

        # Get the data list
        self.data_list = self.infos['data_list']
        
        # Get class names from metainfo
        self.class_names = self.infos['metainfo']['categories']
        self.num_classes = len(self.class_names)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        info = self.data_list[idx]
        
        # --- Robust Image Path Logic ---
        img_path_in_pkl = info['images']['CAM2']['img_path']
        # Attempt 1: Assume path in pkl is already correct relative to root_path
        img_path = os.path.join(self.root_path, img_path_in_pkl)

        if not os.path.exists(img_path):
            # If first attempt fails, and path_in_pkl looks like a bare filename
            # (e.g., "000000.png" instead of "training/image_2/000000.png")
            if not os.path.dirname(img_path_in_pkl): # True if img_path_in_pkl is just "filename.ext"
                if self.split in ['train', 'val', 'trainval']:
                    data_subdir_prefix = 'training'
                elif self.split == 'test':
                    data_subdir_prefix = 'testing'
                else:
                    # Fallback for unknown splits, or you could raise an error
                    print(f"Warning: Unknown split '{self.split}' for path construction. Defaulting to 'training' subdir.")
                    data_subdir_prefix = 'training'
                
                # Attempt 2: Construct path with known KITTI subdirectories
                corrected_img_path = os.path.join(self.root_path, data_subdir_prefix, 'image_2', img_path_in_pkl)
                
                if os.path.exists(corrected_img_path):
                    img_path = corrected_img_path
                else:
                    # If the corrected path also doesn't exist, print a warning and proceed with the original (failed) path.
                    # Image.open() will then raise a FileNotFoundError, which is informative.
                    print(f"Warning: Image file not found. Tried '{img_path}' and '{corrected_img_path}'.")
            # else: If img_path_in_pkl already contained directory components (e.g., "image_2/000000.png")
            # and the first attempt (os.path.join(self.root_path, img_path_in_pkl)) failed,
            # then the path in the .pkl is likely malformed in a way this simple heuristic can't fix.
            # We proceed with the original 'img_path' and let Image.open() fail.
        # else: First attempt was successful, img_path is correct.

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # --- Robust LiDAR Path Logic (similar to image path) ---
        lidar_path_in_pkl = info['lidar_points']['lidar_path']
        # Attempt 1: Assume path in pkl is already correct relative to root_path
        lidar_path = os.path.join(self.root_path, lidar_path_in_pkl)

        if not os.path.exists(lidar_path):
            if not os.path.dirname(lidar_path_in_pkl): # True if lidar_path_in_pkl is just "filename.bin"
                if self.split in ['train', 'val', 'trainval']:
                    data_subdir_prefix = 'training'
                elif self.split == 'test':
                    data_subdir_prefix = 'testing'
                else:
                    print(f"Warning: Unknown split '{self.split}' for path construction. Defaulting to 'training' subdir.")
                    data_subdir_prefix = 'training'
                
                # Attempt 2: Construct path with known KITTI subdirectories
                corrected_lidar_path = os.path.join(self.root_path, data_subdir_prefix, 'velodyne', lidar_path_in_pkl)
                
                if os.path.exists(corrected_lidar_path):
                    lidar_path = corrected_lidar_path
                else:
                    print(f"Warning: LiDAR file not found. Tried '{lidar_path}' and '{corrected_lidar_path}'.")
            # else: Path in pkl likely malformed beyond simple heuristic.
        # else: First attempt was successful.
        
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)  # x, y, z, intensity
        
        # --- Pseudo-image and Label Logic (remains the same) ---
        
        point_cloud_range = [-40, -40, -3, 40, 40, 1]
        points_to_pseudo = PointsToPseudoImage(image_size=(256, 256), point_cloud_range=point_cloud_range)
        pseudo_image_np = points_to_pseudo(points)
        pseudo_image = torch.from_numpy(pseudo_image_np).float()

        labels = []
        for instance in info['instances']:
            if instance['bbox_label_3d'] != -1:
                label_idx_val = instance['bbox_label_3d']
                if not (0 <= label_idx_val < self.num_classes):
                    print(f"Warning: Sample index {info.get('sample_idx', 'N/A')}, instance label {label_idx_val} is out of bounds for num_classes {self.num_classes}. Skipping this label.")
                    continue
                labels.append(label_idx_val)
        
        label_tensor = torch.zeros(self.num_classes)
        for label_idx in labels:
            label_tensor[label_idx] = 1

        return {
            'image': image,
            'pseudo_image': pseudo_image,
            'label': label_tensor,
            'sample_idx': info.get('sample_idx', idx)
        }
    

    def get_class_names(self):
        return self.class_names 