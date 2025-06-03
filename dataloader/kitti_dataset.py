import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

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
        
        # Load image
        img_path = os.path.join(self.root_path, info['images']['CAM2']['img_path'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load point cloud
        lidar_path = os.path.join(self.root_path, info['lidar_points']['lidar_path'])
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)  # x, y, z, intensity
        
        # Convert points to pseudo-image using the existing transform
        from my_pipeline_transforms import PointsToPseudoImage
        point_cloud_range = [-40, -40, -3, 40, 40, 1]  # KITTI typical range
        points_to_pseudo = PointsToPseudoImage(image_size=(256, 256), point_cloud_range=point_cloud_range)
        pseudo_image = points_to_pseudo(points)
        pseudo_image = torch.from_numpy(pseudo_image).float()

        # Get labels
        labels = []
        for instance in info['instances']:
            if instance['bbox_label_3d'] != -1:  # Skip ignored instances
                labels.append(instance['bbox_label_3d'])
        
        # Convert labels to one-hot encoding
        label_tensor = torch.zeros(self.num_classes)
        for label in labels:
            label_tensor[label] = 1

        return {
            'image': image,  # [3, 224, 224]
            'pseudo_image': pseudo_image,  # [1, 256, 256]
            'label': label_tensor,  # [num_classes]
            'sample_idx': info['sample_idx']
        }

    def get_class_names(self):
        return self.class_names 