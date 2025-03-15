# my_pipeline_transforms.py

import numpy as np
from mmdet.datasets import PIPELINES

@PIPELINES.register_module()
class PointsToPseudoImage:
    """Convert raw LiDAR points to a 2D pseudo‐image (toy BEV example)."""

    def __init__(self,
                 image_size=(256, 256),
                 point_cloud_range=None):
        """
        Args:
            image_size (tuple): Desired (H, W) for the pseudo‐image.
            point_cloud_range (list|tuple): [x_min, y_min, z_min, x_max, y_max, z_max].
        """
        self.image_size = image_size
        self.pc_range = point_cloud_range
        if len(self.pc_range) != 6:
            raise ValueError("point_cloud_range must have 6 elements: [x_min, y_min, z_min, x_max, y_max, z_max]")

    def __call__(self, results):
        """
        Args:
            results (dict): The data dict, containing `points` among others.

        Returns:
            dict: The modified data dict with a new key 'lidar_pseudo_img'.
        """
        # 1) Get points (x, y, z, intensity) in [N, 4]
        pts = results['points'].tensor  # <BasePoints>.tensor -> (N, 4)
        pts = pts.cpu().numpy()

        # 2) Prepare an empty single‐channel image [1, H, W].
        H, W = self.image_size
        pseudo_img = np.zeros((1, H, W), dtype=np.float32)

        x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range
        range_x = x_max - x_min
        range_y = y_max - y_min

        if range_x <= 0 or range_y <= 0:
            raise ValueError("Invalid point_cloud_range in PointsToPseudoImage.")

        # Pixel resolution
        res_x = range_x / float(W)
        res_y = range_y / float(H)

        # 3) Fill the pseudo‐image
        #    This is a naive approach: each (x, y) is mapped to a pixel, store max z.
        for (x, y, z, intensity) in pts:
            if (x_min <= x <= x_max) and (y_min <= y <= y_max):
                # Compute pixel indices
                px = int((x - x_min) / res_x)
                py = int((y - y_min) / res_y)
                # Safety check for boundary
                if 0 <= px < W and 0 <= py < H:
                    # For example, store the maximum z encountered (you could store intensity, etc.)
                    current_val = pseudo_img[0, py, px]
                    if z > current_val:
                        pseudo_img[0, py, px] = z

        # 4) Store the result
        results['lidar_pseudo_img'] = pseudo_img  # shape [1, H, W]
        return results

    def __repr__(self):
        return (f"{self.__class__.__name__}(image_size={self.image_size}, "
                f"pc_range={self.pc_range})")
