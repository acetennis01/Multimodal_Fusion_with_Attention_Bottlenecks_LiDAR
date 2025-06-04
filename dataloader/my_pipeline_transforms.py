import numpy as np

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
        self.image_size = tuple(image_size) # Ensure it's a tuple
        self.pc_range = list(point_cloud_range) # Ensure it's a list
        
        if self.pc_range is None:
            raise ValueError("point_cloud_range must be provided.")
        if len(self.pc_range) != 6:
            raise ValueError("point_cloud_range must have 6 elements: [x_min, y_min, z_min, x_max, y_max, z_max]")

    def __call__(self, points_data): # Changed 'results' to 'points_data'
        """
        Args:
            points_data (np.ndarray): LiDAR points as a NumPy array, shape (N, 4) 
                                      for x, y, z, intensity.
        Returns:
            np.ndarray: The generated pseudo-image with shape [1, H, W].
        """
        # 1) Input 'points_data' is already the (N, 4) NumPy array.
        #    No need for:
        #    pts = results['points'].tensor
        #    pts = pts.cpu().numpy()
        #    We can directly use points_data or assign it to pts if preferred.
        pts = points_data

        # 2) Prepare an empty single‐channel image [1, H, W].
        H, W = self.image_size
        pseudo_img = np.zeros((1, H, W), dtype=np.float32)

        x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range
        range_x = x_max - x_min
        range_y = y_max - y_min

        if range_x <= 0 or range_y <= 0:
            # This check should ideally be in __init__ if pc_range is fixed,
            # but keeping it here if pc_range could somehow change per call (unlikely for this class structure)
            raise ValueError(f"Invalid point_cloud_range affecting calculated range_x or range_y in PointsToPseudoImage. range_x={range_x}, range_y={range_y}")

        # Pixel resolution
        res_x = range_x / float(W)
        res_y = range_y / float(H)
        
        if res_x <= 0 or res_y <= 0: # Should not happen if range_x/range_y are positive and W/H are positive
            raise ValueError(f"Pixel resolution is zero or negative. res_x={res_x}, res_y={res_y}")


        # 3) Fill the pseudo‐image
        #    This is a naive approach: each (x, y) is mapped to a pixel, store max z.
        for point_idx in range(pts.shape[0]):
            x, y, z = pts[point_idx, 0], pts[point_idx, 1], pts[point_idx, 2]
            # Intensity is pts[point_idx, 3], if needed

            if (x_min <= x < x_max) and (y_min <= y < y_max): # Use < for x_max, y_max for 0-based indexing safety
                # Compute pixel indices
                # Ensure casting to int happens after division
                px = int((x - x_min) / res_x)
                py = int((y - y_min) / res_y) 
                
                # Clamp pixel values to be within image dimensions
                # This is an alternative/safer way than just `if 0 <= px < W` for px when x is very close to x_max
                px = min(px, W - 1)
                py = min(py, H - 1)
                # px = max(0, px) # Not strictly necessary if x_min <= x check is robust
                # py = max(0, py) # Not strictly necessary if y_min <= y check is robust


                # Ensure indices are valid (already largely handled by clamping and initial range check)
                # if 0 <= px < W and 0 <= py < H: # This check is still good for absolute certainty
                current_val = pseudo_img[0, py, px]
                if z > current_val: # Example: store the maximum z
                    pseudo_img[0, py, px] = z
                # else: # Or you could store intensity:
                # pseudo_img[0, py, px] = pts[point_idx, 3] 
                # else: # Or count points (density)
                # pseudo_img[0, py, px] += 1

        # 4) Return only the pseudo-image NumPy array
        return pseudo_img

    def __repr__(self):
        return (f"{self.__class__.__name__}(image_size={self.image_size}, "
                f"pc_range={self.pc_range})")