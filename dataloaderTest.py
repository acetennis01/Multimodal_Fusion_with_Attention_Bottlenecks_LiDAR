import os
import argparse
import importlib # To check for my_pipeline_transforms
import torch # For assertions on tensor types/shapes
import numpy as np # For type checking if needed

# --- Import the KITTIDataset class from the other file ---
# This assumes 'kitti_dataset_module.py' is in the same directory or Python path
from dataloader.kitti_dataset import KITTIDataset

def run_dataloader_test(dataset_root, split_name, num_samples_to_test=3):
    """
    Tests the KITTIDataset loader with an actual dataset.
    """
    print("--- KITTIDataset Integrity Test ---")
    print(f"Attempting to load dataset from: {dataset_root}")
    print(f"Using split: {split_name}\n")

    # 1. Check for my_pipeline_transforms.py (still useful here as a pre-check)
    try:
        importlib.import_module("my_pipeline_transforms")
        print("SUCCESS: 'my_pipeline_transforms.py' seems to be accessible.")
        print("         (KITTIDataset will attempt to import 'PointsToPseudoImage' from it).")
    except ImportError:
        print("ERROR: Could not import 'my_pipeline_transforms' module.")
        print("       Please ensure 'my_pipeline_transforms.py' (containing PointsToPseudoImage class)")
        print("       is in your Python path or the same directory as this script.")
        print("       The KITTIDataset will likely fail if this is not resolved.")
        # We can choose to exit here or let KITTIDataset fail later,
        # for now, we'll proceed and let KITTIDataset handle the direct import error.
        pass # Proceed to allow KITTIDataset to raise its own error if needed.

    # 2. Instantiate Dataloader
    try:
        dataset = KITTIDataset(root_path=dataset_root, split=split_name)
        print(f"SUCCESS: KITTIDataset initialized.")
    except FileNotFoundError as e:
        print(f"ERROR initializing KITTIDataset: {e}")
        print("Please check if the `root_path` is correct and the corresponding `kitti_infos_{split_name}.pkl` file exists there.")
        return
    except ImportError as e: # This will catch the import error from KITTIDataset if my_pipeline_transforms is missing/bad
        print(f"ERROR initializing KITTIDataset (likely from 'my_pipeline_transforms'): {e}")
        print("Please ensure 'my_pipeline_transforms.py' exists and defines 'PointsToPseudoImage'.")
        return
    except Exception as e:
        print(f"ERROR initializing KITTIDataset: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Basic Checks
    dataset_len = len(dataset)
    print(f"Total samples in '{split_name}' split: {dataset_len}")
    if dataset_len == 0:
        print("WARNING: Dataset is empty. Cannot test item retrieval.")
        return

    class_names = dataset.get_class_names()
    num_classes = dataset.num_classes
    print(f"Class names: {class_names}")
    print(f"Number of classes: {num_classes}\n")

    # 4. Test item retrieval
    indices_to_test = [0]
    if dataset_len > 1:
        indices_to_test.append(min(1, dataset_len - 1)) # Test second item if exists
    if dataset_len > 2: 
        indices_to_test.append(dataset_len // 2)
        if dataset_len -1 not in indices_to_test: indices_to_test.append(dataset_len - 1)
    
    indices_to_test = sorted(list(set(indices_to_test)))[:num_samples_to_test]
    
    count = 0

    for i, sample_idx_to_fetch in enumerate(indices_to_test):
        if sample_idx_to_fetch >= dataset_len:
            print(f"Skipping index {sample_idx_to_fetch} as it's out of bounds for dataset size {dataset_len}.")
            continue

        print(f"--- Testing sample at index: {sample_idx_to_fetch} ---")
        print(count)
        count += 1
        try:
            sample = dataset[sample_idx_to_fetch]

            assert isinstance(sample, dict), f"Sample {sample_idx_to_fetch} is not a dictionary."
            expected_keys = ['image', 'pseudo_image', 'label', 'sample_idx']
            for key in expected_keys:
                assert key in sample, f"Sample {sample_idx_to_fetch} missing key: '{key}'"
            
            print(f"  Sample keys: {list(sample.keys())}")

            img_tensor = sample['image']
            assert isinstance(img_tensor, torch.Tensor), f"Image in sample {sample_idx_to_fetch} is not a Tensor."
            print(f"  Image shape: {img_tensor.shape}, dtype: {img_tensor.dtype}")
            assert img_tensor.dim() == 3 and img_tensor.shape[0] == 3 and img_tensor.shape[1] == 224 and img_tensor.shape[2] == 224, \
                f"Unexpected image shape for sample {sample_idx_to_fetch}: {img_tensor.shape}"

            pseudo_img_tensor = sample['pseudo_image']
            assert isinstance(pseudo_img_tensor, torch.Tensor), f"Pseudo-image in sample {sample_idx_to_fetch} is not a Tensor."
            print(f"  Pseudo-image shape: {pseudo_img_tensor.shape}, dtype: {pseudo_img_tensor.dtype}")
            assert pseudo_img_tensor.dim() == 3 and pseudo_img_tensor.shape[0] == 1 and pseudo_img_tensor.shape[1] == 256 and pseudo_img_tensor.shape[2] == 256, \
                f"Unexpected pseudo-image shape for sample {sample_idx_to_fetch}: {pseudo_img_tensor.shape}"

            label_tensor = sample['label']
            assert isinstance(label_tensor, torch.Tensor), f"Label in sample {sample_idx_to_fetch} is not a Tensor."
            print(f"  Label shape: {label_tensor.shape}, dtype: {label_tensor.dtype}, value: {label_tensor}")
            assert label_tensor.shape == torch.Size([num_classes]), f"Unexpected label shape for sample {sample_idx_to_fetch}: {label_tensor.shape}"

            loaded_sample_idx = sample['sample_idx']
            print(f"  Loaded sample_idx from item: {loaded_sample_idx}")
            assert isinstance(loaded_sample_idx, (int, np.integer)), \
                 f"sample_idx {loaded_sample_idx} is not an int for dataset index {sample_idx_to_fetch}"

            print(f"SUCCESS: Sample at index {sample_idx_to_fetch} seems valid.\n")

        except FileNotFoundError as e:
            print(f"ERROR processing sample at index {sample_idx_to_fetch}: Missing file - {e}")
            print("  Please check that the paths in your .pkl file (e.g., img_path, lidar_path) are correct")
            print("  relative to the dataset_root, and that the files exist.")
        except ImportError as e: # This could be from my_pipeline_transforms within __getitem__
            print(f"ERROR processing sample at index {sample_idx_to_fetch} (ImportError): {e}")
            print("  This likely means 'my_pipeline_transforms.PointsToPseudoImage' could not be used or is missing.")
        except Exception as e:
            print(f"ERROR processing sample at index {sample_idx_to_fetch}: {e}")
            import traceback
            traceback.print_exc()
        
        if i < len(indices_to_test) -1 : print("-" * 20)

    print("--- Dataloader Test Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for KITTIDataset Dataloader.")
    parser.add_argument("dataset_root", type=str, help="Path to the root of the KITTI dataset.")
    parser.add_argument("--split", type=str, default="train", choices=['train', 'val', 'test', 'trainval'],
                        help="Dataset split to test (default: 'train').")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of samples to fetch and inspect (default: 3).")

    args = parser.parse_args()

    if not os.path.isdir(args.dataset_root):
        print(f"Error: Provided dataset_root '{args.dataset_root}' is not a valid directory.")
    else:
        run_dataloader_test(args.dataset_root, args.split, args.num_samples)