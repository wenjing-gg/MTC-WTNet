import os
import re
import nrrd
import random
from typing import Tuple
from monai.transforms import (
    RandFlipd,
    RandZoomd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandShiftIntensityd,
    ToTensord,
    ScaleIntensityRanged,
    RandSpatialCropd,
    Compose,
    RandRotated,
    RandScaleIntensityd,
    RandAdjustContrastd,
    RandBiasFieldd,
    RandGibbsNoised,
    RandKSpaceSpikeNoised,
)
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
import numpy as np
import pandas as pd

class MyNRRDDataSet(Dataset):
    """Custom NRRD format dataset that loads images and masks, processing them to (64,64,64) shape"""

    def __init__(self, root_dir: str, split: str, target_shape=(64,64,64), num_augmentations=2):
        """
        Args:
            root_dir (str): Root directory of the dataset
            split (str): Dataset split, 'train' or 'test'
            target_shape (tuple, optional): Target shape (D, H, W)
            num_augmentations (int): Number of augmented samples generated per original image (only for training)
        """
        self.split = split
        self.target_shape = target_shape
        self.num_augmentations = num_augmentations  # Only used for training set

        # Unified intensity normalization transform (used for both training and testing)
        self.intensity_transform = ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
        )

        if split == 'train':
            self.augmentations = [
                # Original geometric transformations (enhanced intensity)
                RandFlipd(keys=["image", "mask"], spatial_axis=0, prob=0.8),
                RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.8),
                RandFlipd(keys=["image", "mask"], spatial_axis=2, prob=0.8),
                RandZoomd(keys=["image", "mask"], min_zoom=0.8, max_zoom=1.2, prob=0.8),  # Increase variation range
                RandRotated(keys=["image", "mask"], range_x=0.2, range_y=0.2, range_z=0.2, prob=0.8),  # Increase rotation angle

                # Original intensity transformations (enhanced intensity)
                RandGaussianNoised(keys=["image"], mean=0.0, std=0.05, prob=0.7),  # Increase noise intensity
                RandGaussianSmoothd(keys=["image"], sigma_x=(0.1, 1.5), sigma_y=(0.1, 1.5), sigma_z=(0.1, 1.5), prob=0.7),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.7),  # Increase intensity offset
                RandScaleIntensityd(keys=["image"], factors=0.2, prob=0.7),  # Increase intensity scaling
                RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.3), prob=0.7),  # Increase contrast range

                # New advanced medical image augmentations (stronger parameters)
                RandBiasFieldd(
                    keys=["image"],
                    degree=3,
                    coeff_range=(0.0, 0.2),  # Increase coefficient range
                    prob=0.6
                ),
                RandGibbsNoised(
                    keys=["image"],
                    alpha=(0.0, 0.6),  # Increase Gibbs noise intensity
                    prob=0.5
                ),
                RandKSpaceSpikeNoised(
                    keys=["image"],
                    intensity_range=(0.9, 1.1),  # Increase K-space spike intensity range
                    prob=0.4
                ),
                # Spatial cropping at the end
                RandSpatialCropd(keys=["image", "mask"], roi_size=self.target_shape, random_center=True, random_size=False),
            ]
        else:
            self.augmentations = []

        # Convert to tensor (used for all splits)
        self.to_tensor = ToTensord(keys=["image", "mask"])

        self.data_list = []  # Store image data and labels for all samples

        # Load data
        self._load_images_from_folder(os.path.join(root_dir, split, '0'), label=0)  # NoMetastasis
        self._load_images_from_folder(os.path.join(root_dir, split, '1'), label=1)  # Metastasis

    def _load_images_from_folder(self, folder: str, label: int) -> None:
        # Modified regex pattern to add support for pure numeric filenames
        # name_pattern = re.compile(r'(?i)(hz|sz)_(\d+)|sm(\d+)_(\d+)|^(\d+)$', re.IGNORECASE)

        for filename in os.listdir(folder):
            # Only care about .nrrd images, excluding *_label.nrrd
            lower = filename.lower()
            if not lower.endswith(".nrrd") or lower.endswith("_label.nrrd"):
                continue

            img_path  = os.path.join(folder, filename)
            mask_path = os.path.join(folder, filename.replace(".nrrd", "_label.nrrd"))

            if not os.path.exists(mask_path):
                print(f"[WARN] Mask file '{os.path.basename(mask_path)}' "
                    f"not found for image '{filename}'; skipping.")
                continue

            # Read image and mask
            img        = self._process_nrrd(img_path, is_mask=False)
            seg_label  = self._process_nrrd(mask_path, is_mask=True)

            # Remove extension and perform regex matching
            basename = os.path.splitext(filename)[0]
            m = name_pattern.match(basename)

            if m:
                if m.group(1):                     # hz_ or sz_
                    prefix  = m.group(1).lower()   # "hz" / "sz"
                    num     = m.group(2)
                    id_     = f"{prefix}_{num}"
                elif m.group(3):                   # sm<num>_<num>
                    id_     = f"sm{m.group(3)}_{m.group(4)}"
                elif m.group(5):                   # Pure numbers, e.g. "15"
                    id_     = f"num_{m.group(5)}"
                else:
                    id_     = basename  # Fallback solution

                # Store filename information
                self.data_list.append((img, label, seg_label, filename))
            else:
                print(f"[WARN] Filename '{filename}' doesn't match accepted patterns; skipping.")

    def _process_nrrd(self, file_path, is_mask=False):
        """Process NRRD file and return adjusted image

        Args:
            file_path: NRRD file path
            is_mask: Whether it's a mask file
        """
        data, header = nrrd.read(file_path)
        # Convert (H, W, D) to (D, H, W)
        img = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1)

        # Ensure input is 3D data
        if img.ndim != 3:
            raise ValueError(f"Image at {file_path} is not a 3D volume.")

        img = self.interpolate_to_shape(img, self.target_shape, is_mask=is_mask)
        return img

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img, label, seg_label = self.data_list[idx]

        if self.split == 'train':
            augmented_samples = []

            for i in range(self.num_augmentations):
                # Clone a new copy of the original image
                sample_dict = {
                    "image": img.clone().unsqueeze(0),
                    "mask": seg_label.clone().unsqueeze(0)
                }

                # Randomly select and combine multiple augmentation methods (instead of just one)
                num_augs_to_apply = min(5, len(self.augmentations))  # Specify using 5 augmentation methods
                selected_augs = random.sample(self.augmentations, num_augs_to_apply)

                for aug in selected_augs:
                    sample_dict = aug(sample_dict)

                # Check if already tensor
                if not isinstance(sample_dict["image"], torch.Tensor):
                    sample_dict["image"] = torch.tensor(sample_dict["image"])
                if not isinstance(sample_dict["mask"], torch.Tensor):
                    sample_dict["mask"] = torch.tensor(sample_dict["mask"])

                # Normalize image
                normalized_img = self.normalize(sample_dict["image"])
                normalized_mask = sample_dict["mask"].long()

                augmented_samples.append((normalized_img, label, normalized_mask))

            return augmented_samples
        else:
            # Test set: no augmentation, only regular processing
            sample_dict = {
                "image": img.clone().unsqueeze(0),
                "mask": seg_label.clone().unsqueeze(0)
            }  # [C, D, H, W]
            sample_dict = self.intensity_transform(sample_dict)

            # Note: You can remove self.to_tensor(sample_dict) if needed,
            # because img itself might already be a tensor
            # sample_dict = self.to_tensor(sample_dict)  # Keep this if you need to convert to tensor again
            # Check if already tensor
            if not isinstance(sample_dict["image"], torch.Tensor):
                sample_dict["image"] = torch.tensor(sample_dict["image"])
            if not isinstance(sample_dict["mask"], torch.Tensor):
                sample_dict["mask"] = torch.tensor(sample_dict["mask"])
            # Normalize image
            normalized_img = self.normalize(sample_dict["image"])


            # Process mask, ensure type is long
            normalized_mask = sample_dict["mask"].long()

            processed_img = normalized_img
            processed_mask = normalized_mask

            return [(processed_img, label, processed_mask)]

    def normalize(self, img, mean=None, std=None):
        """Perform standard normalization on input image (zero mean, unit variance)"""
        if mean is None:
            mean = torch.mean(img).item()
        if std is None:
            std = torch.std(img).item()

        if std > 0:
            img = (img - mean) / std
        else:
            img = torch.zeros_like(img)
        return img

    def interpolate_to_shape(self, img, target_shape, is_mask=False):
        """Interpolate input 3D image to specified shape

        Args:
            img: Input 3D image tensor
            target_shape: Target shape (D, H, W)
            is_mask: Whether it's a mask, True uses nearest neighbor interpolation, False uses scipy.ndimage.zoom resampling
        """
        current_shape = img.shape
        if current_shape == target_shape:
            # print(f"Image size already matches target size {target_shape}, no adjustment needed.")
            return img

        if is_mask:
            # Mask uses nearest neighbor interpolation
            img = img.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
            img = F.interpolate(img, size=target_shape, mode='nearest')
            img = img.squeeze(0).squeeze(0)      # (D, H, W)
            return img
        else:
            # Original image uses scipy.ndimage.zoom resampling
            img_numpy = img.cpu().numpy()  # Convert to numpy array
            imx, imy, imz = img_numpy.shape
            tx, ty, tz = target_shape
            zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
            img_resampled = ndimage.zoom(img_numpy, zoom_ratio, order=0, prefilter=False)
            return torch.tensor(img_resampled, dtype=torch.float32)  # Convert back to tensor

    @staticmethod
    def collate_fn(batch):
        # batch is a list where each element is a sample (containing multiple augmented samples during training)
        # Each augmented sample is (augmented_img, class_label, seg_label)
        # or during testing it's a single (img, class_label, seg_label)
        all_samples = [sample for sublist in batch for sample in sublist]
        all_imgs, all_class_labels, all_seg_labels = zip(*all_samples)
        all_imgs = torch.stack(all_imgs, dim=0)          # [B_total, C, D, H, W]
        all_class_labels = torch.tensor(all_class_labels, dtype=torch.long)  # [B_total]
        all_seg_labels = torch.stack(all_seg_labels, dim=0)  # [B_total, C_seg, D, H, W]
        return all_imgs, all_class_labels, all_seg_labels
