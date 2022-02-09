import glob
import os

import monai
from monai.transforms import (
    Compose,
    AddChanneld,
    AsDiscrete,
    CastToTyped,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    EnsureTyped,
    EnsureType,
)


def get_transforms(mode="train", keys=("image", "label")):
    xforms = [
        LoadImaged(keys),
        AddChanneld(keys),
        Orientationd(keys, axcodes="LPS"),
        Spacingd(keys, pixdim=(1.25, 1.25, 5.0), mode=("bilinear", "nearest")[: len(keys)]),
        ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
    ]
    if mode == "train":
        xforms.extend(
            [
                SpatialPadd(keys, spatial_size=(192, 192, -1), mode="reflect"),  # ensure at least 192x192
                RandAffined(
                    keys,
                    prob=0.15,
                    rotate_range=(0.05, 0.05, None),  # 3 parameters control the transform on 3 dimensions
                    scale_range=(0.1, 0.1, None),
                    mode=("bilinear", "nearest"),
                    as_tensor_output=False,
                ),
                RandCropByPosNegLabeld(keys, label_key=keys[1], spatial_size=(192, 192, 16), num_samples=3),
                RandGaussianNoised(keys[0], prob=0.15, std=0.01),
                RandFlipd(keys, spatial_axis=0, prob=0.5),
                RandFlipd(keys, spatial_axis=1, prob=0.5),
                RandFlipd(keys, spatial_axis=2, prob=0.5),
            ]
        )
        dtype = (np.float32, np.uint8)
    if mode == "val":
        dtype = (np.float32, np.uint8)
    if mode == "infer":
        dtype = (np.float32,)
    xforms.extend([CastToTyped(keys, dtype=dtype), EnsureTyped(keys)])
    return Compose(xforms)


class MonaiDataset:
    def __init__(self, data_folder, train_frac=0.8, mode="train_val") -> None:
        # data_folder = "/home/zhaozixiao/projects/datasets/COVID-19-20_v2/Train/"
        images = sorted(glob.glob(os.path.join(data_folder, "*_ct.nii.gz")))
        labels = sorted(glob.glob(os.path.join(data_folder, "*_seg.nii.gz")))
        print(f"training: total image/label: ({len(images)}) from folder: {data_folder}")

        if mode == "train_val":
            self.keys = ("image", "label")
            val_frac = 1 - train_frac
            n_train = int(train_frac * len(images)) + 1
            n_val = min(len(images) - n_train, int(val_frac * len(images)))
            print(f"split: train {n_train} val {n_val}, folder: {data_folder}")

            self.train_files = [{self.keys[0]: img, self.keys[1]: seg} for img, seg in zip(images[:n_train], labels[:n_train])]
            self.val_files = [{self.keys[0]: img, self.keys[1]: seg} for img, seg in zip(images[-n_val:], labels[-n_val:])]
        elif mode == "infer":
            self.keys = ("img",)
            self.infer_files = [{"img": img} for img in data_folder]

    def train_set(self):
        train_transforms = get_transforms(mode="train", keys=self.keys)
        train_ds = monai.data.CacheDataset(data=self.train_files, transform=train_transforms)
        return train_ds

    def val_set(self):
        val_transforms = get_transforms(mode="val", keys=self.keys)
        val_ds = monai.data.CacheDataset(data=self.val_files, transform=val_transforms)
        return val_ds
    
    def infer_set(self):
        infer_transforms = get_transforms(mode="infer", keys=self.keys)
        infer_ds = monai.data.Dataset(data=self.infer_files, transform=infer_transforms)
        return infer_ds
