# %% import dependencies
import glob
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

import numpy as np
import time
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import monai
from monai.data import decollate_batch
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
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
# from monai.networks.layers import Norm

# %% ddp configuration
# parser = argparse.ArgumentParser()
# parser.add_argument("--local_rank", default=-1, type=int)
# FLAGS = parser.parse_args()
# local_rank = FLAGS.local_rank
dist.init_process_group(backend='nccl')
local_rank = dist.get_rank()

torch.cuda.set_device(local_rank)

# %% transforms: different for train, validation and inference
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


# %% dataset
data_folder = "../datasets/COVID-19-20_v2/Train/"
images = sorted(glob.glob(os.path.join(data_folder, "*_ct.nii.gz")))
labels = sorted(glob.glob(os.path.join(data_folder, "*_seg.nii.gz")))
print(f"training: total image/label: ({len(images)}) from folder: {data_folder}")

keys = ("image", "label")
train_frac, val_frac = 0.8, 0.2
n_train = int(train_frac * len(images)) + 1
n_val = min(len(images) - n_train, int(val_frac * len(images)))
print(f"split: train {n_train} val {n_val}, folder: {data_folder}")

train_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[:n_train], labels[:n_train])]
val_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[-n_val:], labels[-n_val:])]

# create a training data loader
batch_size = 2
print(f"batch size {batch_size}")
train_transforms = get_transforms("train", keys)
train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
train_loader = monai.data.DataLoader(
    train_ds,
    batch_size=batch_size,
    # shuffle=True,
    num_workers=2,
    pin_memory=False,#torch.cuda.is_available(),
    sampler=train_sampler
)

# create a validation data loader
val_transforms = get_transforms("val", keys)
val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms)
val_loader = monai.data.DataLoader(
    val_ds,
    batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
    num_workers=2,
    pin_memory=False#torch.cuda.is_available(),
)

# %% model and loss
num_classes = 2
model_folder = "./model/"
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model = monai.networks.nets.UNet(
#     spatial_dims=3,
#     in_channels=1,
#     out_channels=num_classes,
#     channels=(16, 32, 64, 128, 256),
#     strides=(2, 2, 2, 2),
#     num_res_units=2,
#     norm=Norm.BATCH,
# )
model = monai.networks.nets.BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        features=(32, 32, 64, 128, 256, 32),
        dropout=0.1,
    )

model.to(local_rank)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

lr= 1e-4
opt = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = DiceLoss(to_onehot_y=True, softmax=True).to(local_rank)
dice_metric = DiceMetric(include_background=False, reduction="mean")

patch_size = (192, 192, 16)
sw_batch_size, overlap = 4, 0.5
# %% train
max_epochs = 1
val_interval = 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    start_time = time.time()
    model.train()
    epoch_loss = 0
    step = 0

    train_loader.sampler.set_epoch(epoch)

    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(local_rank),
            batch_data["label"].to(local_rank),
        )
        opt.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}, "
            f"train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(local_rank),
                    val_data["label"].to(local_rank),
                )
                val_outputs = sliding_window_inference(val_inputs, patch_size, sw_batch_size, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)

            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            print(f"val dice: {metric}")
            # reset the status for next validation round
            dice_metric.reset()

            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                if dist.get_rank() == 0 and model_folder is not None:
                    torch.save(model.state_dict(), os.path.join(
                        model_folder, "best_metric_model_ddp.pth"))
                    print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )
        print(f"epoch time = {time.time() - start_time}")
        
time.sleep(5)
# %% inference
keys = ("img",)
infer_transforms = get_transforms(mode="infer", keys=keys)

img_path = "../datasets/COVID-19-20_v2/Train/volume-covid19-A-0698_ct.nii.gz"
# img = infer_transforms(img_path)
infer_files = [{"img": img} for img in [img_path]]

infer_ds = monai.data.Dataset(data=infer_files, transform=infer_transforms)
infer_loader = monai.data.DataLoader(
    infer_ds,
    batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

inferer = monai.inferers.SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
        padding_mode="replicate",
    )
saver = monai.data.NiftiSaver(output_dir="out", mode="nearest")

model_path = "./model/best_metric_model_ddp.pth"
# if dist.get_rank() == 0 and model_path is not None:
m = torch.load(model_path)
model.load_state_dict(m)
model.eval()

with torch.no_grad():
    for infer_data in infer_loader:
        print(f"segmenting {infer_data['img_meta_dict']['filename_or_obj']}")
        preds = inferer(infer_data[keys[0]].to(local_rank), model)
        preds = (preds.argmax(dim=1, keepdims=True)).float()
        saver.save_batch(preds, infer_data["img_meta_dict"])

# %%
