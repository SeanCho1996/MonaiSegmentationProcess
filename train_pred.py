# %% import dependencies
import glob
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

import torch
import torch.nn as nn
from time import time

import monai
from monai.data import decollate_batch
from monai.transforms import (
    Compose,
    AsDiscrete,
    EnsureType,
)
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from dataset import MonaiDataset



# %% from dataset to dataloader
dataset = MonaiDataset(data_folder="../datasets/COVID-19-20_v2/Train/", mode="train_val")
train_ds = dataset.train_set()
val_ds = dataset.val_set()

# create a training data loader
batch_size = 4
train_loader = monai.data.DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

# create a validation data loader
val_loader = monai.data.DataLoader(
    val_ds,
    batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

# %% model and loss
num_classes = 2
model_folder = "./model/"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = monai.networks.nets.BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        features=(32, 32, 64, 128, 256, 32),
        dropout=0.1,
    )

# A test DP training strategy
model = nn.DataParallel(model) 
model.to(device)

lr= 1e-4
opt = torch.optim.Adam(model.parameters(), lr=lr)

patch_size = (192, 192, 16)
sw_batch_size, overlap = 4, 0.5

loss_function = DiceLoss(to_onehot_y=True, softmax=True)
dice_metric = DiceMetric(include_background=False, reduction="mean")
# %% train
max_epochs = 600
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
    start_time = time()
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
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
                    val_data["image"].to(device),
                    val_data["label"].to(device),
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
                torch.save(model.state_dict(), os.path.join(
                    model_folder, "best_metric_model.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )
        print(f"epoch time = {time() - start_time}")
        
# %% inference
infer_ds = MonaiDataset(data_folder="../datasets/COVID-19-20_v2/Validation", mode="infer").infer_set()

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

model.load_state_dict(torch.load(
    "./model/best_metric_model.pth"))
model.eval()

with torch.no_grad():
    for infer_data in infer_loader:
        print(f"segmenting {infer_data['img_meta_dict']['filename_or_obj']}")
        preds = inferer(infer_data["img"].to(device), model)
        preds = (preds.argmax(dim=1, keepdims=True)).float()
        saver.save_batch(preds, infer_data["img_meta_dict"])

# %%
