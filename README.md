# MonaiSegmentationProcess
A full torch-like 3d segmentation process (from training to validation and to prediction)using Monai

Collated from Monai official tutorials: [Baseline for Covid19 Segmentation Competition](https://github.com/Project-MONAI/tutorials/blob/0c7add4a48615433e44dd8a5a7d12e7ef153ee24/3d_segmentation/challenge_baseline/run_net.py) and [Unet Training Tutorial](https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/torch/unet_training_array.py).

To further improve the efficiency (not really), I rewrote the original DP (data parallel) mode into a DDP (distributed data parallel) mode which corresponds to `train_pred_ddp.py`. 
As expected, using DDP means there is no "master" GPU to ensamble the gradient and all gpus are in a real "parallel" situation. Therefore, the batch-size can be subtly increased so that the speed can be improved a little.

to run:(DP mode)
```shell
python train_pred.py
```
(DDP mode)
```shell
CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node 4 train_pred_ddp.py 
```