# Configuration Guide for RoadVision3D

This document explains how to use and modify configuration files in RoadVision3D. Configuration files are stored in YAML format and are used to specify dataset parameters, model architecture, optimizer settings, and training schedules.

---

## Understanding a Configuration File

A typical configuration file consists of several key sections:

- **Dataset Configuration**: Defines the dataset location, preprocessing settings, and evaluation parameters.
- **Model Configuration**: Specifies the model type, backbone, and loss functions.
- **Optimizer Configuration**: Defines learning rate, weight decay, and optimization strategy.
- **Scheduler Configuration**: Controls learning rate decay.
- **Trainer Configuration**: Manages training duration, evaluation frequency, and checkpoint saving.
- **Tester Configuration**: Handles testing parameters such as confidence thresholds and model checkpoint paths.
- **WandB Logging Configuration**: Controls Weights & Biases (wandb) integration for experiment tracking.

---

## 1. Dataset Configuration

```yaml
dataset:
  type: 'kitti'  # Dataset type (kitti, dair, rope3d, etc.)
  data_dir: '/path/to/kitti'  # Root directory of the dataset
  label_dir: '/path/to/kitti/training/label_2'  # Directory containing labels
  split_dir: '/path/to/kitti/ImageSets'  # Directory for train/val/test splits
  eval_cls: ['Car','Pedestrian','Cyclist']  # Classes to evaluate
  batch_size: 24  # Batch size for training
  num_workers: 16  # Number of parallel data loading workers
  class_merging: False  # Whether to merge similar classes
  use_dontcare: False  # Whether to use "DontCare" labeled regions
  use_3d_center: True  # Use 3D center as target
  writelist: ['Car','Pedestrian','Cyclist']
  random_flip: 0.5  # Probability of random horizontal flip
  random_crop: 0.5  # Probability of random cropping
  random_mix: 0.5  # Probability of randomly mixing images
  scale: 0.4  # Scaling factor for augmentations
  shift: 0.1  # Shift range for augmentations
  drop_last_val: True # Drop last iteration if batch is not completed
  cls_mean_size: [[1.69076256, 1.95625980, 4.59149442], # classes mean size of the dataset
                  [1.64686968, 0.56291266, 0.52403671],
                  [1.42871768, 0.64937362, 1.68556033]]
  max_objs: 100
  resolution: [1280, 384]  # Input image resolution
  mean: [0.485, 0.456, 0.406] # Img processing
  std: [0.229, 0.224, 0.225]
  downsample: 4
```

---

## 2. Model Configuration

```yaml
model:
  type: 'SMOKE'  # Model type
  backbone: 'dla34'  # Backbone network (e.g., ResNet, DLA, etc.)
  neck: 'DLAUp'  # Feature aggregation method
  head: 'SmokeHead'  # Detection head
  downsample: 4  # Downsampling factor for feature extraction
  loss_weight: [10., 5.]  # Weights for different loss components
  depth_ref: [28.01, 16.32]  # Depth estimation reference values
```

---

## 3. Optimizer Configuration

```yaml
optimizer:
  type: 'adam'  # Optimization algorithm
  lr: 0.001  # Initial learning rate
  weight_decay: 0.00001  # Weight decay
```

---

## 4. Learning Rate Scheduler

```yaml
lr_scheduler:
  warmup: True  # Use warmup for gradual learning rate increase
  decay_rate: 0.1  # Learning rate decay factor
  decay_list: [60, 90, 120]  # Epochs at which learning rate decays
```

---

## 5. Trainer Configuration

```yaml
trainer:
  max_epoch: 150  # Total number of training epochs
  eval_start: 15  # Epoch to start evaluation
  eval_frequency: 15  # Frequency of evaluation (every N epochs)
  save_frequency: 50  # Frequency of checkpoint saving (every N epochs)
  disp_frequency: 10  # Interval for displaying training progress
  log_dir: 'work_dirs/SMOKE_KITTI_150e/logs/'  # Path to store logs
  out_dir: 'work_dirs/SMOKE_KITTI_150e/output/'  # Output directory for results
  resume_model: ''  # Path to resume training from checkpoint (if any)
```

---

## 6. Tester Configuration

```yaml
tester:
  threshold: 0.2  # Confidence threshold for detections
  top_k: 50  # Maximum number of output detections
  out_dir: './SMOKE_KITTI_150e/testset_out'  # Output directory for test results
  resume_model: ''  # Path to pre-trained model for testing
```

---

## 7. Weights & Biases (wandb) Configuration

Manages experiment tracking using **Weights & Biases**. Logging is enabled by default.

```yaml
wandb:
  name: "SMOKE_KITTI_150e"  # Experiment name
  notes: "SMOKE trained on KITTI for 150 epochs"  # Notes for tracking
```

**To explicitly enable wandb logging:**
```bash
WANDB_MODE=online python train_val.py --config configs/model_name.yaml
```

**To disable wandb logging:**
```bash
WANDB_MODE=disabled python train_val.py --config configs/model_name.yaml
```

---

## Conclusion

Configuration files allow RoadVision3D to be highly flexible and customizable. By modifying these files, users can easily adapt the framework to different datasets, models, and training strategies. 

For a complete list of configuration options, refer to the specific YAML files in `roadvision3d/configs/`.
