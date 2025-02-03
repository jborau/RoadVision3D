# Training and Testing with RoadVision3D

This document provides an overview of how to train and test 3D object detection models using RoadVision3D.

---

## Training a Model

### 1. Configure Training Parameters

Training parameters are defined in YAML configuration files located in the `roadvision3d/configs/` directory. Below is an example configuration:

```yaml
dataset:
  type: 'kitti'
  data_dir: '/path/to/kitti'
  batch_size: 24
  num_workers: 16
  eval_cls: ['Car','Pedestrian','Cyclist']

model:
  type: 'SMOKE'
  backbone: 'dla34'
  head: 'SmokeHead'
  downsample: 4

optimizer:
  type: 'adam'
  lr: 0.001
  weight_decay: 0.00001

trainer:
  max_epoch: 150
  eval_frequency: 15
  save_frequency: 50
  log_dir: 'work_dirs/SMOKE/logs/'
```

For a complete list of configuration options, refer to [configs.md](docs/configs.md).

### 2. Run Training

Execute the following command to start training:

```bash
python roadvision3d/tools/train_val.py --config path/to/config.yaml
```

For example, to train MonoLSS on KITTI dataset for 150 epochs:

```bash
python roadvision3d/tools/train_val.py --config roadvision3d/configs/MonoLSS/MonoLSS_KITTI_150e.yaml
```


### 3. Monitor Training Progress on wandb

RoadVision3D supports logging with **Weights & Biases (wandb)**, which is **enabled by default**.  
Before using wandb, ensure you are logged in. If you haven't set up wandb, follow the [official wandb login guide](https://docs.wandb.ai/quickstart#1.-set-up-wandb).

To explicitly enable logging:

```bash
WANDB_MODE=online python train_val.py --config configs/model_name.yaml
```

If you prefer to disable logging, use:

```bash
WANDB_MODE=disabled python train_val.py --config configs/model_name.yaml
```
---

## Testing a Model

To evaluate or test a trained model, specify the checkpoint file:

```bash
# Evaluate
python train_val.py --config path/to/config.yaml --evaluate --resume_model /path/to/checkpoint.pth

# Test
python train_val.py --config path/to/config.yaml --test --resume_model /path/to/checkpoint.pth
```
---

## Configuration

The training and testing process relies on configuration files located in the `roadvision3d/configs/` directory. These YAML files define dataset parameters, model architecture, optimizer settings, and training schedules.

For a detailed explanation of configuration options, refer to [configs.md](docs/configs.md).

---

## Notes

- Ensure all dependencies are installed before training/testing.
- Use a GPU for efficient training and inference.
- Modify the configuration files to suit your dataset and model.

