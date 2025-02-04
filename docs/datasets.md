# Dataset Usage in RoadVision3D

This document provides an overview of how to use datasets with RoadVision3D, including structure, splitting, and preprocessing.

---

## Supported Datasets

RoadVision3D currently supports the following datasets:

- **KITTI**: Benchmark dataset for 3D object detection and tracking.
- **DAIR-V2X**: Focused on vehicle-to-infrastructure (V2I) perception.
- **ROPE3D**: Specialized for roadside infrastructure-based object detection and pose estimation.

For a detailed explanation of dataset configuration, refer to the YAML files in `roadvision3d/configs/`.

---

## Dataset Details

### **KITTI**
- **Overview**: A widely used benchmark for 3D object detection, primarily used for autonomous driving research.
- **Data Modalities**: RGB images, LiDAR point clouds, calibration files, annotations.
- **Considerations**: No ground truth for the test set; data splits are stored in `ImageSets/`.

### **DAIR-V2X Infrastructure**
- **Overview**: A dataset for vehicle-to-infrastructure (V2I) perception, focusing on roadside sensor data.
- **Data Modalities**: Camera images, LiDAR, cooperative perception data.
- **Format Support**: DAIR-V2X can be used in **KITTI format** by setting `DAIR_KITTI` in the configuration.

### **ROPE3D**
- **Overview**: A dataset designed for roadside infrastructure-based object detection and pose estimation.
- **Data Modalities**: Camera images.

---

## Dataset Splitting

Each dataset should be split into training, validation, and testing sets as follows:

- **Training Set**: Used for model training and should be listed in `train.txt`.
- **Validation Set**: Used for hyperparameter tuning and early stopping, listed in `val.txt`.
- **Test Set**: Used for final evaluation, listed in `test.txt`.

The predefined splits for **KITTI**, **DAIR-V2X**, and **ROPE3D** are located in their respective `splits/` directories. The paths to these split directories should be specified in the dataset configuration file under the `split_dir` parameter, for example:

```yaml
dataset:
  type: 'kitti'
  data_dir: '/path/to/kitti_data'
  split_dir: '/path/to/kitti/ImageSets'
```

---

## Dataset Directory Structure

Each dataset should be structured in the following way:

### **KITTI Dataset Structure**
```
/path/to/kitti/
│── training/
│   ├── image_2/          # RGB images
│   ├── label_2/          # Annotations
│   ├── calib/            # Calibration files
│   ├── velodyne/         # LiDAR point clouds
│── testing/
│   ├── image_2/
│   ├── calib/
│   ├── velodyne/
│── ImageSets/
│   ├── train.txt         # Training split
│   ├── val.txt           # Validation split
│   ├── test.txt          # Test split
```

### **DAIR-V2X Dataset Structure**
```
/path/to/dair_v2x/single-infrastructure-side/
│── image/               
│── velodyne/             
│── label/
│   ├── camera/
│   ├── virtuallidar/
│── calib/
│   ├── camera_intrinsic/
│   ├── virtuallidar_to_camera/
│── splits/
│   ├── train.txt
│   ├── val.txt
│   ├── trainval.txt   
│   ├── test.txt
```
To use **DAIR-V2X** in **KITTI format**, set `DAIR_KITTI` as the dataset type in the configuration.

### **ROPE3D Dataset Structure**
```
/path/to/rope3d/
│── training/
│   ├── calib/
│   ├── denorm/
│   ├── depth_2/
│   ├── extrinsics/
│   ├── image_2/
│   ├── label_2/
│── validation/
│   ├── calib/
│   ├── denorm/
│   ├── depth_2/
│   ├── extrinsics/
│   ├── image_2/
│   ├── label_2/
│── splits/
│   ├── train.txt
│   ├── val.txt
│   ├── test.txt
```

Ensure that your dataset structure follows the format above. The `split_dir` paths could be outside the dataset directory but should always be correctly specified in the configuration files.

---

## Configuring a Dataset

To use a dataset, update the corresponding configuration file in `roadvision3d/configs/`. Below is an example for **KITTI**:

```yaml
dataset:
  type: 'kitti'
  data_dir: '/path/to/kitti_data'
  split_dir: '/path/to/kitti/ImageSets'
  batch_size: 24
  num_workers: 16
  eval_cls: ['Car','Pedestrian','Cyclist']
```

For more details on dataset configuration, refer to the specific dataset config files in `roadvision3d/configs/`.

---

