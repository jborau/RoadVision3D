import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from roadvision3d.src.datasets.utils import angle2class
from roadvision3d.src.datasets.utils import gaussian_radius
from roadvision3d.src.datasets.utils import draw_umich_gaussian
from roadvision3d.src.datasets.utils import get_angle_from_box3d,check_range
from roadvision3d.src.datasets.utils import encode_targets
from roadvision3d.src.datasets.object_3d import Calibration
from roadvision3d.src.datasets.object_3d import get_affine_transform
from roadvision3d.src.datasets.object_3d import affine_transform
from roadvision3d.src.datasets.object_3d import compute_box_3d
from .data_augmentation import DataAugmention

import cv2 as cv
import torchvision.ops.roi_align as roi_align
import math
from roadvision3d.src.datasets.object_3d import Object3d

class Rope3D(data.Dataset):
    def __init__(self, split, cfg):
        # basic configuration
        self.num_classes = cfg['cls_num']
        self.max_objs = cfg['max_objs']
        self.class_name = cfg['eval_cls']
        self.cls2id = {cls: idx for idx, cls in enumerate(self.class_name)}
        self.resolution = np.array(cfg['resolution'])

        self.use_3d_center = cfg['use_3d_center']
        self.writelist = cfg['writelist']
        if cfg['class_merging']:
            self.writelist.extend(['Van', 'Truck'])
        if cfg['use_dontcare']:
            self.writelist.extend(['DontCare'])

        # Load mean size for each class
        self.cls_mean_size = np.array(cfg['cls_mean_size'])                                
                              
        assert split in ['train', 'val', 'trainval', 'test']
        self.split = split
        split_dir = os.path.join(cfg['split_dir'], split + '.txt')

        self.idx_list = [x.strip() for x in open(split_dir).readlines()]

        # path configuration
        # if split in ['train', 'val', 'trainval']:
        if split in ['train', 'val', 'trainval']:
            self.data_dir = os.path.join(cfg['data_dir'], 'training')
        else:
            self.data_dir = os.path.join(cfg['data_dir'], 'validation')
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.depth_dir = os.path.join(self.data_dir, 'depth_2')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')
        
        # data augmentation configuration
        self.data_augmentation = DataAugmention(cfg, dataset=self)

        # statistics
        self.mean = np.array(cfg['mean'], dtype=np.float32)
        self.std  = np.array(cfg['std'], dtype=np.float32)

        # others
        self.downsample = cfg['downsample']

    def get_image(self, idx):
        # Use idx directly as part of the filename
        img_file = os.path.join(self.image_dir, f'{idx}.jpg')
        
        # Check that the file exists
        assert os.path.exists(img_file), f"Image file does not exist: {img_file}"
        
        # Open and return the image in RGB mode
        return Image.open(img_file)
    
    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, f'{idx}.txt')
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, f'{idx}.txt')
        assert os.path.exists(calib_file)
        return Calibration.from_rope3d_calib_file(calib_file)

    def __len__(self):
        return self.idx_list.__len__()
    
    def __getitem__(self, item):
        # ================== 1) Load fundamental data ==================
        index = self.idx_list[item]  # index mapping
        img = self.get_image(index)
        img_size = np.array(img.size, dtype=np.float32)  # (W,H)
        calib = self.get_calib(index)

        # If training, get label-objects
        if self.split in ['train', 'trainval']:
            objects = self.get_label(index)
            # 2) Apply data augmentation => returns possibly updated (img, calib, center, crop_size, objects)
            img, calib, center, crop_size, objects = self.data_augmentation(
                img, calib, objects
            )
        else:
            objects = []
            center = img_size / 2.0
            crop_size = img_size

        # ================== 3) Affine Transform to final resolution ==================
        trans, trans_inv = get_affine_transform(
            center, crop_size, 0, self.resolution, inv=1
        )
        img = img.transform(
            tuple(self.resolution.tolist()),
            method=Image.AFFINE,
            data=tuple(trans_inv.reshape(-1).tolist()),
            resample=Image.BILINEAR
        )

        coord_range = np.array([center - crop_size / 2, center + crop_size / 2], dtype=np.float32)

        # ================== 4) Convert image to tensor format ==================
        img = np.array(img, dtype=np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # (C,H,W)

        features_size = self.resolution // self.downsample  # [W', H']

        # ================== 5) If NOT train => no label encoding ==================
        if self.split not in ['train', 'trainval']:
            inputs = img
            targets = {}
            info = {
                'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size / features_size
            }
            return inputs, calib.P2, coord_range, targets, info

        # ================== 6) Encode targets in a helper function ==================
        targets = encode_targets(
                        objects=objects,
                        calib=calib,
                        trans=trans,
                        features_size=features_size,
                        num_classes=self.num_classes,
                        max_objs=self.max_objs,
                        use_3d_center=self.use_3d_center,
                        downsample=self.downsample,
                        cls_mean_size=self.cls_mean_size,
                        cls2id=self.cls2id,
                        writelist=self.writelist
                    )
        
        # ================== 7) Final return ==================
        inputs = img
        info = {
            'img_id': index,
            'img_size': img_size,
            'bbox_downsample_ratio': img_size / features_size
        }
        return inputs, calib.P2, coord_range, targets, info
    

def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object3d.from_kitti_line(line) for line in lines]
    return objects