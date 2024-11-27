import torch

from roadvision3d.src.datasets.object_3d import Calibration, Object3d
from roadvision3d.src.datasets.object_3d import get_affine_transform

import numpy as np
from PIL import Image

def inference(img_tensor, calib, model, cfg, device, coord_range):

    model = model.to(device)
    calib_tensor = torch.from_numpy(calib.P2).float().unsqueeze(0).to(device)
    img_tensor = img_tensor.to(device)
    coord_range = coord_range.unsqueeze(0).to(device)

    info = {
        'calibs': [calib],
        'img_id': torch.tensor([0]),  # Image ID as a tensor with batch dimension
        'img_size': torch.tensor([cfg['dataset']['resolution']]),  # Image size with batch dimension
        'bbox_downsample_ratio': torch.tensor([[4, 4]])  # Downsample ratio with batch dimension
    }


    dets = model(img_tensor, calib_tensor, coord_ranges=coord_range, mode='test', info=info)

    keys = list(dets.keys())
    first_key = keys[0]

    results = result_to_object3d(dets[first_key])

    return results


def result_to_object3d(dets):
    results = []
    for det in dets:
         # Extract each value
        cls_type = det[0]   # class type
        truncation = 0.0
        occlusion = 0.0             
        alpha = det[1]                   # alpha
        bbox2d = [
            float(det[2]),               # x1 (converted from tensor)
            float(det[3]),               # y1 (converted from tensor)
            float(det[4]),               # x2 (converted from tensor)
            float(det[5])                # y2 (converted from tensor)
        ]
        h = det[6]                       # height
        w = det[7]                       # width
        l = det[8]                       # length
        pos = [
            float(det[9]),               # x position
            float(det[10]),              # y position
            float(det[11])               # z position
        ]
        ry = float(det[12])              # rotation around y-axis (converted from tensor)
        score = det[13] if len(det) > 13 else None  # score (if exists)
        result = Object3d(cls_type, alpha, bbox2d, h, w, l, pos, ry, trucation=truncation, occlusion=occlusion, score=score)
        results.append(result)
    return results

def process_image(img_path, cfg):
    raw_img = Image.open(img_path)
    img_size = np.array(raw_img.size)

    img_resolution = np.array(cfg['dataset']['resolution'])
    # statistics
    mean = np.array(cfg['dataset']['mean'], dtype=np.float32)
    std  = np.array(cfg['dataset']['std'], dtype=np.float32)

    center = np.array(img_size) / 2
    crop_size = img_size

    trans, trans_inv = get_affine_transform(center, crop_size, 0, img_resolution, inv=1)

    img_rescaled = raw_img.transform(tuple(img_resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)

    # image encoding
    img = np.array(img_rescaled).astype(np.float32) / 255.0
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # C * H * W

    img_tensor = torch.from_numpy(img)

    # Ensure the tensor is of the correct dtype (float32) and device (CPU or CUDA)
    img_tensor = img_tensor.float().unsqueeze(0)  # Ensures tensor is in float32, if not already

    coord_range = torch.from_numpy(np.array([center-crop_size/2,center+crop_size/2]).astype(np.float32)).float()


    return img_tensor, coord_range