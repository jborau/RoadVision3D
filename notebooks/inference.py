import numpy as np
from PIL import Image
import torch

from roadvision3d.src.datasets.kitti_utils import get_affine_transform
from roadvision3d.src.datasets.kitti import KITTI
from roadvision3d.src.engine.dataloader import build_dataloader
from roadvision3d.src.engine.model_builder import build_model
from roadvision3d.src.engine.model_saver import load_checkpoint
from roadvision3d.src.engine.decode_helper import extract_dets_from_outputs
from roadvision3d.src.engine.decode_helper import decode_detections
from roadvision3d.src.datasets.kitti_utils import Object3d


def preprocess_image(img, dataset):
    img_size = np.array(img.size)
    # data augmentation for image
    center = np.array(img_size) / 2
    crop_size = img_size

    trans, trans_inv = get_affine_transform(center, crop_size, 0, dataset.resolution, inv=1)

    img_rescaled = img.transform(tuple(dataset.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)

    # image encoding
    img = np.array(img_rescaled).astype(np.float32) / 255.0
    img = (img - dataset.mean) / dataset.std
    img = img.transpose(2, 0, 1)  # C * H * W

    img_tensor = torch.from_numpy(img)

    # Ensure the tensor is of the correct dtype (float32) and device (CPU or CUDA)
    img_tensor = img_tensor.float()  # Ensures tensor is in float32, if not already

    coord_range = torch.from_numpy(np.array([center-crop_size/2,center+crop_size/2]).astype(np.float32)).float()


    return img_tensor, coord_range

def prepare_data(dataset, data_id):
    # Load and process the image
    img_raw = dataset.get_image(data_id)
    img_torch, coord_range = preprocess_image(img_raw, dataset)

    # Load and process the calib
    calib = dataset.get_calib(8)
    calib_torch = torch.from_numpy(calib.P2).float()

    return img_torch, calib_torch, coord_range

def inference_on_dataset(data_id, split, cfg, device):
    dataset = KITTI(split=split, cfg=cfg['dataset'])


    model = build_model(cfg['model'], cfg['dataset']['cls_mean_size'])
    load_checkpoint(model = model,
                        optimizer = None,
                        filename = cfg['tester']['resume_model'],
                        map_location=device)    

    img_tensor, calib_tensor, coord_ranges_tensor = prepare_data(dataset=dataset, data_id=data_id)

    img_tensor = img_tensor.unsqueeze(0).to(device)
    calib_tensor = calib_tensor.unsqueeze(0).to(device)
    coord_ranges_tensor = coord_ranges_tensor.unsqueeze(0).to(device)

    model = model.to(device)

    outputs = model(img_tensor, coord_ranges_tensor, calib_tensor,K=50,mode='test')
    dets = extract_dets_from_outputs(outputs=outputs, K=50)
    dets = dets.detach().cpu().numpy()

    # get corresponding calibs & transform tensor to numpy
    calib_object = dataset.get_calib(data_id)
    calibs = [calib_object]  # treat it as a batch

    info = {
        'img_id': torch.tensor([data_id]),  # Image ID as a tensor with batch dimension
        'img_size': torch.tensor([[1242, 375]]),  # Image size with batch dimension
        'bbox_downsample_ratio': torch.tensor([[3.8813, 3.9062]])  # Downsample ratio with batch dimension
    }

    info = {key: val.detach().cpu().numpy() for key, val in info.items()}


    # cls_mean_size = KITTI(split='train', cfg=cfg['dataset']).cls_mean_size
    cls_mean_size = cfg['dataset']['cls_mean_size']
    print(cls_mean_size)
    dets = decode_detections(dets = dets,
                            info = info,
                            calibs = calibs,
                            cls_mean_size=cls_mean_size,
                            threshold = cfg['tester']['threshold']
                            )
    
    results_lines = convert_to_kitti_line(dets[data_id])
    objects = [Object3d(line) for line in results_lines]

    return objects

def convert_to_kitti_line(inference_output):
    label_format = "{cls_type} {trucation:.6f} {occlusion:.6f} {alpha:.6f} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {h:.6f} {w:.6f} {l:.6f} {x:.6f} {y:.6f} {z:.6f} {ry:.6f} {score:.6f}"
    cls_type = "Car"  # Assuming the class type is Car, update if needed
    lines = []
    for obj in inference_output:
        trucation = 0.0  # Assuming truncation is 0, update if needed
        occlusion = 0.0  # Assuming occlusion is 0, update if needed
        alpha = obj[1]
        x1, y1, x2, y2 = obj[2], obj[3], obj[4], obj[5]
        h, w, l = obj[6], obj[7], obj[8]
        x, y, z = obj[9], obj[10], obj[11]
        ry = obj[12]
        score = obj[13]
        
        line = label_format.format(
            cls_type=cls_type,
            trucation=trucation,
            occlusion=occlusion,
            alpha=alpha,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            h=h,
            w=w,
            l=l,
            x=x,
            y=y,
            z=z,
            ry=ry,
            score=score
        )
        lines.append(line)
    return lines
