import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np 
from .smoke_coder import SMOKECoder
from .focal_loss import focal_loss
from roadvision3d.src.engine.decode_helper import _transpose_and_gather_feat
# from smoke.layers.utils import select_point_of_interest


class SMOKELossComputation(nn.Module):
    def __init__(self,
                 smoke_coder,
                 cls_loss,
                 reg_loss,
                 loss_weight,
                 max_objs,
                 device,
                 cls_mean_size):
        super().__init__()
        self.smoke_coder = smoke_coder
        self.cls_loss = cls_loss
        self.reg_loss = reg_loss
        self.loss_weight = loss_weight
        self.max_objs = max_objs
        self.device = device
        self.cls_mean_size = torch.tensor(cls_mean_size, device=device)
        self.stat = {}

    def forward(self, preds, targets, calibs, info):
        if targets['mask_2d'].sum() == 0:
            device = targets['mask_2d'].device  # Get the device of the tensor
            bbox2d_loss = torch.tensor(0.0, device=device)
            bbox3d_loss = torch.tensor(0.0, device=device)
            self.stat['offset2d_loss'] = torch.tensor(0.0, device=device)
            self.stat['size2d_loss'] = torch.tensor(0.0, device=device)
            self.stat['position_loss'] = torch.tensor(0.0, device=device)
            self.stat['dimension_loss'] = torch.tensor(0.0, device=device)
            self.stat['rotation_loss'] = torch.tensor(0.0, device=device)
        else:
            bbox2d_loss = self.compute_bbox2d_loss(preds, targets)
            bbox3d_loss = self.compute_bbox3d_loss(preds, targets, calibs, info)

        seg_loss = self.compute_segmentation_loss(preds, targets)

        mean_loss = seg_loss + bbox2d_loss + bbox3d_loss
        return float(mean_loss), self.stat
    
    def compute_segmentation_loss(self, input, target):
        input['heatmap'] = torch.clamp(input['heatmap'].sigmoid_(), min=1e-4, max=1 - 1e-4)
        loss = focal_loss(input['heatmap'], target['heatmap']) * 5.0
        self.stat['heatmap_loss'] = loss
        return loss
    
    def compute_bbox2d_loss(self, input, target):
        # compute size2d loss
        
        size2d_input = extract_input_from_tensor(input['size_2d'], target['indices'], target['mask_2d'])
        size2d_target = extract_target_from_tensor(target['size_2d'], target['mask_2d'])
        size2d_loss = F.l1_loss(size2d_input, size2d_target, reduction='mean')

        # compute offset2d loss
        offset2d_input = extract_input_from_tensor(input['offset_2d'], target['indices'], target['mask_2d'])
        offset2d_target = extract_target_from_tensor(target['offset_2d'], target['mask_2d'])
        offset2d_loss = F.l1_loss(offset2d_input, offset2d_target, reduction='mean')

        loss = offset2d_loss + size2d_loss   
        self.stat['offset2d_loss'] = offset2d_loss
        self.stat['size2d_loss'] = size2d_loss
        return loss
    
    def compute_bbox3d_loss(self, input, target, calibs, info):
        B, M = target['mask_2d'].shape
        device = input['heatmap'].device
        size3d_input = extract_input_from_tensor(input['size_3d_offset'], target['indices'], target['mask_2d'])
        pred_size_3d = size3d_input.exp() * torch.tensor(self.cls_mean_size, device=size3d_input.device)
        size3d_target = extract_target_from_tensor(target['size_3d_smoke'], target['mask_2d'])
        # print(f"Dims pred: {[f'{value:.2f}' for value in pred_size_3d[0].tolist()]}", 
        # f"Dims target: {[f'{value:.2f}' for value in size3d_target[0].tolist()]}")
        size3d_loss = F.l1_loss(pred_size_3d, size3d_target, reduction='mean')

        depth_offsets = extract_input_from_tensor(input['depth'], target['indices'], target['mask_2d'])
        pred_depths = self.smoke_coder.decode_depth(depth_offsets)
        # target_depths = extract_target_from_tensor(target['depth'], target['mask_2d'])
        # depth_loss = F.l1_loss(pred_depth, target_depth, reduction='mean')

        # Recover the x and y coordinates from the flattened indices
        w = input['heatmap'].shape[3]  # Get width of the feature map
        h = input['heatmap'].shape[2]  # Get height of the feature map
        x_coords = target['indices'] % w
        y_coords = target['indices'] // w

        # Filter out invalid entries using the mask
        valid_x_coords = x_coords[target['mask_2d']]
        valid_y_coords = y_coords[target['mask_2d']]

        # Stack x and y into a tensor of shape [N, 2], where N is the total number of objects
        points = torch.stack([valid_x_coords, valid_y_coords], dim=-1)
        pred_proj_offsets = extract_input_from_tensor(input['offset_3d'], target['indices'], target['mask_2d'])


        # Create a batch index array of shape [B, M]
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, M)
        mask = target['mask_2d'].bool()

        # Flatten and select only valid entries
        obj_batch_idx = batch_indices.flatten()[mask.flatten()]  # shape: [N], where N is number of valid objects   

        # Extract Ks and invert for each object
        Ks = calibs[:, :3, :3]  # shape [B, 3, 3]
        Ks_inv = Ks.inverse()[obj_batch_idx]  # shape [N, 3, 3], aligned with points

        downsamples = info['bbox_downsample_ratio'].to(device=device)
        downsamples_per_object = downsamples.unsqueeze(1).expand(B, M, 2)
        filtered_downsamples = downsamples_per_object.flatten(end_dim=1)[mask.flatten()]

        pred_locations = self.smoke_coder.decode_location(
            points,
            pred_proj_offsets,
            pred_depths,
            Ks_inv,
            downsample_ratio=filtered_downsamples,
        )
        # Correctly compute the full predicted height
        pred_height = pred_size_3d[:, 0]  # Get the predicted height

        # Shift the predicted center to the bottom of the bounding box
        pred_locations[:, 1] += pred_height / 2
        target_locations = extract_target_from_tensor(target['position'], target['mask_2d'])
        # Print formatted values directly
        # print(f"Pos pred: {[f'{value:.2f}' for value in pred_locations[0].tolist()]}", 
        # f"pos target: {[f'{value:.2f}' for value in target_locations[0].tolist()]}")
       
        position_loss = F.l1_loss(pred_locations, target_locations, reduction='mean')

        # ---- Orientation Loss ----
        # Extract predicted orientation in [sin, cos] format
        pred_ori = extract_input_from_tensor(input['ori'], target['indices'], target['mask_2d'])

        # Decode predicted orientation into rotys (and alphas, if needed)
        pred_rotys, pred_alphas = self.smoke_coder.decode_orientation(pred_ori, pred_locations)

        # Extract target roty, which is already given in a decoded angle format
        target_rotys = extract_target_from_tensor(target['rotation_y'], target['mask_2d'])

        # print('Rotys pred:', pred_rotys[0].item(), 'Rotys target:', target_rotys[0].item())
        # print('\n')
        # Compute rotation loss between decoded predicted rotys and target rotys
        rotation_loss = F.l1_loss(pred_rotys, target_rotys, reduction='mean')


        loss = size3d_loss + position_loss + rotation_loss
        self.stat['dimension_loss'] = size3d_loss * 40.0
        self.stat['position_loss'] = position_loss * 20.0
        self.stat['rotation_loss'] = rotation_loss * 10.0
        return loss



def build_smoke_loss(cfg, device):
    smoke_coder_depth_reference = cfg['model']['depth_ref']
    smoke_coder_dimension_reference = cfg['dataset']['cls_mean_size']
    smoke_device = device
    # smoke_coder = SMOKECoder(
    #     cfg.MODEL.SMOKE_HEAD.DEPTH_REFERENCE,
    #     cfg.MODEL.SMOKE_HEAD.DIMENSION_REFERENCE,
    #     cfg.MODEL.DEVICE,
    # )
    # focal_loss = FocalLoss(
    #     cfg.MODEL.SMOKE_HEAD.LOSS_ALPHA,
    #     cfg.MODEL.SMOKE_HEAD.LOSS_BETA,
    # )
    smoke_coder = SMOKECoder(
        smoke_coder_depth_reference,
        smoke_coder_dimension_reference,
        smoke_device,
    )

    loss_evaluator = SMOKELossComputation(
        smoke_coder,
        cls_loss= focal_loss,
        # reg_loss=cfg.MODEL.SMOKE_HEAD.LOSS_TYPE[1],
        reg_loss = "DisL1",
        # loss_weight=cfg.MODEL.SMOKE_HEAD.LOSS_WEIGHT,
        loss_weight = cfg['model']['loss_weight'],
        # max_objs=cfg.DATASETS.MAX_OBJECTS,
        max_objs = 50,
        device=smoke_device,
        cls_mean_size=cfg['dataset']['cls_mean_size'],
        )

    return loss_evaluator

### ======================  auxiliary functions  =======================

def extract_input_from_tensor(input, ind, mask):
    input = _transpose_and_gather_feat(input, ind)  # B*C*H*W --> B*K*C
    return input[mask]  # B*K*C --> M * C


def extract_target_from_tensor(target, mask):
    return target[mask]