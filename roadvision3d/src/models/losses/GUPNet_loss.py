import torch
from torch.nn import functional as F
import torch.nn as nn

from .focal_loss import focal_loss_cornernet
from .loss_function import extract_input_from_tensor, extract_target_from_tensor


class GUPNetLossComputation(nn.Module):
    def __init__(self):
        super().__init__()
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
            bbox3d_loss = self.compute_bbox3d_loss(preds, targets)

        seg_loss = self.compute_segmentation_loss(preds, targets)

        mean_loss = seg_loss + bbox2d_loss + bbox3d_loss
        return float(mean_loss), self.stat
    
    def compute_segmentation_loss(self, input, target):
        input['heatmap'] = torch.clamp(input['heatmap'].sigmoid_(), min=1e-4, max=1 - 1e-4)
        loss = focal_loss_cornernet(input['heatmap'], target['heatmap'])
        self.stat['seg_loss'] = loss
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

    def compute_bbox3d_loss(self, input, target, mask_type = 'mask_2d'):
        
        # compute depth loss        
        depth_input = input['depth'][input['train_tag']] 
        depth_input, depth_log_variance = depth_input[:, 0:1], depth_input[:, 1:2]
        depth_target = extract_target_from_tensor(target['depth'], target[mask_type])
        depth_loss = laplacian_aleatoric_uncertainty_loss(depth_input, depth_target, depth_log_variance)
        
        # compute offset3d loss
        offset3d_input = input['offset_3d'][input['train_tag']]  
        offset3d_target = extract_target_from_tensor(target['offset_3d'], target[mask_type])
        offset3d_loss = F.l1_loss(offset3d_input, offset3d_target, reduction='mean')
        
        # compute size3d loss
        size3d_input = input['size_3d'][input['train_tag']] 
        size3d_target = extract_target_from_tensor(target['size_3d'], target[mask_type])
        size3d_loss = F.l1_loss(size3d_input[:,1:], size3d_target[:,1:], reduction='mean')*2/3+\
               laplacian_aleatoric_uncertainty_loss(size3d_input[:,0:1], size3d_target[:,0:1], input['h3d_log_variance'][input['train_tag']])/3
        #size3d_loss = F.l1_loss(size3d_input[:,1:], size3d_target[:,1:], reduction='mean')+\
        #       laplacian_aleatoric_uncertainty_loss(size3d_input[:,0:1], size3d_target[:,0:1], input['h3d_log_variance'][input['train_tag']])
        # compute heading loss
        heading_loss = compute_heading_loss(input['heading'][input['train_tag']] ,
                                            target[mask_type],  ## NOTE
                                            target['heading_bin'],
                                            target['heading_res'])
        
        # ABS loss
        # depth_loss = torch.abs(depth_loss)
        # size3d_loss = torch.abs(size3d_loss)


        loss = depth_loss + offset3d_loss + size3d_loss + heading_loss
        
        self.stat['depth_loss'] = depth_loss
        self.stat['offset3d_loss'] = offset3d_loss
        self.stat['size3d_loss'] = size3d_loss
        self.stat['heading_loss'] = heading_loss
        
        
        return loss
    
def laplacian_aleatoric_uncertainty_loss(input, target, log_variance, reduction='mean'):
    '''
    References:
        MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships, CVPR'20
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    '''
    assert reduction in ['mean', 'sum']
    loss = 1.4142 * torch.exp(-0.5*log_variance) * torch.abs(input - target) + 0.5*log_variance
    return loss.mean() if reduction == 'mean' else loss.sum()

def compute_heading_loss(input, mask, target_cls, target_reg):
    mask = mask.view(-1)   # B * K  ---> (B*K)
    target_cls = target_cls.view(-1)  # B * K * 1  ---> (B*K)
    target_reg = target_reg.view(-1)  # B * K * 1  ---> (B*K)

    # classification loss
    input_cls = input[:, 0:12]
    target_cls = target_cls[mask]
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')
    
    # regression loss
    input_reg = input[:, 12:24]
    target_reg = target_reg[mask]
    cls_onehot = torch.zeros(target_cls.shape[0], 12).cuda().scatter_(dim=1, index=target_cls.view(-1, 1), value=1)
    input_reg = torch.sum(input_reg * cls_onehot, 1)
    reg_loss = F.l1_loss(input_reg, target_reg, reduction='mean')
    
    return cls_loss + reg_loss    

def build_GUPNet_loss(cfg):
    loss_evaluator = GUPNetLossComputation()
    return loss_evaluator


