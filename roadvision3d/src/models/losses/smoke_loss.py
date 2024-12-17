import torch
from torch.nn import functional as F

from .smoke_coder import SMOKECoder
from .focal_loss import focal_loss
# from smoke.layers.utils import select_point_of_interest


class SMOKELossComputation():
    def __init__(self,
                 smoke_coder,
                 cls_loss,
                 reg_loss,
                 loss_weight,
                 max_objs,
                 device):
        self.smoke_coder = smoke_coder
        self.cls_loss = cls_loss
        self.reg_loss = reg_loss
        self.loss_weight = loss_weight
        self.max_objs = max_objs
        self.device = device
        self.stat = {}

    def prepare_targets(self, targets):

        depth = targets['depth']
        size_2d = targets['size_2d']
        heatmap = targets['heatmap']
        offset_2d = targets['offset_2d']
        indices = targets['indices']
        size_3d = targets['size_3d']
        offset_3d = targets['offset_3d']
        heading_bin = targets['heading_bin']
        heading_res = targets['heading_res']
        cls_ids = targets['cls_ids']
        mask_2d = targets['mask_2d']
        vis_depth = targets['vis_depth']
        rotation_y = targets['rotation_y']
        position = targets['position']

        return heatmap, dict(depth=depth, size_2d=size_2d, offset_2d=offset_2d, 
                            indices=indices, size_3d=size_3d, offset_3d=offset_3d,
                            heading_bin=heading_bin, heading_res=heading_res,
                            cls_ids=cls_ids, mask_2d=mask_2d, vis_depth=vis_depth,
                            rotation_y=rotation_y, position=position)


    def prepare_predictions(self, targets_variables, pred_regression, calibs):
        batch, channel = pred_regression.shape[0], pred_regression.shape[1]
        targets_proj_points = targets_variables["indices"]

        # obtain prediction from points of interests
        pred_regression_pois = select_point_of_interest(
            batch, targets_proj_points, pred_regression
        )
        pred_regression_pois = pred_regression_pois.view(-1, channel)
        # FIXME: fix hard code here
        pred_depths_offset = pred_regression_pois[:, 0]
        pred_proj_offsets = pred_regression_pois[:, 1:3]
        pred_dimensions_offsets = pred_regression_pois[:, 3:6]
        pred_orientation = pred_regression_pois[:, 6:]    

        pred_depths = self.smoke_coder.decode_depth(pred_depths_offset)

        # Assuming pred_regression is your feature map output
        w = pred_regression.shape[3]  # Get width of the feature map

        # Recover the x and y coordinates from the flattened indices
        x_coords = targets_proj_points % w
        y_coords = targets_proj_points // w

        # Flatten the batch and object dimensions
        x_coords = x_coords.view(-1)
        y_coords = y_coords.view(-1)

        # Stack x and y into a tensor of shape [N, 2], where N is the total number of objects
        points = torch.stack([x_coords, y_coords], dim=-1)

        pred_locations = self.smoke_coder.decode_location(
            points,
            pred_proj_offsets,
            pred_depths,
            calibs,
            downsample_ratio=4, # CHECK
        )
        # pred_dimensions = self.smoke_coder.decode_dimension(
        #     targets_variables["cls_ids"],
        #     pred_dimensions_offsets,
        # )

        pred_dimensions = pred_dimensions_offsets #.exp() # if self.use_log_space else pred_dimensions_offsets

        # we need to change center location to bottom location
        pred_locations[:, 1] += pred_dimensions[:, 1] / 2

        pred_rotys, alphas = self.smoke_coder.decode_orientation(
            pred_orientation,
            pred_locations,
            # targets_variables["flip_mask"]
        )



        return dict(rot=pred_rotys,
                    dim=pred_dimensions,
                    loc=pred_locations, )

        # elif self.reg_loss == "L1":
        #     pred_box_3d = self.smoke_coder.encode_box3d(
        #         pred_rotys,
        #         pred_dimensions,
        #         pred_locations
        #     )
        #     return pred_box_3d

    def __call__(self, predictions, targets, calibs):
        pred_heatmap, pred_regression = predictions[0], predictions[1]

        # targets_heatmap, targets_regression, targets_variables \
        #     = self.prepare_targets(targets)
        targets_heatmap, targets_variables = self.prepare_targets(targets)

        predict_boxes3d = self.prepare_predictions(targets_variables, pred_regression, calibs)

        # hm_loss = self.cls_loss(pred_heatmap, targets_heatmap, alpha=2., beta=4.) * self.loss_weight[0]
        hm_loss = self.cls_loss(pred_heatmap, targets_heatmap)

        # Reshape predictions
        batch_size, max_objs = targets_variables["mask_2d"].shape
        orientation_pred = predict_boxes3d['rot'].view(batch_size, max_objs)
        dimension_pred = predict_boxes3d['dim'].view(batch_size, max_objs, 3)
        position_pred = predict_boxes3d['loc'].view(batch_size, max_objs, 3)

        # Move target tensors to the same device
        orientation_target = targets_variables["rotation_y"].to(self.device).view(batch_size, max_objs)
        dimension_target = targets_variables["size_3d"].to(self.device)
        position_target = targets_variables["position"].to(self.device)


        mask_2d = targets_variables["mask_2d"].to(self.device)

        # Create valid mask
        valid_mask = mask_2d.bool()

        # Select valid predictions and targets
        orientation_pred_valid = orientation_pred[valid_mask]
        orientation_target_valid = orientation_target[valid_mask]
        dimension_pred_valid = dimension_pred[valid_mask]
        dimension_target_valid = dimension_target[valid_mask]
        position_pred_valid = position_pred[valid_mask]
        position_target_valid = position_target[valid_mask]

        # print('location pred: ', position_pred_valid[0].detach().cpu().numpy(), 
        #     position_target_valid[0].detach().cpu().numpy())
            
        # print('dimensions pred: ', dimension_pred_valid[0].detach().cpu().numpy(), 
        #     dimension_target_valid[0].detach().cpu().numpy())
        
        # print('orientation pred: ', orientation_pred_valid[0].detach().cpu().numpy(), 
        #     orientation_target_valid[0].detach().cpu().numpy())


        num_objs = valid_mask.sum()
        # Compute L1 loss
        if self.reg_loss == 'DisL1':
            if num_objs > 0:                # reg_loss_ori = F.l1_loss(orientation_pred_valid, orientation_target_valid) / (self.loss_weight[1] * num_objs)
                # reg_loss_dim = F.l1_loss(dimension_pred_valid, dimension_target_valid) / (self.loss_weight[1] * num_objs)
                reg_loss_ori = F.l1_loss(orientation_pred_valid, orientation_target_valid) * self.loss_weight[1]
                reg_loss_dim = F.l1_loss(dimension_pred_valid, dimension_target_valid) * self.loss_weight[1]
                reg_loss_pos = F.l1_loss(position_pred_valid, position_target_valid) * self.loss_weight[1]

            else:
                reg_loss_ori = torch.tensor(0.0, device=self.device)
                reg_loss_dim = torch.tensor(0.0, device=self.device)
                reg_loss_pos = torch.tensor(0.0, device=self.device)


        self.stat = {
            'heatmap_loss': hm_loss,
            'position_loss': reg_loss_pos,
            'dimension_loss': reg_loss_dim,
            'rotation_loss': reg_loss_ori,
        }

        return self.stat


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
        )

    return loss_evaluator

def select_point_of_interest(batch, indices, feature_maps):
    '''
    Select POI (points of interest) on feature map using flat indices.
    Args:
        batch: batch size
        indices: flattened indices (directly as integers referring to positions in the feature map)
        feature_maps: regression feature map in [N, C, H, W] format

    Returns:
        Features at the points of interest, of shape [batch, num_pois, channel]
    '''
    # Reshape the feature maps from [N, C, H, W] to [N, H*W, C]
    feature_maps = feature_maps.permute(0, 2, 3, 1).contiguous()
    channel = feature_maps.shape[-1]
    feature_maps = feature_maps.view(batch, -1, channel)

    # Reshape indices to match the batch size
    indices = indices.view(batch, -1)

    # Expand indices to cover the channel dimension
    indices = indices.unsqueeze(-1).repeat(1, 1, channel)

    # Gather the feature values using the indices
    selected_features = feature_maps.gather(1, indices.long())

    return selected_features