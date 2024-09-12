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
                 max_objs):
        self.smoke_coder = smoke_coder
        self.cls_loss = cls_loss
        self.reg_loss = reg_loss
        self.loss_weight = loss_weight
        self.max_objs = max_objs

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

        return heatmap, dict(depth=depth, size_2d=size_2d, offset_2d=offset_2d, 
                            indices=indices, size_3d=size_3d, offset_3d=offset_3d,
                            heading_bin=heading_bin, heading_res=heading_res,
                            cls_ids=cls_ids, mask_2d=mask_2d, vis_depth=vis_depth)


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
        pred_dimensions = self.smoke_coder.decode_dimension(
            targets_variables["cls_ids"],
            pred_dimensions_offsets,
        )
        # we need to change center location to bottom location
        pred_locations[:, 1] += pred_dimensions[:, 1] / 2

        pred_rotys = self.smoke_coder.decode_orientation(
            pred_orientation,
            targets_variables["locations"],
            targets_variables["flip_mask"]
        )

        if self.reg_loss == "DisL1":
            pred_box3d_rotys = self.smoke_coder.encode_box3d(
                pred_rotys,
                targets_variables["dimensions"],
                targets_variables["locations"]
            )
            pred_box3d_dims = self.smoke_coder.encode_box3d(
                targets_variables["rotys"],
                pred_dimensions,
                targets_variables["locations"]
            )
            pred_box3d_locs = self.smoke_coder.encode_box3d(
                targets_variables["rotys"],
                targets_variables["dimensions"],
                pred_locations
            )

            return dict(ori=pred_box3d_rotys,
                        dim=pred_box3d_dims,
                        loc=pred_box3d_locs, )

        elif self.reg_loss == "L1":
            pred_box_3d = self.smoke_coder.encode_box3d(
                pred_rotys,
                pred_dimensions,
                pred_locations
            )
            return pred_box_3d

    def __call__(self, predictions, targets, calibs):
        pred_heatmap, pred_regression = predictions[0], predictions[1]

        # targets_heatmap, targets_regression, targets_variables \
        #     = self.prepare_targets(targets)
        targets_heatmap, targets_variables = self.prepare_targets(targets)

        predict_boxes3d = self.prepare_predictions(targets_variables, pred_regression, calibs)

        hm_loss = self.cls_loss(pred_heatmap, targets_heatmap) * self.loss_weight[0]

        targets_regression = targets_regression.view(
            -1, targets_regression.shape[2], targets_regression.shape[3]
        )

        reg_mask = targets_variables["reg_mask"].flatten()
        reg_mask = reg_mask.view(-1, 1, 1)
        reg_mask = reg_mask.expand_as(targets_regression)

        if self.reg_loss == "DisL1":
            reg_loss_ori = F.l1_loss(
                predict_boxes3d["ori"] * reg_mask,
                targets_regression * reg_mask,
                reduction="sum") / (self.loss_weight[1] * self.max_objs)

            reg_loss_dim = F.l1_loss(
                predict_boxes3d["dim"] * reg_mask,
                targets_regression * reg_mask,
                reduction="sum") / (self.loss_weight[1] * self.max_objs)

            reg_loss_loc = F.l1_loss(
                predict_boxes3d["loc"] * reg_mask,
                targets_regression * reg_mask,
                reduction="sum") / (self.loss_weight[1] * self.max_objs)

            return hm_loss, reg_loss_ori + reg_loss_dim + reg_loss_loc


def build_smoke_loss(cfg):
    smoke_coder_depth_reference = (28.01, 16.32)
    smoke_coder_dimension_reference = ((3.88, 1.63, 1.53),
                                        (1.78, 1.70, 0.58),
                                        (0.88, 1.73, 0.67))
    smoke_device = 'cuda:0'
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
        loss_weight = (1., 10.),
        # max_objs=cfg.DATASETS.MAX_OBJECTS,
        max_objs = 30
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