
from torch import nn
import numpy as np

from .smoke_predictor import build_smoke_predictor
from roadvision3d.src.models.losses.smoke_loss import build_smoke_loss
from roadvision3d.src.models.losses.smoke_postprocessor import build_smoke_postprocessor



class SmokeHead(nn.Module):
    def __init__(self, cfg, in_channels, first_level):
        super(SmokeHead, self).__init__()

        self.predictor = build_smoke_predictor(cfg, in_channels)
        self.loss_evaluator = build_smoke_loss(cfg)
        self.post_processor = build_smoke_postprocessor(cfg)


    def forward(self, features, calib, targets=None, coord_ranges=None, epoch=1, mode='train', calibs_tmp=None, info=None, cls_mean_size=None):
        # x = self.predictor(features, calib, coord_ranges, targets, mode=mode)
        x = self.predictor(features)
        
        if self.training:
            # loss_heatmap, loss_regression = self.loss_evaluator(x, targets, calib)
            losses_dict = self.loss_evaluator(x, targets, calib)

            # return {}, dict(hm_loss=loss_heatmap,
            #                 reg_loss=loss_regression, )
            return {}, losses_dict
        else:
            result = self.post_processor(x, calib, info=info, cls_mean_size=cls_mean_size)
            return result, {}


def build_smoke_head(cfg, in_channels, first_level):
    return SmokeHead(cfg, in_channels, first_level)









