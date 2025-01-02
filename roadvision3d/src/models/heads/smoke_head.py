
from torch import nn
import numpy as np

from .smoke_predictor import build_smoke_predictor
from roadvision3d.src.models.losses.smoke_loss import build_smoke_loss
from roadvision3d.src.models.losses.smoke_postprocessor import build_smoke_postprocessor



class SmokeHead(nn.Module):
    def __init__(self, cfg, in_channels, first_level, device):
        super(SmokeHead, self).__init__()

        self.predictor = build_smoke_predictor(cfg, in_channels)
        self.loss_evaluator = build_smoke_loss(cfg, device)
        self.post_processor = build_smoke_postprocessor(cfg, device)
        self.cls_mean_size = np.array(cfg['dataset']['cls_mean_size'])


    def forward(self, features, calib, targets=None, coord_ranges=None, epoch=1, mode='train', info=None, cls_mean_size=None):
        # x = self.predictor(features, calib, coord_ranges, targets, mode=mode)
        x = self.predictor(features)
        if mode=='train':
            # loss_heatmap, loss_regression = self.loss_evaluator(x, targets, calib)
            total_loss, loss_terms = self.loss_evaluator(x, targets, calib, info)
            return {}, loss_terms
        else:
            result = self.post_processor(x, calib, info=info, cls_mean_size=self.cls_mean_size)
            # result = x
            return result, {}


def build_smoke_head(cfg, in_channels, first_level, device):
    return SmokeHead(cfg, in_channels, first_level, device)









