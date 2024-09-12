
from torch import nn
import numpy as np

from .smoke_predictor import build_smoke_predictor
from roadvision3d.src.models.losses.smoke_loss import build_smoke_loss



class SmokeHead(nn.Module):
    def __init__(self, cfg, in_channels, first_level):
        super(SmokeHead, self).__init__()

        self.predictor = build_smoke_predictor(cfg, in_channels)
        self.loss_evaluator = build_smoke_loss(cfg)


    def forward(self, features, calib, targets=None, coord_ranges=None, epoch=1, mode='train', calibs_tmp=None, info=None, cls_mean_size=None):
        # x = self.predictor(features, calib, coord_ranges, targets, mode=mode)
        x = self.predictor(features)
        
        if self.training:
            loss_heatmap, loss_regression = self.loss_evaluator(x, targets, calib)

            return {}, dict(hm_loss=loss_heatmap,
                            reg_loss=loss_regression, )
        else:
            result = post_processor(x, calibs_tmp, info=info, cls_mean_size=cls_mean_size)
            return result, {}


def build_smoke_head(cfg, in_channels, first_level):
    return SmokeHead(cfg, in_channels, first_level)


from roadvision3d.src.engine.decode_helper import extract_dets_from_outputs
from roadvision3d.src.engine.decode_helper import decode_detections
def post_processor(outputs, calib_tmp, info, cls_mean_size):
    dets = extract_dets_from_outputs(outputs, K=50)
    dets = dets.detach().cpu().numpy()
                
    # get corresponding calibs & transform tensor to numpy
    calibs = calib_tmp
    info = info
    cls_mean_size = cls_mean_size
    dets = decode_detections(dets = dets,
                            info = info,
                            calibs = info['calibs'],
                            cls_mean_size=cls_mean_size,
                            threshold = 0.2)
    return dets
