from torch import nn
import numpy as np

from .monolss_predictor import build_monolss_predictor
from roadvision3d.src.models.losses.loss_function import LSS_Loss


class MonoLSSHead(nn.Module):
    def __init__(self, cfg, in_channels, first_level):
        super(MonoLSSHead, self).__init__()

        self.predictor = build_monolss_predictor(cfg, in_channels, first_level)
        # self.loss_evaluator = build_monolss_loss(cfg)
        # self.post_processor = build_monolss_post_processor(cfg)
        # self.loss_evaluator = LSS_Loss(1)

        self.cls_mean_size = np.array(cfg['dataset']['cls_mean_size'])

    def forward(self, features, calib, targets=None, coord_ranges=None, epoch=1, mode='train', info=None):
        x = self.predictor(features, calib, coord_ranges, targets, mode=mode)
        if mode == 'train':
            criterion = LSS_Loss(epoch)
            total_loss, loss_terms = criterion(x, targets)
            return None, loss_terms
        else:
            result = post_processor(x, coord_ranges, info=info, cls_mean_size=self.cls_mean_size)
            return result, {}
        # if self.training:
        #     loss_heatmap, loss_regression = self.loss_evaluator(x, targets)

        #     return {}, dict(hm_loss=loss_heatmap,
        #                     reg_loss=loss_regression, )
        # if not self.training:
        #     result = self.post_processor(x, targets)

        #     return result, {}


def build_MonoLSS_head(cfg, in_channels, first_level):
    return MonoLSSHead(cfg, in_channels, first_level)


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