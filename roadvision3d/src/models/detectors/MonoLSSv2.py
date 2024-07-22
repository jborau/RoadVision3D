import numpy as np
import torch.nn as nn

from roadvision3d.src.models.backbones.backbone import build_backbone, build_neck
from roadvision3d.src.models.heads.head import build_heads

class MonoLSSv2(nn.Module):
    def __init__(self, cfg):
        super(MonoLSSv2, self).__init__()
        self.backbone = build_backbone(cfg)

        #TODO: add this to cfg
        channels = self.backbone.channels
        downsample = 4
        self.first_level = int(np.log2(downsample))
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]

        self.neck = build_neck(cfg, channels[self.first_level:], scales_list=scales)

        self.heads = build_heads(cfg, self.backbone.channels, self.first_level)


    def forward(self, input, calib, targets=None, coord_ranges=None, epoch=1, K=50, mode='train', calib_tmp=None, info=None, cls_mean_size=None):
        """
        Args:
            images:
            targets:

        Returns:

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        # images = to_image_list(images)
        feat = self.backbone(input)
        feat = self.neck(feat[self.first_level:])
        result, detector_losses = self.heads(feat, calib, targets, coord_ranges, epoch, mode=mode, calibs_tmp=calib_tmp, info=info, cls_mean_size=cls_mean_size)

        # if self.training:
        #     losses = {}
        #     losses.update(detector_losses)
        if mode=='train':
            losses = {}
            losses.update(detector_losses)
            return losses

        return result