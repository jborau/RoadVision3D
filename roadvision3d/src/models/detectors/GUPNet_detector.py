import numpy as np
import torch.nn as nn

from roadvision3d.src.models.backbones.backbone import build_backbone, build_neck
from roadvision3d.src.models.heads.head import build_heads

class GUPNetDetector(nn.Module):
    def __init__(self, cfg, device):
        super(GUPNetDetector, self).__init__()
        self.device = device
        self.backbone = build_backbone(cfg['model'])

        channels = self.backbone.channels
        downsample = cfg['model']['downsample']
        self.first_level = int(np.log2(downsample))
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]

        self.neck = build_neck(cfg['model'], channels[self.first_level:], scales_list=scales)

        self.heads = build_heads(cfg, self.backbone.channels, self.first_level, device)


    def forward(self, input, calib, targets=None, coord_ranges=None, epoch=1, mode='train', info=None):
        """
        Args:
            images:
            targets:

        Returns:

        """
        if mode=='train' and targets is None:
            raise ValueError("In training mode, targets should be passed")
        # images = to_image_list(images)
        feat = self.backbone(input)
        feat = self.neck(feat[self.first_level:])
        result, detector_losses = self.heads(feat, calib, targets, coord_ranges, epoch, mode=mode, info=info)

        # if self.training:
        #     losses = {}
        #     losses.update(detector_losses)
        if mode=='train':
            losses = {}
            losses.update(detector_losses)
            return losses

        return result