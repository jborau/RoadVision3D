from roadvision3d.src.models.detectors.MonoLSS import MonoLSS
from roadvision3d.src.models.detectors.keypoint_detector import KeypointDetector

import numpy as np


def build_model(cfg):
    if cfg['model']['type'] == 'MonoLSS':
        # mean_size = np.array(cfg['dataset']['cls_mean_size'])
        return MonoLSS(cfg)
    if cfg['model']['type'] == 'SMOKE':
        return KeypointDetector(cfg)
    else:
        raise NotImplementedError("%s model is not supported" % cfg['model']['type'])
