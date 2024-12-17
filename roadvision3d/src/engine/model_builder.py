from roadvision3d.src.models.detectors.MonoLSS import MonoLSS
from roadvision3d.src.models.detectors.keypoint_detector import KeypointDetector

import numpy as np


def build_model(cfg, device):
    if cfg['model']['type'] == 'MonoLSS':
        # mean_size = np.array(cfg['dataset']['cls_mean_size'])
        return MonoLSS(cfg, device)
    if cfg['model']['type'] == 'SMOKE':
        return KeypointDetector(cfg, device)
    else:
        raise NotImplementedError("%s model is not supported" % cfg['model']['type'])
