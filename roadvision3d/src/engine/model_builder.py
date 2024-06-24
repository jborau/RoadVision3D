from roadvision3d.src.models.detectors.MonoLSS import MonoLSS
import numpy as np


def build_model(cfg, mean_size):
    if cfg['type'] == 'MonoLSS':
        mean_size = np.array(mean_size)
        return MonoLSS(backbone=cfg['backbone'], neck=cfg['neck'], mean_size=mean_size)
    else:
        raise NotImplementedError("%s model is not supported" % cfg['type'])
