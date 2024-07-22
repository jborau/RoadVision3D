from roadvision3d.src.models.detectors.MonoLSS import MonoLSS
from roadvision3d.src.models.detectors.MonoLSSv2 import MonoLSSv2
import numpy as np


def build_model(cfg, mean_size):
    if cfg['type'] == 'MonoLSS':
        mean_size = np.array(mean_size)
        return MonoLSS(backbone=cfg['backbone'], neck=cfg['neck'], loss = cfg['loss'], mean_size=mean_size)
    if cfg['type'] == 'MonoLSSv2':
        mean_size = np.array(mean_size)
        return MonoLSSv2(cfg)
    else:
        raise NotImplementedError("%s model is not supported" % cfg['type'])
