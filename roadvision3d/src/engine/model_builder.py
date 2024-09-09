from roadvision3d.src.models.detectors.MonoLSS import MonoLSS
import numpy as np


def build_model(cfg):
    if cfg['model']['type'] == 'MonoLSS':
        # mean_size = np.array(cfg['dataset']['cls_mean_size'])
        return MonoLSS(cfg)
    else:
        raise NotImplementedError("%s model is not supported" % cfg['model']['type'])
