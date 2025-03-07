from .monolss_head import build_MonoLSS_head
from .smoke_head import build_smoke_head
from .GUPNet_head import build_GUPNet_head


def build_heads(cfg, in_channels, first_level, device='cuda:0'):
    if cfg['model']['head'] == 'MonoLSSHead':
        return build_MonoLSS_head(cfg, in_channels, first_level)
    if cfg['model']['head'] == 'SmokeHead':
        return build_smoke_head(cfg, in_channels, first_level, device)
    if cfg['model']['head'] == 'GUPNetHead':
        return build_GUPNet_head(cfg, in_channels, first_level)