from .monolss_head import build_MonoLSS_head


def build_heads(cfg, in_channels, first_level):
    if cfg['head'] == 'MonoLSSHead':
        return build_MonoLSS_head(cfg, in_channels, first_level)