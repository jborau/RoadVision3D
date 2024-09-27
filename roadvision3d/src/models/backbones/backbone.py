from .dla import dla34
from .dlaup import DLAUp, DLAUpv2

def build_backbone(cfg):
    if cfg['backbone'] == 'dla34':
        return dla34(pretrained=True, return_levels=True)
    else:
        raise NotImplementedError("%s backbone is not supported" % cfg['backbone'])
    
def build_neck(cfg, channels, scales_list):
    # if cfg['neck'] == 'DLAUp':
    #     return DLAUp(in_channels_list=cfg['in_channels_list'], scales_list=cfg['scales_list'])
    # elif cfg['neck'] == 'DLAUpv2':
    #     return DLAUpv2(in_channels_list=cfg['in_channels_list'], scales_list=cfg['scales_list'])
    # else:
    #     raise NotImplementedError("%s neck is not supported" % cfg['neck'])
    if cfg['neck'] == 'DLAUp':
        return DLAUp(in_channels_list=channels, scales_list=scales_list)
    elif cfg['neck'] == 'DLAUpv2':
        return DLAUpv2(in_channels_list=cfg['in_channels_list'], scales_list=cfg['scales_list'])
    else:
        raise NotImplementedError("%s neck is not supported" % cfg['neck'])