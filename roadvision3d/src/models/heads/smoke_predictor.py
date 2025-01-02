import torch
from torch import nn
from torch.nn import functional as F


_HEAD_NORM_SPECS = {    "BN": nn.BatchNorm2d,
    # "GN": group_norm,
}



class SMOKEPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(SMOKEPredictor, self).__init__()

        classes = cfg['dataset']['cls_num']
        regression = 8
        regression_channels = (1, 2, 3, 2)
        head_conv = 256
        norm_func = _HEAD_NORM_SPECS['BN']

        assert sum(regression_channels) == regression, \
            "the sum of {} must be equal to regression channel of {}".format(
                cfg.MODEL.SMOKE_HEAD.REGRESSION_CHANNEL, cfg.MODEL.SMOKE_HEAD.REGRESSION_HEADS
            )

        self.dim_channel = get_channel_spec(regression_channels, name="dim")
        self.ori_channel = get_channel_spec(regression_channels, name="ori")


        #TODO: check this
        in_channels = in_channels[2] 

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels,
                      head_conv,
                      kernel_size=3,
                      padding=1,
                      bias=True),

            norm_func(head_conv),

            nn.ReLU(inplace=True),

            nn.Conv2d(head_conv,
                      classes,
                      kernel_size=1,
                      padding=1 // 2,
                      bias=True)
        )

        self.heatmap_head[-1].bias.data.fill_(-2.19)

        # 2D BBOX MODULE
        self.size_2d = nn.Sequential(nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True))
        self.offset_2d = nn.Sequential(nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True))
        _fill_fc_weights(self.size_2d)
        _fill_fc_weights(self.offset_2d)

        # 3D REGRESSION MODULE
        self.regression_head = nn.Sequential(
            nn.Conv2d(in_channels,
                      head_conv,
                      kernel_size=3,
                      padding=1,
                      bias=True),

            norm_func(head_conv),

            nn.ReLU(inplace=True),

            nn.Conv2d(head_conv,
                      regression,
                      kernel_size=1,
                      padding=1 // 2,
                      bias=True)
        )
        _fill_fc_weights(self.regression_head)

    def forward(self, features):
        ret = {}

        heatmap_reg = self.heatmap_head(features)

        size_2d_reg = self.size_2d(features)
        offset_2d_reg = self.offset_2d(features)

        regression_reg = self.regression_head(features)

        ret['heatmap'] = heatmap_reg
        ret['size_2d'] = size_2d_reg
        ret['offset_2d'] = offset_2d_reg


        # Depth: Directly extract (no sigmoid needed)
        ret['depth'] = regression_reg[:, 0:1, :, :]

        # Offset 3D: Directly extract (no sigmoid needed)
        ret['offset_3d'] = regression_reg[:, 1:3, :, :]

        # Dimension offset: Apply sigmoid and shift by -0.5
        dimension_offset = regression_reg[:, self.dim_channel, :, :].clone()
        ret['size_3d_offset'] = torch.sigmoid(dimension_offset) - 0.5

        # Orientation: Normalize
        vector_ori = regression_reg[:, self.ori_channel, :, :].clone()
        ret['ori'] = F.normalize(vector_ori)

        return ret


def get_channel_spec(reg_channels, name):
    if name == "dim":
        s = sum(reg_channels[:2])
        e = sum(reg_channels[:3])
    elif name == "ori":
        s = sum(reg_channels[:3])
        e = sum(reg_channels)

    return slice(s, e, 1)

def _fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def sigmoid_hm(hm_features):
    x = hm_features.sigmoid_()
    x = x.clamp(min=1e-4, max=1 - 1e-4)

    return x

def group_norm(out_channels):
    num_groups = cfg.MODEL.GROUP_NORM.NUM_GROUPS
    # _C.MODEL.GROUP_NORM.NUM_GROUPS = 32
    if out_channels % 32 == 0:
        return nn.GroupNorm(num_groups, out_channels)
    else:
        return nn.GroupNorm(num_groups // 2, out_channels)


def build_smoke_predictor(cfg, in_channels):
    return SMOKEPredictor(cfg, in_channels)
