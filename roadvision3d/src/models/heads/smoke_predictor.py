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
        # classes = len(cfg.DATASETS.DETECT_CLASSES)
        # _C.DATASETS.DETECT_CLASSES = ("Car",)

        regression = 8
        # regression = cfg.MODEL.SMOKE_HEAD.REGRESSION_HEADS
        # _C.MODEL.SMOKE_HEAD.REGRESSION_HEADS = 8

        regression_channels = (1, 2, 3, 2)
        # regression_channels = cfg.MODEL.SMOKE_HEAD.REGRESSION_CHANNEL
        # _C.MODEL.SMOKE_HEAD.REGRESSION_CHANNEL = (1, 2, 3, 2)

        head_conv = 256
        # head_conv = cfg.MODEL.SMOKE_HEAD.NUM_CHANNEL
        # _C.MODEL.SMOKE_HEAD.NUM_CHANNEL = 256

        norm_func = _HEAD_NORM_SPECS['BN']
        # norm_func = _HEAD_NORM_SPECS[cfg.MODEL.SMOKE_HEAD.USE_NORMALIZATION]
        # _C.MODEL.SMOKE_HEAD.USE_NORMALIZATION = "GN" 

        assert sum(regression_channels) == regression, \
            "the sum of {} must be equal to regression channel of {}".format(
                cfg.MODEL.SMOKE_HEAD.REGRESSION_CHANNEL, cfg.MODEL.SMOKE_HEAD.REGRESSION_HEADS
            )

        self.dim_channel = get_channel_spec(regression_channels, name="dim")
        self.ori_channel = get_channel_spec(regression_channels, name="ori")


        #TODO: check this
        in_channels = in_channels[2] 

        self.class_head = nn.Sequential(
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

        # TODO: what is datafill here
        self.class_head[-1].bias.data.fill_(-2.19)

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

        # 2D BBOX MODULE
        self.size_2d = nn.Sequential(nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(head_conv, 2, kernel_size=1, stride=1, padding=0, bias=True))
        _fill_fc_weights(self.size_2d)

    def forward(self, features):
        head_class = self.class_head(features)
        head_regression = self.regression_head(features)

        # head_class = sigmoid_hm(head_class)
        heatmap_head = torch.clamp(head_class.sigmoid_(), min=1e-4, max=1 - 1e-4)
        # (N, C, H, W)
        offset_dims = head_regression[:, self.dim_channel, ...].clone()
        head_regression[:, self.dim_channel, ...] = torch.sigmoid(offset_dims) - 0.5

        vector_ori = head_regression[:, self.ori_channel, ...].clone()
        head_regression[:, self.ori_channel, ...] = F.normalize(vector_ori)

        # 2d head
        size_2d_reg = self.size_2d(features)

        return [heatmap_head, head_regression, size_2d_reg] 


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
