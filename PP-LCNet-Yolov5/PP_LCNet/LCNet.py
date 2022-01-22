import os
import torch
import torch.nn as nn
import thop

# try:
#     import softpool_cuda
#     from SoftPool import soft_pool2d, SoftPool2d
# except ImportError:
#     print('Please install SoftPool first: https://github.com/alexandrosstergiou/SoftPool')
#     exit(0)

NET_CONFIG = {
    "blocks2":
    # k, in_c, out_c, s, use_se
    [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [[3, 128, 256, 2, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]]
}


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return x * self.relu6(x+3) / 6


class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return (self.relu6(x+3)) / 6


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            HardSigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DepthwiseSeparable(nn.Module):
    def __init__(self, inp, oup, dw_size, stride, use_se=False):
        super(DepthwiseSeparable, self).__init__()
        self.use_se = use_se
        self.stride = stride
        self.inp = inp
        self.oup = oup
        self.dw_size = dw_size
        self.dw_sp = nn.Sequential(
            nn.Conv2d(self.inp, self.inp, kernel_size=self.dw_size, stride=self.stride,
                      padding=autopad(self.dw_size, None), groups=self.inp, bias=False),
            nn.BatchNorm2d(self.inp),
            HardSwish(),

            nn.Conv2d(self.inp, self.oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.oup),
            HardSwish(),
        )
        self.se = SELayer(self.oup)

    def forward(self, x):
        x = self.dw_sp(x)
        if self.use_se:
            x = self.se(x)
        return x


class PP_LCNet(nn.Module):
    def __init__(self, scale=1.0, class_num=10, class_expand=1280, dropout_prob=0.2):
        super(PP_LCNet, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(3, out_channels=make_divisible(16 * self.scale),
                               kernel_size=3, stride=2, padding=1, bias=False)
        # k, in_c, out_c, s, use_se   inp, oup, dw_size, stride, use_se=False
        self.blocks2 = nn.Sequential(*[
            DepthwiseSeparable(inp=make_divisible(in_c * self.scale),
                               oup=make_divisible(out_c * self.scale),
                               dw_size=k, stride=s, use_se=use_se)
            for i, (k, in_c, out_c, s, use_se) in enumerate(NET_CONFIG["blocks2"])
        ])

        self.blocks3 = nn.Sequential(*[
            DepthwiseSeparable(inp=make_divisible(in_c * self.scale),
                               oup=make_divisible(out_c * self.scale),
                               dw_size=k, stride=s, use_se=use_se)
            for i, (k, in_c, out_c, s, use_se) in enumerate(NET_CONFIG["blocks3"])
        ])

        self.blocks4 = nn.Sequential(*[
            DepthwiseSeparable(inp=make_divisible(in_c * self.scale),
                               oup=make_divisible(out_c * self.scale),
                               dw_size=k, stride=s, use_se=use_se)
            for i, (k, in_c, out_c, s, use_se) in enumerate(NET_CONFIG["blocks4"])
        ])
        # k, in_c, out_c, s, use_se  inp, oup, dw_size, stride, use_se=False
        self.blocks5 = nn.Sequential(*[
            DepthwiseSeparable(inp=make_divisible(in_c * self.scale),
                               oup=make_divisible(out_c * self.scale),
                               dw_size=k, stride=s, use_se=use_se)
            for i, (k, in_c, out_c, s, use_se) in enumerate(NET_CONFIG["blocks5"])
        ])

        self.blocks6 = nn.Sequential(*[
            DepthwiseSeparable(inp=make_divisible(in_c * self.scale),
                               oup=make_divisible(out_c * self.scale),
                               dw_size=k, stride=s, use_se=use_se)
            for i, (k, in_c, out_c, s, use_se) in enumerate(NET_CONFIG["blocks6"])
        ])

        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.last_conv = nn.Conv2d(in_channels=make_divisible(NET_CONFIG["blocks6"][-1][2] * scale),
                                   out_channels=class_expand,
                                   kernel_size=1, stride=1, padding=0, bias=False)

        self.hardswish = HardSwish()
        self.dropout = nn.Dropout(p=dropout_prob)

        self.fc = nn.Linear(class_expand, class_num)

    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)
        x = self.blocks2(x)
        print(x.shape)
        x = self.blocks3(x)
        print(x.shape)
        x = self.blocks4(x)
        print(x.shape)
        x = self.blocks5(x)
        print(x.shape)
        x = self.blocks6(x)
        print(x.shape)

        x = self.GAP(x)
        x = self.last_conv(x)
        x = self.hardswish(x)
        x = self.dropout(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc(x)
        return x


def PPLCNET_x0_25(**kwargs):
    model = PP_LCNet(scale=0.25, **kwargs)
    return model


def PPLCNET_x0_35(**kwargs):
    model = PP_LCNet(scale=0.35, **kwargs)
    return model


def PPLCNET_x0_5(**kwargs):
    model = PP_LCNet(scale=0.5, **kwargs)
    return model


def PPLCNET_x0_75(**kwargs):
    model = PP_LCNet(scale=0.75, **kwargs)
    return model


def PPLCNET_x1_0(**kwargs):
    model = PP_LCNet(scale=1.0, **kwargs)
    return model


def PPLCNET_x1_5(**kwargs):
    model = PP_LCNet(scale=1.5, **kwargs)
    return model


def PPLCNET_x2_0(**kwargs):
    model = PP_LCNet(scale=2.0, **kwargs)
    return model

def PPLCNET_x2_5(**kwargs):
    model = PP_LCNet(scale=2.5, **kwargs)
    return model




if __name__ == '__main__':
    # input = torch.randn(1, 3, 640, 640)
    # model = PPLCNET_x2_5()
    # flops, params = thop.profile(model, inputs=(input,))
    # print('flops:', flops / 1000000000)
    # print('params:', params / 1000000)

    model = PPLCNET_x1_0()
    # model_1 = PW_Conv(3, 16)
    input = torch.randn(2, 3, 256, 256)
    print(input.shape)
    output = model(input)
    print(output.shape)  # [1, num_class]


