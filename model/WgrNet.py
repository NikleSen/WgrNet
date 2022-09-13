import torch.nn.functional as fun
import torch.nn as nn
import torch as t
import torchsummary as ts

class NormConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        return fun.conv2d(x, w, self.bias, self.stride, self.padding,
                          self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False, dilations=1):
    padding = 1
    if dilations == 2:
        padding = 2
    return NormConv2d(cin, cout, kernel_size=3, stride=stride, padding=padding, bias=bias, groups=groups,
                      dilation=dilations)


def conv1x1(cin, cout, stride=1, groups=1, bias=False):
    return NormConv2d(cin, cout, kernel_size=1, stride=stride, groups=groups, padding=0, bias=bias)


def conv1x3(cin, cout, stride=1, groups=1, bias=False, dilations=1):
    return NormConv2d(cin, cout, kernel_size=(1, 3), stride=stride, padding=(0, 1), bias=bias, groups=groups,
                      dilation=dilations)


def conv3x1(cin, cout, stride=1, groups=1, bias=False, dilations=1):
    return NormConv2d(cin, cout, kernel_size=(3, 1), stride=stride, padding=(1, 0), bias=bias, groups=groups,
                      dilation=dilations)


class Rootblock(nn.Module):
    def __init__(self,
                 cin=3,
                 cout=64,
                 stride=1,
                 ):
        super(Rootblock, self).__init__()
        self.cout = cout
        self.act_fun = nn.ReLU(inplace=True)
        self.normer = nn.BatchNorm2d(cout)
        self.conv1 = NormConv2d(cin, self.cout, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        return self.act_fun(self.normer(self.conv1(x)))


class Basicblock(nn.Module):

    def __init__(self,
                 cin: int = 64,
                 cout: int = 128,
                 light: int = 0,
                 kernel: int = 3,
                 stride: int = 1,
                 expansion: float = -1):
        super(Basicblock, self).__init__()

        self.short_cut = False
        if stride != 1 or cin != cout:
            self.short_cut = True
            if kernel == 1:
                self.conv0 = conv1x1(cin, cout, stride)
            else:
                self.conv0 = conv3x3(cin, cout, stride)
            self.norm0 = nn.BatchNorm2d(cout)

        mid = cout + int((cout - cin) * expansion)

        self.conv1x1 = conv1x1(cout, cout)
        self.conv3x3 = conv3x3(cout, cout, groups=cout)

        self.act_fun = nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm2d(mid)
        self.norm2 = nn.BatchNorm2d(cout)
        self.norm3 = nn.BatchNorm2d(cout)
        if light == 1:
            self.conv3x1_1 = conv3x1(cin, mid, stride=stride, groups=4)
            self.conv1x3_1 = conv1x3(cin, mid, stride=stride, groups=4)
            self.conv3x1_2 = conv3x1(mid, cout, stride=1, groups=8)
            self.conv1x3_2 = conv1x3(mid, cout, stride=1, groups=8)

        else:
            self.conv3x1_1 = conv3x1(cin, mid, stride=stride)
            self.conv1x3_1 = conv1x3(cin, mid, stride=stride)
            self.conv3x1_2 = conv3x1(mid, cout, stride=1)
            self.conv1x3_2 = conv1x3(mid, cout, stride=1)

    def forward(self, x):
        identity = x
        if self.short_cut:
            identity = self.norm0(self.conv0(identity))
        x = self.act_fun(self.norm1(self.conv1x3_1(x) + self.conv3x1_1(x)))
        x = self.act_fun(self.norm2(self.conv1x3_2(x) + self.conv3x1_2(x)))
        x = identity + self.norm3(self.conv1x1(x) + self.conv3x3(x))

        return self.act_fun(x)


def _make_stage(in_channels, out_channels, depth, light=0, kernel=3, stride=1, expansion=-1):
    blocks = []
    for i in range(0, depth - 1):
        blocks.append(Basicblock(in_channels, in_channels, light, kernel, 1, expansion))
    blocks.append(Basicblock(in_channels, out_channels, light, kernel, stride, expansion))

    return nn.Sequential(*blocks)


class WGRNetGroup(nn.Module):
    def __init__(self,
                 inplanes,
                 num_classes=1000,
                 depth_list=None,
                 width_list=None,
                 light=0,
                 kernel=3,
                 stride=1,
                 expansion=-1):
        super(WGRNetGroup, self).__init__()
        self.inplanes = inplanes
        self.num_classes = num_classes
        if depth_list is None:
            depth_list = [1, 2, 3, 1]
        if width_list is None:
            width_list = [64,128, 256, 512, 1024]

        self.root = Rootblock(self.inplanes, width_list[0], stride=1)
        self.stage1 = _make_stage(width_list[0], width_list[1], depth_list[0], light, kernel, stride, expansion)
        self.stage2 = _make_stage(width_list[1], width_list[2], depth_list[1], light, kernel, 2, expansion)
        self.stage3 = _make_stage(width_list[2], width_list[3], depth_list[2], light, kernel, 2, expansion)
        self.stage4 = _make_stage(width_list[3], width_list[4], depth_list[3], light, kernel, 2, expansion)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Dropout(0.5),
            nn.Linear(width_list[4], self.num_classes),
        )

    def forward(self, x):
        x = self.root(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        return x


def wgrnet(in_channel=3, outchannel=1000, depth=0, width=0, light=0, kernel=3, stride=1, expansion=None):
    depth_list = [[1, 1, 2, 1], [1, 2, 3, 1], [1, 3, 4, 1], [1, 3, 6, 1], [1, 3, 9, 1]]
    width_list = [[32, 64, 128, 256, 512], [64, 96, 192, 384, 512], [64, 128, 256, 384, 768], [64, 128, 256, 512, 1024],[64, 128, 256, 512, 1536],[64, 128, 256, 512, 1536]]

    return WGRNetGroup(in_channel, outchannel, depth_list[depth], width_list[width], light=light, kernel=kernel,
                  stride=stride, expansion=expansion)


if __name__ == '__main__':
    depth_id = 0
    width_id = 0
    light = 0
    kernel = 3
    expansion = 1  # mid = cout + int((cout - cin) * expansion)
    stride = 1
    x = t.randn(100, 3, 32, 32)
    B, C, H, W = x.shape

    model = wgrnet(3, 100, depth=depth_id, width=width_id, light=light, kernel=kernel, stride=stride,
                   expansion=expansion)
    x = model(x)
    model.to('cuda')
    ts.summary(model, (C, H, W), B)

