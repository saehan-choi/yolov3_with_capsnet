"""
Implementation of CapsyoloNet architecture
"""

import torch
import torch.nn as nn
from torchsummary import summary
from capsnet import *

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""


config = [
    # Original Image size
    # 416x416x3
    # out_channels, kernel_size, stride = module
    # padding=1 if kernel_size == 3 else 0,
    (32, 3, 1),
    # 416x416x32
    (64, 3, 2),
    # 208x208x64
    ["B", 1],
    (128, 3, 2),
    # 104x104x128
    ["B", 2],
    (256, 3, 2),
    # 52x52x256
    ["B", 8],
    (512, 3, 2),
    # 26x26x512
    ["B", 8],
    (1024, 3, 2),
    # 13x13x1024
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    # 13x13x512
    (1024, 3, 1),
    # 13x13x1024
    "S",  
    # out_channels = (num_classes + 5) * 3
    # [batch_size, 3, 13, 13, numclasses+5]
    (256, 1, 1),
    # 13x13x256
    "U",
    # 26x26x256
    (256, 1, 1),
    # 26x26x256
    (512, 3, 1),
    # 26x26x512
    "CapsNet26",
    # 26x26x512

    "S",
    # out_channels = (num_classes + 5) * 3
    # [batch_size, 3, 26, 26, numclasses+5]
    (128, 1, 1),
    # 26x26x128
    "U",
    # 52x52x128
    (128, 1, 1),
    # 52x52x128
    (256, 3, 1),
    # 52x52x256
    "CapsNet52",
    # 52x52x256

    "S",
    # out_channels = (num_classes + 5) * 3
    # [batch_size, 3, 52, 52, numclasses+5]
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        # print(f'x_pred_shape:{self.pred(x).shape}')
        # print(f'x_pred_reshape:{self.pred(x).reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]).shape}')
        # print(f'x_pred_permute:{self.pred(x).reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3]).permute(0, 1, 3, 4, 2).shape}')
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )


class CapsyoloNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []

        cnt=0
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x)) 
                continue

            x = layer(x)
            print(f'len_route_connections:{len(route_connections)}')
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

            elif isinstance(layer, CapsNet):
                # x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections[-1] = route_connections[-1]+x
                # route_connections.pop()

            print(layer)
            # print(x)
            print(x.size())
            print(cnt)
            cnt+=1

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3
                
                elif module == "CapsNet26":
                    layers.append(CapsNet(image_size=26,in_channel=512))
                    
                elif module == "CapsNet52":
                    layers.append(CapsNet(image_size=52,in_channel=256))

        return layers


if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = CapsyoloNet(num_classes=num_classes).cuda()
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE)).cuda()
    out = model(x)

    # print(model)

    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")
    
