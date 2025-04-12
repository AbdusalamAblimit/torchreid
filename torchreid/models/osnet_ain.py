from __future__ import division, absolute_import
import warnings
import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    'osnet_ain_x1_0', 'osnet_ain_x0_75', 'osnet_ain_x0_5', 'osnet_ain_x0_25'
]

pretrained_urls = {
    'osnet_ain_x1_0':
    'https://drive.google.com/uc?id=1-CaioD9NaqbHK_kzSMW8VE4_3KcsRjEo',
    'osnet_ain_x0_75':
    'https://drive.google.com/uc?id=1apy0hpsMypqstfencdH-jKIUEFOW4xoM',
    'osnet_ain_x0_5':
    'https://drive.google.com/uc?id=1KusKvEYyKGDTUBVRxRiz55G31wkihB6l',
    'osnet_ain_x0_25':
    'https://drive.google.com/uc?id=1SxQt2AvmEcgWNhaRb2xC4rP6ZwVDP0Wt'
}


##########
# Basic layers
##########
class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        IN=False
    ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups
        )
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1, bn=True):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False
        )
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class Conv3x3(nn.Module):
    """3x3 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution.

    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            groups=out_channels
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        return self.relu(x)


class LightConvStream(nn.Module):
    """Lightweight convolution stream."""

    def __init__(self, in_channels, out_channels, depth):
        super(LightConvStream, self).__init__()
        assert depth >= 1, 'depth must be equal to or larger than 1, but got {}'.format(
            depth
        )
        layers = []
        layers += [LightConv3x3(in_channels, out_channels)]
        for i in range(depth - 1):
            layers += [LightConv3x3(out_channels, out_channels)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


##########
# Building blocks for omni-scale feature learning
##########
class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(
        self,
        in_channels,
        num_gates=None,
        return_gates=False,
        gate_activation='sigmoid',
        reduction=16,
        layer_norm=False
    ):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels,
            in_channels // reduction,
            kernel_size=1,
            bias=True,
            padding=0
        )
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(
            in_channels // reduction,
            num_gates,
            kernel_size=1,
            bias=True,
            padding=0
        )
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU()
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError(
                "Unknown gate activation: {}".format(gate_activation)
            )

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(self, in_channels, out_channels, reduction=4, T=4, **kwargs):
        super(OSBlock, self).__init__()
        assert T >= 1
        assert out_channels >= reduction and out_channels % reduction == 0
        mid_channels = out_channels // reduction

        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList()
        for t in range(1, T + 1):
            self.conv2 += [LightConvStream(mid_channels, mid_channels, t)]
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2_t = conv2_t(x1)
            x2 = x2 + self.gate(x2_t)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return F.relu(out)


class OSBlockINin(nn.Module):
    """Omni-scale feature learning block with instance normalization."""

    def __init__(self, in_channels, out_channels, reduction=4, T=4, **kwargs):
        super(OSBlockINin, self).__init__()
        assert T >= 1
        assert out_channels >= reduction and out_channels % reduction == 0
        mid_channels = out_channels // reduction

        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList()
        for t in range(1, T + 1):
            self.conv2 += [LightConvStream(mid_channels, mid_channels, t)]
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels, bn=False)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2_t = conv2_t(x1)
            x2 = x2 + self.gate(x2_t)
        x3 = self.conv3(x2)
        x3 = self.IN(x3) # IN inside residual
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return F.relu(out)


##########
# Network architecture
##########
class OSNet(nn.Module):
    """Omni-Scale Network.
    
    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. TPAMI, 2021.
    """

    def __init__(
        self,
        num_classes,
        blocks,
        layers,
        channels,
        feature_dim=512,
        loss='softmax',
        conv1_IN=False,
        **kwargs
    ):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.loss = loss
        self.feature_dim = feature_dim

        # convolutional backbone
        self.conv1 = ConvLayer(
            3, channels[0], 7, stride=2, padding=3, IN=conv1_IN
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(
            blocks[0], layers[0], channels[0], channels[1]
        )
        self.pool2 = nn.Sequential(
            Conv1x1(channels[1], channels[1]), nn.AvgPool2d(2, stride=2)
        )
        self.conv3 = self._make_layer(
            blocks[1], layers[1], channels[1], channels[2]
        )
        self.pool3 = nn.Sequential(
            Conv1x1(channels[2], channels[2]), nn.AvgPool2d(2, stride=2)
        )
        self.conv4 = self._make_layer(
            blocks[2], layers[2], channels[2], channels[3]
        )
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # fully connected layer
        self.fc = self._construct_fc_layer(
            self.feature_dim, channels[3], dropout_p=None
        )
        # identity classification layer
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._init_params()

    def _make_layer(self, blocks, layer, in_channels, out_channels):
        layers = []
        layers += [blocks[0](in_channels, out_channels)]
        for i in range(1, len(blocks)):
            layers += [blocks[i](out_channels, out_channels)]
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None

        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x, return_featuremaps=False):
        x = self.featuremaps(x)
        if return_featuremaps:
            return x
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


def init_pretrained_weights(model, key=''):
    """Initializes model with pretrained weights.
    
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import os
    import errno
    import gdown
    from collections import OrderedDict

    def _get_torch_home():
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(
                    os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
                )
            )
        )
        return torch_home

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise
    filename = key + '_imagenet.pth'
    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        gdown.download(pretrained_urls[key], cached_file, quiet=False)

    state_dict = torch.load(cached_file)
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights from "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(cached_file)
        )
    else:
        print(
            'Successfully loaded imagenet pretrained weights from "{}"'.
            format(cached_file)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )


##########
# Instantiation
##########
def osnet_ain_x1_0(
    num_classes=1000, pretrained=True, loss='softmax', **kwargs
):
    model = OSNet(
        num_classes,
        blocks=[
            [OSBlockINin, OSBlockINin], [OSBlock, OSBlockINin],
            [OSBlockINin, OSBlock]
        ],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        loss=loss,
        conv1_IN=True,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, key='osnet_ain_x1_0')
    return model


def osnet_ain_x0_75(
    num_classes=1000, pretrained=True, loss='softmax', **kwargs
):
    model = OSNet(
        num_classes,
        blocks=[
            [OSBlockINin, OSBlockINin], [OSBlock, OSBlockINin],
            [OSBlockINin, OSBlock]
        ],
        layers=[2, 2, 2],
        channels=[48, 192, 288, 384],
        loss=loss,
        conv1_IN=True,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, key='osnet_ain_x0_75')
    return model


def osnet_ain_x0_5(
    num_classes=1000, pretrained=True, loss='softmax', **kwargs
):
    model = OSNet(
        num_classes,
        blocks=[
            [OSBlockINin, OSBlockINin], [OSBlock, OSBlockINin],
            [OSBlockINin, OSBlock]
        ],
        layers=[2, 2, 2],
        channels=[32, 128, 192, 256],
        loss=loss,
        conv1_IN=True,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, key='osnet_ain_x0_5')
    return model


def osnet_ain_x0_25(
    num_classes=1000, pretrained=True, loss='softmax', **kwargs
):
    model = OSNet(
        num_classes,
        blocks=[
            [OSBlockINin, OSBlockINin], [OSBlock, OSBlockINin],
            [OSBlockINin, OSBlock]
        ],
        layers=[2, 2, 2],
        channels=[16, 64, 96, 128],
        loss=loss,
        conv1_IN=True,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, key='osnet_ain_x0_25')
    return model



# ---------------------------
# 修改后的 OSNet 结构：引入全局分支和局部分支
# ---------------------------
class OSNet_Modified(nn.Module):
    """
    修改后的 OSNet：
      - 输入图像 (B, 3, 256, 128) 经过前几层得到特征图 x_base，尺寸为 (B, 256, 64, 32)。
      - 在 conv2 后分裂为两条分支：
          * 全局分支：原有结构 —— pool2, conv3, pool3, conv4, conv5, global_avgpool，再经 FC 得到全局特征向量。
          * 局部分支：先将 x_base 与经 1×1 卷积变换的热图（输入热图尺寸为 (B,17,64,32)）做逐元素乘法，得到融合特征，
            再经过一套结构与全局分支相同（但参数独立）的后续层，得到局部特征向量。
      - 最后将两个分支得到的特征向量拼接后归一化，作为最终输出。
    """
    def __init__(self, num_classes, blocks, layers, channels, feature_dim=512, loss='softmax', conv1_IN=False, **kwargs):
        super(OSNet_Modified, self).__init__()
        self.loss = loss
        self.feature_dim = feature_dim
        fc_dim = feature_dim
        # 前处理部分
        # 输入图像尺寸：(B,3,256,128)
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=conv1_IN)
        # conv1 输出 -> (B,64,256/2,128/2) ≈ (B,64,128,64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        # maxpool 输出 -> (B,64,64,32)

        # 中间特征提取（分支分裂前）
        # conv2 将通道数从 64 扩展到 256，输出尺寸保持 (B,256,64,32)
        self.conv2 = self._make_layer(blocks[0], layers[0], channels[0], channels[1])
        # 之后在 conv2 输出上进行分裂

        # 全局分支后续部分（与原结构相同）
        self.pool2 = nn.Sequential(
            Conv1x1(channels[1], channels[1]),
            nn.AvgPool2d(2, stride=2)  # 下采样：尺寸 (B,256,64,32) -> (B,256,32,16)
        )
        self.conv3 = self._make_layer(blocks[1], layers[1], channels[1], channels[2])
        self.pool3 = nn.Sequential(
            Conv1x1(channels[2], channels[2]),
            nn.AvgPool2d(2, stride=2)  # 下采样，例如 (B,384,32,16) -> (B,384,16,8)
        )
        self.conv4 = self._make_layer(blocks[2], layers[2], channels[2], channels[3])
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)  # 输出 (B,512,1,1)
        self.fc = self._construct_fc_layer(self.feature_dim, channels[3], dropout_p=None)
        self.global_subnet = nn.Sequential(
            self.pool2,
            self.conv3,
            self.pool3,
            self.conv4,
            self.conv5,
            self.global_avgpool
        )
        self.global_fc = self.fc

        # 局部分支：先对热图进行 1×1 卷积变换，再与 conv2 输出逐元素相乘，
        # 然后经过与全局分支结构一致但参数独立的一套层
        self.heatmap_conv = nn.Sequential(
            nn.Conv2d(17, channels[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU()
        )
        self.local_pool2 = nn.Sequential(
            Conv1x1(channels[1], channels[1]),
            nn.AvgPool2d(2, stride=2)
        )
        self.local_conv3 = self._make_layer(blocks[1], layers[1], channels[1], channels[2])
        self.local_pool3 = nn.Sequential(
            Conv1x1(channels[2], channels[2]),
            nn.AvgPool2d(2, stride=2)
        )
        self.local_conv4 = self._make_layer(blocks[2], layers[2], channels[2], channels[3])
        self.local_conv5 = Conv1x1(channels[3], channels[3])
        self.local_avgpool = nn.AdaptiveAvgPool2d(1)
        self.local_fc = self._construct_fc_layer(self.feature_dim, channels[3], dropout_p=None)

        # 修改后的分类器：最终特征向量为全局与局部拼接后的 2*feature_dim
        self.local_classifier = nn.Linear(fc_dim , num_classes)
        self.global_classifier = nn.Linear(fc_dim , num_classes)

        self._init_params()

    def _make_layer(self, blocks, layer, in_channels, out_channels):
        layers = [blocks[0](in_channels, out_channels)]
        for i in range(1, len(blocks)):
            layers.append(blocks[i](out_channels, out_channels))
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None
        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, heatmap, return_featuremaps=False):
        # x: 输入图像，尺寸 (B, 3, 256, 128)
        # heatmap: 输入热图，尺寸 (B, 17, 64, 32)
        # ----- 前处理 -----
        x = self.conv1(x)  # -> (B, 64, 128, 64)
        x = self.maxpool(x)  # -> (B, 64, 64, 32)
        x = self.conv2(x)  # -> (B, 256, 64, 32)
        # 此处 x 为分支分裂前的特征图（x_base）

        # ----- 全局分支 -----
        # 经过 pool2 下采样： (B,256,64,32) -> (B,256,32,16)
        # conv3、pool3、conv4、conv5、global_avgpool 后，输出尺寸为 (B,512,1,1)
        x_global = self.global_subnet(x)  # -> (B, 512, 1, 1)
        x_global = x_global.view(x_global.size(0), -1)  # -> (B, 512)
        x_global = self.global_fc(x_global)  # -> (B, feature_dim)；feature_dim 通常为 512

        # ----- 局部分支 -----
        # 对输入热图先进行 1×1 卷积变换： (B,17,64,32) -> (B,256,64,32)
        h = self.heatmap_conv(heatmap)  # -> (B, 256, 64, 32)
        # 将 conv2 的输出与变换后的热图逐元素相乘（通道、空间完全对应）
        x_fused = x * h  # -> (B, 256, 64, 32)
        # 经过局部分支的各层：
        x_local = self.local_pool2(x_fused)  # -> (B,256,32,16)
        x_local = self.local_conv3(x_local)    # -> (B,384,32,16)
        x_local = self.local_pool3(x_local)      # -> (B,384,16,8)
        x_local = self.local_conv4(x_local)      # -> (B,512,16,8)
        x_local = self.local_conv5(x_local)      # -> (B,512,16,8)
        x_local = self.local_avgpool(x_local)  # -> (B,512,1,1)
        x_local = x_local.view(x_local.size(0), -1)     # -> (B,512)
        x_local = self.local_fc(x_local)                # -> (B, feature_dim)


        # ----- 特征融合 -----
        # 拼接全局与局部特征向量，得到 (B, 2*feature_dim)
        # out = torch.cat([x_global, x_local], dim=1)
        # L2 归一化
        # out = F.normalize(out, p=2, dim=1)

        if self.training:
            # 训练阶段返回分类 logits 以及归一化特征
            local_logits = self.local_classifier(x_local)  # -> (B, num_classes)
            golbal_logits = self.global_classifier(x_global)  # -> (B, num_classes)
            return x_global,x_local,golbal_logits,local_logits
        else:
            # 测试阶段只返回最终特征向量
            return x_global,x_local
        



def init_pretrained_weights_mod(model, key=''):
    """Initializes OSNet_Modified model with pretrained weights.
    
    该函数首先加载预训练参数，然后：
      - 对于模型中与预训练参数键名直接匹配的层，直接加载；
      - 对于局部分支（local_开头）的层，尝试使用对应全局分支的权重（去掉 local_ 前缀）进行初始化。
    """
    import os
    import errno
    import gdown
    from collections import OrderedDict
    import torch
    import warnings

    def _get_torch_home():
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch')
            )
        )
        return torch_home

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    filename = key + '_imagenet.pth'
    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        gdown.download(pretrained_urls[key], cached_file, quiet=False)

    state_dict = torch.load(cached_file)
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    # ---------------------------
    # 第一轮：直接匹配全局分支及共享部分的权重
    # ---------------------------
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # 去掉前缀 "module."
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    # ---------------------------
    # 第二轮：针对局部分支，将对应全局分支权重复制过去
    # 遍历模型参数，找到以 "local_" 开头的键，去掉该前缀后查找预训练权重
    # ---------------------------
    for k in model_dict.keys():
        if k.startswith('local_') and k not in new_state_dict:
            # 构造对应全局分支的键
            global_key = k.replace('local_', '', 1)
            if global_key in new_state_dict:
                new_state_dict[k] = new_state_dict[global_key].clone()
                matched_layers.append(k)
            elif global_key in state_dict and state_dict[global_key].size() == model_dict[k].size():
                new_state_dict[k] = state_dict[global_key]
                matched_layers.append(k)
            else:
                discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights from "{}" cannot be loaded, please check the key names manually '
            '(** ignored and continue **)'.format(cached_file)
        )
    else:
        print('Successfully loaded imagenet pretrained weights from "{}"'.format(cached_file))
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded due to unmatched keys or layer size: {}'
                .format(discarded_layers)
            )



def osnetmod_ain_x1_0(
    num_classes=1000, pretrained=True, loss='softmax', **kwargs
):
    model = OSNet_Modified(
        num_classes,
        blocks=[
            [OSBlockINin, OSBlockINin], [OSBlock, OSBlockINin],
            [OSBlockINin, OSBlock]
        ],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        loss=loss,
        conv1_IN=True,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights_mod(model, key='osnet_ain_x1_0')
    return model