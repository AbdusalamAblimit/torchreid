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
class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False, groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

###############################################
# 定义局部子网络，用于处理单一关键点分支
###############################################
class LocalSubnet(nn.Module):
    def __init__(self, blocks, layers, channels, feature_dim=512, dropout_p=None):
        """
        blocks: 包含各阶段 block 构造函数的列表（例如 OSNet 使用的 block 列表）
        layers: 原始 OSNet 配置中各阶段的层数（用于与 blocks 长度保持一致，本例中可忽略）
        channels: 如 [64, 256, 384, 512]，对应 conv1, conv2, conv3, conv4/conv5 等通道数
        feature_dim: 输出局部特征的维度，通常设置为 512
        """
        super(LocalSubnet, self).__init__()
        # 先将单通道热图经 1×1 卷积扩充到 conv2 输出通道数（例如 256）
        self.heatmap_conv = nn.Sequential(
            nn.Conv2d(1, channels[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True)
        )
        # 后续结构与全局分支类似
        self.pool2 = nn.Sequential(
            Conv1x1(channels[1], channels[1]),
            nn.AvgPool2d(2, stride=2)  # 下采样: (B,256,64,32) -> (B,256,32,16)
        )
        # 注意此处 _make_layer 改为只传入 blocks 列表、in_channels 及 out_channels
        self.conv3 = self._make_layer(blocks[1], channels[1], channels[2])
        self.pool3 = nn.Sequential(
            Conv1x1(channels[2], channels[2]),
            nn.AvgPool2d(2, stride=2)  # 下采样: (B,384,32,16) -> (B,384,16,8)
        )
        self.conv4 = self._make_layer(blocks[2], channels[2], channels[3])
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 输出尺寸 (B, channels[3], 1, 1)
        self.fc = self._construct_fc_layer(feature_dim, channels[3], dropout_p=dropout_p)
        self._init_local_params()

    def _make_layer(self, blocks, in_channels, out_channels):
        # 与原始 OSNet 中的实现一致：
        layers = []
        layers.append(blocks[0](in_channels, out_channels))
        for i in range(1, len(blocks)):
            layers.append(blocks[i](out_channels, out_channels))
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dim, input_dim, dropout_p=None):
        if fc_dim is None or fc_dim < 0:
            return None
        fc_layers = []
        fc_layers.append(nn.Linear(input_dim, fc_dim))
        fc_layers.append(nn.BatchNorm1d(fc_dim))
        fc_layers.append(nn.ReLU(inplace=True))
        if dropout_p is not None:
            fc_layers.append(nn.Dropout(p=dropout_p))
        return nn.Sequential(*fc_layers)

    def _init_local_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, heatmap_channel):
        """
        x: conv2 输出，形状 (B,256,64,32)
        heatmap_channel: 单个部位热图，形状 (B,1,64,32)
        """
        # 将单通道热图经过 1×1 卷积扩充为 (B,256,64,32)
        h = self.heatmap_conv(heatmap_channel)
        # 与 conv2 输出逐元素相乘
        x_fused = x * h  
        x_fused = self.pool2(x_fused)
        x_fused = self.conv3(x_fused)
        x_fused = self.pool3(x_fused)
        x_fused = self.conv4(x_fused)
        x_fused = self.conv5(x_fused)
        x_fused = self.avgpool(x_fused)   # (B, channels[3], 1, 1)
        x_fused = x_fused.view(x_fused.size(0), -1)  # 展平为 (B, channels[3])
        x_feat = self.fc(x_fused)  # (B, feature_dim)
        return x_feat

#########################################
# 修改后的 OSNet 模型，包含全局和 17 个局部分支
#########################################
class OSNet_Modified(nn.Module):
    def __init__(self, num_classes, blocks, layers, channels, feature_dim=512, loss='softmax', conv1_IN=False, dropout_p=None, **kwargs):
        """
        num_classes: 分类数
        blocks, layers, channels: 根据 OSNet 配置，例如 channels = [64, 256, 384, 512]
        feature_dim: 全局与局部特征最终维度（例如 512）
        loss: 损失类型（例如 softmax）
        conv1_IN: 控制 conv1 是否使用 InstanceNorm
        dropout_p: FC层是否使用 dropout
        """
        super(OSNet_Modified, self).__init__()
        self.loss = loss
        self.feature_dim = feature_dim
        self.num_parts = 17  # 关键点数量
        # 前处理部分，输入图像尺寸 (B,3,256,128)
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=conv1_IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        # 中间特征提取（分支分裂前）：conv2 将通道数扩展到 256，输出 (B,256,64,32)
        self.conv2 = self._make_layer(blocks[0], channels[0], channels[1])
        
        # -----------------
        # 全局分支（保持原设计）
        # -----------------
        self.pool2 = nn.Sequential(
            Conv1x1(channels[1], channels[1]),
            nn.AvgPool2d(2, stride=2)  # (B,256,64,32) -> (B,256,32,16)
        )
        self.conv3 = self._make_layer(blocks[1], channels[1], channels[2])
        self.pool3 = nn.Sequential(
            Conv1x1(channels[2], channels[2]),
            nn.AvgPool2d(2, stride=2)  # (B,384,32,16) -> (B,384,16,8)
        )
        self.conv4 = self._make_layer(blocks[2], channels[2], channels[3])
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)  # 输出 (B, channels[3],1,1)
        self.global_fc = self._construct_fc_layer(feature_dim, channels[3], dropout_p=dropout_p)
        self.global_subnet = nn.Sequential(
            self.pool2,
            self.conv3,
            self.pool3,
            self.conv4,
            self.conv5,
            self.global_avgpool
        )
        self.global_classifier = nn.Linear(feature_dim, num_classes)
        
        # -----------------
        # 局部分支：独立的 17 个局部子网络
        # -----------------
        self.local_subnets = nn.ModuleList([
            LocalSubnet(blocks, layers, channels, feature_dim, dropout_p=dropout_p)
            for _ in range(self.num_parts)
        ])
        # 定义局部分类器：先拼接 17 个局部特征（[B, 17*feature_dim]），再计算分类 logits
        self.local_classifier = nn.Linear(feature_dim, num_classes)

        self._init_params()

    def _make_layer(self, blocks, in_channels, out_channels):
        # 与原始 OSNet 的 _make_layer 类似：遍历 blocks 列表构建层
        layers_list = []
        layers_list.append(blocks[0](in_channels, out_channels))
        for i in range(1, len(blocks)):
            layers_list.append(blocks[i](out_channels, out_channels))
        return nn.Sequential(*layers_list)

    def _construct_fc_layer(self, fc_dim, input_dim, dropout_p=None):
        if fc_dim is None or fc_dim < 0:
            self.feature_dim = input_dim
            return None
        if isinstance(fc_dim, int):
            fc_dims = [fc_dim]
        else:
            fc_dims = fc_dim

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
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
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, heatmap, visibility, return_featuremaps=False):
        """
        x: 输入图像，尺寸 (B,3,256,128)
        heatmap: 关键点热图，尺寸 (B,17,64,32)
        visibility: 可见性信息，尺寸 (B,17,1) 或 (B,17)，表示每个关键点的置信度
        return_featuremaps: 如果 True，则返回 feature maps 供调试使用（默认为 False）
        """
        # ------ 前处理 ------
        x = self.conv1(x)       # (B,64,128,64)
        x = self.maxpool(x)     # (B,64,64,32)
        x = self.conv2(x)       # (B,256,64,32)
        
        # ------ 全局分支 ------
        x_global = self.global_subnet(x)   # (B,channels[3],1,1)
        x_global = x_global.view(x_global.size(0), -1)
        x_global = self.global_fc(x_global)  # (B, feature_dim)
        global_logits = self.global_classifier(x_global)
        
        # ------ 局部分支 ------
        # 将 heatmap 按通道分割为 17 个 (B,1,64,32)
        heatmap_channels = torch.split(heatmap, 1, dim=1)
        local_feats = []
        for i in range(self.num_parts):
            feat_local = self.local_subnets[i](x, heatmap_channels[i])  # 输出形状 (B, feature_dim)
            local_feats.append(feat_local)
        # 堆叠得到 (B,17,feature_dim)
        local_feats_tensor = torch.stack(local_feats, dim=1)  # shape: (B, 17, feature_dim)
        
        # 调整 visibility 的形状：若为 (B,17,1) 则 squeeze 成 (B,17)
        if visibility.dim() == 3:
            visibility = visibility.squeeze(-1)
        
        # 针对每个样本，根据阈值筛选局部特征
        threshold = 0.3
        batch_size = local_feats_tensor.size(0)
        aggregated_local_list = []
        for b in range(batch_size):
            vis_b = visibility[b]            # shape: (17,)
            feats_b = local_feats_tensor[b]    # shape: (17, feature_dim)
            # 找到置信度大于阈值的索引
            valid_idx = (vis_b > threshold).nonzero(as_tuple=False).squeeze()
            if valid_idx.numel() >= 3:
                selected_feats = feats_b[valid_idx]
            else:
                # 如果不足 3 个，则取置信度最高的 3 个索引
                sorted_vis, sorted_idx = torch.sort(vis_b, descending=True)
                top3_idx = sorted_idx[:3]
                selected_feats = feats_b[top3_idx]
            # 简单地均值聚合（当前各向量权重均为 1.0）
            aggregated_feat = selected_feats.mean(dim=0)  # shape: (feature_dim,)
            aggregated_local_list.append(aggregated_feat)
        aggregated_local = torch.stack(aggregated_local_list, dim=0)  # shape: (B, feature_dim)
        
        # 根据新的 aggregated_local 计算局部分类 logits
        local_logits = self.local_classifier(aggregated_local)
        
        if self.training:
            return x_global, aggregated_local, global_logits, local_logits
        else:
            return x_global, aggregated_local

############################################
# 初始化预训练权重（修改后的版本）
############################################
def init_pretrained_weights_mod(model, key=''):
    """
    初始化 OSNet_Modified 模型的预训练权重：
      1. 优先加载与预训练权重键名直接匹配的层（全局分支和共享部分）。
      2. 对于局部子网络中的层（位于 local_subnets 模块中），若去掉局部前缀后与全局分支中对应的层匹配，
         则复制其权重。
    注意：局部子网络中独有的层（例如 heatmap_conv）由于没有全局对应结构，不进行初始化。
    """
    import os, errno, gdown, warnings
    from collections import OrderedDict
    torch_home = os.path.expanduser(
        os.getenv('TORCH_HOME', os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch'))
    )
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    filename = key + '_imagenet.pth'
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        gdown.download(pretrained_urls[key], cached_file, quiet=False)
    state_dict = torch.load(cached_file)
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []
    # 第一轮：直接匹配全局分支及共享部分
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # 第二轮：针对局部子网络，从全局分支复制对应权重
    mapping = {
        'pool2': 'pool2',
        'conv3': 'conv3',
        'pool3': 'pool3',
        'conv4': 'conv4',
        'conv5': 'conv5',
        'fc': 'global_fc'
    }
    for k in model_dict.keys():
        if k.startswith('local_subnets.'):
            parts = k.split('.')
            if len(parts) < 3:
                continue
            sub_layer = parts[2]
            if sub_layer not in mapping:
                continue
            global_key = mapping[sub_layer]
            if len(parts) > 3:
                global_key = global_key + '.' + '.'.join(parts[3:])
            if global_key in new_state_dict and new_state_dict[global_key].size() == model_dict[k].size():
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
        warnings.warn('The pretrained weights from "{}" cannot be loaded, please check the key names manually (** ignored and continue **)'.format(cached_file))
    else:
        print('Successfully loaded imagenet pretrained weights from "{}"'.format(cached_file))
        if len(discarded_layers) > 0:
            print('** The following layers are discarded due to unmatched keys or layer size: {}'.format(discarded_layers))

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



def osnetmod_ain_x0_25(
    num_classes=1000, pretrained=True, loss='softmax', **kwargs
):
    model = OSNet_Modified(
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
        init_pretrained_weights_mod(model, key='osnet_ain_x0_25')
    return model