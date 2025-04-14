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

class OSNet_Modified(nn.Module):
    def __init__(self, num_classes, blocks, layers, channels, feature_dim=512,
                 loss='softmax', conv1_IN=False, dropout_p=None, **kwargs):
        """
        num_classes: 分类数
        blocks, layers, channels: 根据 OSNet 配置，例如 channels = [64, 256, 384, 512]
        feature_dim: 全局与局部特征最终维度（例如 512）
        loss: 损失类型（例如 softmax）
        conv1_IN: 是否使用 InstanceNorm
        dropout_p: 全连接层 dropout 概率
        """
        super(OSNet_Modified, self).__init__()
        self.loss = loss
        self.feature_dim = feature_dim
        self.num_parts = 17  # 关键点数量

        # 前处理部分，输入图像尺寸 (B,3,256,128)
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=conv1_IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        # conv2：输出 (B,256,64,32)
        self.conv2 = self._make_layer(blocks[0], channels[0], channels[1])
        
        # -----------------
        # 全局分支（保持原设计）
        # -----------------
        self.pool2 = nn.Sequential(
            Conv1x1(channels[1], channels[1]),
            nn.AvgPool2d(2, stride=2)
        )
        self.conv3 = self._make_layer(blocks[1], channels[1], channels[2])
        self.pool3 = nn.Sequential(
            Conv1x1(channels[2], channels[2]),
            nn.AvgPool2d(2, stride=2)
        )
        self.conv4 = self._make_layer(blocks[2], channels[2], channels[3])
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
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
        # 局部分支：新的设计
        # 我们利用 heatmap 与可见性对 conv2 输出进行融合，
        # 然后用一个 1x1 卷积（local_heatmap_conv）将 17 通道的 heatmap 转换为 256 通道，
        # 最后与基础特征做加法，再通过共享网络 local_branch 提取最终局部特征。
        # -----------------
        # 用于对原始 heatmap（17通道）进行 1x1 卷积，映射到 channels[1]=256 通道
        self.local_heatmap_conv = nn.Sequential(
            nn.Conv2d(17, channels[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True)
        )
        # 共享局部分支，结构可仿照全局分支后半部分，但输入为 (B,256,64,32)
        self.local_branch = nn.Sequential(
            Conv1x1(channels[1], channels[1]),
            nn.AvgPool2d(2, stride=2),                                # (B,256,32,16)
            self._make_layer(blocks[1], channels[1], channels[2]),     # (B,384,32,16)
            nn.AvgPool2d(2, stride=2),                                # (B,384,16,8)
            self._make_layer(blocks[2], channels[2], channels[3]),     # (B,512,16,8)
            Conv1x1(channels[3], channels[3]),
            nn.AdaptiveAvgPool2d(1)                                   # (B,512,1,1)
        )
        self.local_fc = self._construct_fc_layer(feature_dim, channels[3], dropout_p=dropout_p)
        self.local_classifier = nn.Linear(feature_dim, num_classes)

        self._init_params()

    def _make_layer(self, blocks, in_channels, out_channels):
        layers_list = []
        layers_list.append(blocks[0](in_channels, out_channels))
        for i in range(1, len(blocks)):
            layers_list.append(blocks[i](out_channels, out_channels))
        return nn.Sequential(*layers_list)

    def _construct_fc_layer(self, fc_dim, input_dim, dropout_p=None):
        if fc_dim is None or fc_dim < 0:
            self.feature_dim = input_dim
            return None
        fc_dims = [fc_dim] if isinstance(fc_dim, int) else fc_dim
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
        return_featuremaps: 如果 True，则返回 feature maps 用于调试（默认为 False）
        """
        # ------ 前处理 ------
        x = self.conv1(x)          # (B,64,128,64)
        x = self.maxpool(x)        # (B,64,64,32)
        x = self.conv2(x)          # (B,256,64,32)
        
        # ------ 全局分支 ------
        x_global = self.global_subnet(x)      # (B,channels[3],1,1)
        x_global = x_global.view(x_global.size(0), -1)
        x_global = self.global_fc(x_global)     # (B, feature_dim)
        global_logits = self.global_classifier(x_global)
        
        # ------ 局部分支（新设计）------
        # 处理 heatmap 与可见性：先将每个 heatmap 通道乘上对应的可见性
        if visibility.dim() == 3:
            visibility = visibility.squeeze(-1)    # (B,17)
        vis = visibility.view(visibility.size(0), visibility.size(1), 1, 1)  # (B,17,1,1)
        new_heatmap = heatmap  # * vis            # (B,17,64,32)
        # 通过 1×1 卷积将 17 通道映射为 256 通道
        new_heatmap_proj = self.local_heatmap_conv(new_heatmap)  # (B,256,64,32)
        # 与基础特征 x 相加融合
        local_fused = x * new_heatmap_proj     # (B,256,64,32)
        # 输入共享局部分支网络，提取局部特征
        local_feature_map = self.local_branch(local_fused)  # (B,channels[3],1,1)
        local_feature_map = local_feature_map.view(local_feature_map.size(0), -1)  # (B, channels[3])
        aggregated_local = self.local_fc(local_feature_map)   # (B, feature_dim)
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
      1. 优先加载与预训练权重键名直接匹配的层（主要包括全局分支和公共层）。
      2. 对于局部分支中部分模块（如 local_fc 和 local_classifier），若在全局分支中有对应模块，
         则将全局权重复制到局部分支中，以利用预训练知识。
    注意：
      - 对于新设计中没有全局对应关系的模块（例如 local_heatmap_conv、local_branch），
        将不进行权重复制，而使用随机初始化（或者你可额外设置其他初始化方案）。
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
    matched_layers = []
    discarded_layers = []

    # 第一轮：直接匹配（主要针对全局分支和共享部分）
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    # 第二轮：对于新局部模块中的部分进行权重复制
    # 例如：将 local_fc 初始化为 global_fc 的权重，
    #        将 local_classifier 初始化为 global_classifier 的权重
    mapping = {
        'local_fc': 'global_fc',
        'local_classifier': 'global_classifier'
    }
    for local_prefix, global_prefix in mapping.items():
        for k, v in model_dict.items():
            if k.startswith(local_prefix):
                # 构造对应的全局 key：替换 local_prefix 为 global_prefix
                global_key = k.replace(local_prefix, global_prefix, 1)
                if global_key in new_state_dict and new_state_dict[global_key].size() == v.size():
                    new_state_dict[k] = new_state_dict[global_key].clone()
                    matched_layers.append(k)
                elif global_key in state_dict and state_dict[global_key].size() == v.size():
                    new_state_dict[k] = state_dict[global_key]
                    matched_layers.append(k)
                else:
                    discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights from "{}" cannot be loaded, please check the key names manually '
            '(** ignored and continue **).'.format(cached_file)
        )
    else:
        print('Successfully loaded imagenet pretrained weights from "{}"'.format(cached_file))
        if len(discarded_layers) > 0:
            print('** The following layers are discarded due to unmatched keys or layer size: {}'
                  .format(discarded_layers))

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