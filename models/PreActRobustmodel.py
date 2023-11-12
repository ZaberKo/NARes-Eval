import math
from re import S
from tokenize import group
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from models.Activation import avaliable_activations
    from models.Normalization import avaliable_normalizations
except:
    from Activation import avaliable_activations
    from Normalization import avaliable_normalizations


class PreActBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride, kernel_size=3, drop_rate=0.0, activation='ReLU', normalization='BatchNorm', **kwargs):
        super(PreActBasicBlock, self).__init__()
        self.act = avaliable_activations[activation](inplace=True)
        self.bn1 = avaliable_normalizations[normalization](in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn2 = avaliable_normalizations[normalization](planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
            )

    def forward(self, x):
        out = self.act(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.act(self.bn2(out)))
        return out + shortcut


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, stride, kernel_size=3, drop_rate=0.0, block_type='basic_block',
                 activation='ReLU', normalization='BatchNorm'):
        super(NetworkBlock, self).__init__()
        self.block_type = block_type
        if block_type == 'basic_block':
            block = PreActBasicBlock
        else:
            raise('Unknown block: %s' % (block_type))

        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, kernel_size,
                                      drop_rate, activation, normalization)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, kernel_size, drop_rate, activation, normalization):
        layers = []
        for i in range(int(nb_layers)):
            if i == 0:
                in_planes = in_planes
            else:
                in_planes = out_planes
            layers.append(block(in_planes, out_planes, i == 0 and stride or 1, kernel_size=kernel_size,
                                drop_rate=drop_rate, activation=activation, normalization=normalization)
                          )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class PreActRobustNetwork(nn.Module):
    def __init__(self, num_classes=10, channel_configs=(16, 160, 320, 640),
                 depth_configs=(5, 5, 5), stride_config=(1, 2, 2), stem_stride=1,
                 drop_rate_config=(0.0, 0.0, 0.0), zero_init_residual=False,
                 kernel_size_configs=(3, 3, 3),
                 block_types=('basic_block', 'basic_block', 'basic_block'),
                 activations=('ReLU', 'ReLU', 'ReLU'),
                 normalizations=('BatchNorm', 'BatchNorm', 'BatchNorm'), use_init=True):
        super(PreActRobustNetwork, self).__init__()
        assert len(channel_configs) - 1 == len(depth_configs) == len(stride_config) == len(drop_rate_config)
        self.channel_configs = channel_configs
        self.depth_configs = depth_configs
        self.stride_config = stride_config
        self.get_feature = False
        self.get_stem_out = False
        self.block_types = block_types

        self.stem_conv = nn.Conv2d(3, channel_configs[0], kernel_size=3, stride=stem_stride, padding=1, bias=False)
        self.blocks = nn.ModuleList([])

        out_planes = channel_configs[0]
        for i, stride in enumerate(stride_config):
            self.blocks.append(NetworkBlock(nb_layers=depth_configs[i],
                                            in_planes=out_planes,
                                            out_planes=channel_configs[i+1],
                                            stride=stride,
                                            drop_rate=drop_rate_config[i],
                                            kernel_size=kernel_size_configs[i],
                                            block_type=block_types[i],
                                            activation=activations[i],
                                            normalization=normalizations[i]
                                            ))
            out_planes = channel_configs[i+1]

        # global average pooling and classifier
        self.norm1 = avaliable_normalizations[normalizations[-1]](out_planes)
        self.act1 = avaliable_activations[activations[-1]](inplace=True)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_planes, num_classes)
        self.fc_size = out_planes
        if use_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, x):
        out = self.stem_conv(x)
        for i, block in enumerate(self.blocks):
            out = block(out)
        out = self.act1(self.norm1(out))
        out = self.global_pooling(out)
        out = out.view(-1, self.fc_size)
        self.features = out
        out = self.fc(out)
        return out


if __name__ == '__main__':
    from torchprofile import profile_macs

    drop_rate_config = (0.0, 0.0, 0.0)
    stride_config = [1, 2, 2]
    depth, mulpliers = [34, 34, 17], [13, 16, 8]

    channels = [16, 16*mulpliers[0], 32*mulpliers[1], 64*mulpliers[2]]
    expected_depth, expected_width = sum(depth), sum(mulpliers)
    kernel_size = [3, 3, 3]
    widen_factor = 9
    se_reduction = 16
    cardinality, base_width, scales = 1, 160, 1
    activations = ('ReLU', 'ReLU', 'ReLU')
    normalizations = ('BatchNorm', 'BatchNorm', 'BatchNorm')
    block_types = ['basic_block', 'basic_block', 'basic_block']
    model = PreActRobustNetwork(
        num_classes=10, channel_configs=channels, depth_configs=depth,
        stride_config=stride_config, stem_stride=1,
        drop_rate_config=drop_rate_config,
        kernel_size_configs=kernel_size,
        zero_init_residual=False,
        block_types=block_types,
        activations=activations,
        normalizations=normalizations,
        use_init=True,
    )
    print(model)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    data = torch.rand(1, 3, 32, 32)
    out = model(data)
    flops = profile_macs(model, data) / 1e6
    print(expected_depth / (expected_depth + expected_width))
    print('depth@{}-{}-width@{}-{}-channels@{}-block@{}-params = {:.3f}, flops = {:.3f}'.format(expected_depth, depth, expected_width, mulpliers, channels, block_types[0], param_count, flops / 1000))

