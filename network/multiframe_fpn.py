import torch
import torch.nn as nn
import torch.nn.functional as F


cfg = {
    'X': [16, 'M', 32, 32, 'M', 32, 32, 'M', 64, 64, 'M', 64, 64, 'M'],
}


class TemporalFusionModule(nn.Module):
    def __init__(self, input_channels=9, reduced_channels=3):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(input_channels, input_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels // 2, reduced_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.fusion(x)


def make_multiframe_modules(cfg, batch_norm=False, input_channels=3):
    modules = nn.ModuleList()
    out_channels = []

    in_channels = input_channels
    layers = []

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            modules.append(nn.Sequential(*layers))
            out_channels.append(in_channels)
            layers = []
        else:
            if batch_norm:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    assert len(layers) == 0

    return modules, out_channels


class MultiFrameFPN(nn.Module):
    def __init__(self, layers, out_channels, lateral_channels, return_layers=None, input_channels=3):
        super(MultiFrameFPN, self).__init__()

        assert len(layers) == len(out_channels)

        self.layers = layers
        self.out_channels = out_channels
        self.lateral_channels = lateral_channels
        self.input_channels = input_channels
        self.lateral_layers = nn.ModuleList()
        self.smooth_layers = nn.ModuleList()

        if return_layers is None:
            self.return_layers = list(range(len(layers) - 1))
        else:
            self.return_layers = return_layers
        self.min_returned_layer = min(self.return_layers)

        for i in range(self.min_returned_layer, len(self.layers)):
            self.lateral_layers.append(
                nn.Conv2d(out_channels[i], self.lateral_channels, kernel_size=1, stride=1, padding=0)
            )

        # 去掉原先的self.temporal_fusion模块
        self.temporal_fusion = None

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # 输入x现在应该是3通道（降维后）

        # Bottom-up
        c = []
        for m in self.layers:
            x = m(x)
            c.append(x)

        # Top-down
        p = [self.lateral_layers[-1](c[-1])]
        for i in range(len(c) - 2, self.min_returned_layer - 1, -1):
            temp = self._upsample_add(p[-1], self.lateral_layers[i - self.min_returned_layer](c[i]))
            p.append(temp)

        p = p[::-1]

        out_tensors = []
        for ndx, l in enumerate(self.return_layers):
            out_tensors.append(p[l - self.min_returned_layer])

        return out_tensors


class MultiFrameFusionFPN(nn.Module):
    def __init__(self, cfg_name='X', num_frames=3, batch_norm=True, lateral_channels=32, return_layers=None):
        super().__init__()
        input_channels = 3 * num_frames
        self.temporal_fusion = TemporalFusionModule(input_channels=input_channels, reduced_channels=3)
        layers, out_channels = make_multiframe_modules(cfg[cfg_name], batch_norm, input_channels=3)
        if return_layers is None:
            return_layers = [1, 3]
        self.fpn = MultiFrameFPN(layers, out_channels, lateral_channels, return_layers, input_channels=3)

    def forward(self, x):
        # x shape: [B, 3*num_frames, H, W]
        x = self.temporal_fusion(x)  # 降维成3通道
        out = self.fpn(x)
        return out


# 下面是示例调用
if __name__ == '__main__':
    model = MultiFrameFusionFPN(cfg_name='X', num_frames=3)
    dummy_input = torch.randn(16, 9, 720, 1280)
    outputs = model(dummy_input)
    for i, feat in enumerate(outputs):
        print(f'Output feature map {i} shape: {feat.shape}')

