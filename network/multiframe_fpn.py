# Multi-frame FPN network for Early Fusion
# 支持多帧输入的特征金字塔网络

import torch.nn as nn
import torch.nn.functional as F


cfg = {
    # Config according to the Table 1 from the FootAndBall paper
    'X': [16, 'M', 32, 32, 'M', 32, 32, 'M', 64, 64, 'M', 64, 64, 'M'],
}


def make_multiframe_modules(cfg, batch_norm=False, input_channels=3):
    """
    Create modules with configurable input channels for multi-frame input
    
    Args:
        cfg: Network configuration
        batch_norm: Whether to use batch normalization
        input_channels: Number of input channels (3 for single frame, 9 for 3 frames, etc.)
    """
    # Each module is a list of sequential layers operating at the same spacial dimension followed by MaxPool2d
    modules = nn.ModuleList()
    # Number of output channels in each module
    out_channels = []

    in_channels = input_channels  # Modified to support multi-frame input
    layers = []

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            # Create new module with accumulated layers and flush layers list
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

    # 'M' should be the last layer - and all layers should be flushed
    assert len(layers) == 0

    return modules, out_channels


def make_modules(cfg, batch_norm=False):
    """Original single-frame function for backward compatibility"""
    return make_multiframe_modules(cfg, batch_norm, input_channels=3)


class MultiFrameFPN(nn.Module):
    """FPN with support for multi-frame input through Early Fusion"""
    def __init__(self, layers, out_channels, lateral_channels, return_layers=None, input_channels=3):
        # return_layers: index of layers (numbered from 0) for which feature maps are returned
        # input_channels: number of input channels (3*num_frames for Early Fusion)
        super(MultiFrameFPN, self).__init__()

        assert len(layers) == len(out_channels)

        self.layers = layers
        self.out_channels = out_channels
        self.lateral_channels = lateral_channels
        self.input_channels = input_channels
        self.lateral_layers = nn.ModuleList()
        self.smooth_layers = nn.ModuleList()
        
        if return_layers is None:
            # Feature maps from all FPN levels are returned
            self.return_layers = list(range(len(layers)-1))
        else:
            self.return_layers = return_layers
        self.min_returned_layer = min(self.return_layers)

        # Make lateral layers (for channel reduction) and smoothing layers
        for i in range(self.min_returned_layer, len(self.layers)):
            self.lateral_layers.append(nn.Conv2d(out_channels[i], self.lateral_channels, kernel_size=1, stride=1,
                                                 padding=0))

        # Optional: Add a 1x1 conv layer to reduce channels from multi-frame input
        # This can help the network learn better representations from temporal information
        '''
        if input_channels > 3:
            self.temporal_fusion = nn.Sequential(
                nn.Conv2d(input_channels, input_channels // 2, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(input_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(input_channels // 2, 3, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True)
            )
        else:
            self.temporal_fusion = None
        '''
        self.temporal_fusion = None

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # Optional temporal fusion step
        if self.temporal_fusion is not None:
            x = self.temporal_fusion(x)
        
        # Bottom-up pass, store all intermediary feature maps in list c
        c = []
        for m in self.layers:
            x = m(x)
            c.append(x)

        # Top-down pass
        p = [self.lateral_layers[-1](c[-1])]

        for i in range(len(c)-2, self.min_returned_layer-1, -1):
            temp = self._upsample_add(p[-1],  self.lateral_layers[i-self.min_returned_layer](c[i]))
            p.append(temp)

        # Reverse the order of tensors in p
        p = p[::-1]

        out_tensors = []
        for ndx, l in enumerate(self.return_layers):
            temp = p[l-self.min_returned_layer]
            out_tensors.append(temp)

        return out_tensors


class FPN(nn.Module):
    """Original FPN class for backward compatibility"""
    def __init__(self, layers, out_channels, lateral_channels, return_layers=None):
        super(FPN, self).__init__()
        self.fpn = MultiFrameFPN(layers, out_channels, lateral_channels, return_layers, input_channels=3)
    
    def forward(self, x):
        return self.fpn(x)


def create_multiframe_fpn(cfg_name='X', num_frames=3, batch_norm=True, lateral_channels=32, return_layers=None):
    """
    Create a multi-frame FPN network
    
    Args:
        cfg_name: Configuration name from cfg dictionary
        num_frames: Number of input frames
        batch_norm: Whether to use batch normalization  
        lateral_channels: Number of channels in lateral connections
        return_layers: Which layers to return features from
    
    Returns:
        MultiFrameFPN network
    """
    input_channels = 3 * num_frames
    layers, out_channels = make_multiframe_modules(cfg[cfg_name], batch_norm, input_channels)
    
    if return_layers is None:
        return_layers = [1, 3]  # Default return layers for ball and player detection
    
    fpn_net = MultiFrameFPN(layers, out_channels, lateral_channels, return_layers, input_channels)
    return fpn_net
