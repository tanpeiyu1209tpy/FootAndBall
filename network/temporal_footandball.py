# Temporal FootAndBall: Multi-frame Feature Aggregation
# Extension of the original FootAndBall detector with temporal information

import torch
import torch.nn as nn
import torch.nn.functional as F

import network.fpn as fpn
import network.nms as nms
from network.footandball import FootAndBall, create_groundtruth_maps, count_parameters
from data.augmentation import BALL_LABEL, PLAYER_LABEL, BALL_BBOX_SIZE


class TemporalFeatureAggregator(nn.Module):
    """Aggregates features from multiple temporal frames"""
    
    def __init__(self, in_channels, n_frames=3, aggregation_type='attention'):
        super().__init__()
        self.n_frames = n_frames
        self.aggregation_type = aggregation_type
        
        if aggregation_type == 'attention':
            # Attention-based aggregation
            self.temporal_attention = nn.Sequential(
                nn.Conv2d(in_channels * n_frames, in_channels, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, n_frames, kernel_size=1),
                nn.Softmax(dim=1)
            )
            # Motion variance branch
            self.motion_branch = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 2, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        elif aggregation_type == 'conv3d':
            # 3D convolution aggregation
            self.conv3d = nn.Conv3d(in_channels, in_channels, 
                                   kernel_size=(n_frames, 3, 3), 
                                   padding=(0, 1, 1))
        else:  # 'mean' or 'max'
            pass
    
    def forward(self, features):
        """
        Args:
            features: List of feature maps from consecutive frames
                     Each element has shape (B, C, H, W)
        Returns:
            Aggregated features (B, C, H, W)
        """
        B, C, H, W = features[0].shape
        
        if self.aggregation_type == 'attention':
            # Stack features
            stacked = torch.stack(features, dim=1)  # (B, T, C, H, W)
            
            # Compute temporal variance (motion indicator)
            variance = torch.var(stacked, dim=1)  # (B, C, H, W)
            motion_weight = self.motion_branch(variance)  # (B, 1, H, W)
            
            # Concatenate all frames
            concat_features = torch.cat(features, dim=1)  # (B, T*C, H, W)
            
            # Compute attention weights
            attention = self.temporal_attention(concat_features)  # (B, T, H, W)
            
            # Apply attention
            weighted_features = []
            for i in range(self.n_frames):
                weight = attention[:, i:i+1, :, :]  # (B, 1, H, W)
                weighted = features[i] * weight
                weighted_features.append(weighted)
            
            # Combine with motion weighting
            aggregated = sum(weighted_features)
            aggregated = aggregated * motion_weight + features[self.n_frames//2] * (1 - motion_weight)
            
        elif self.aggregation_type == 'conv3d':
            # Stack and apply 3D convolution
            stacked = torch.stack(features, dim=2)  # (B, C, T, H, W)
            aggregated = self.conv3d(stacked).squeeze(2)  # (B, C, H, W)
            
        elif self.aggregation_type == 'mean':
            aggregated = torch.mean(torch.stack(features, dim=0), dim=0)
            
        elif self.aggregation_type == 'max':
            aggregated = torch.max(torch.stack(features, dim=0), dim=0)[0]
            
        else:
            raise ValueError(f"Unknown aggregation type: {self.aggregation_type}")
        
        return aggregated


class TemporalFootAndBall(FootAndBall):
    """FootAndBall detector with temporal feature aggregation"""
    
    def __init__(self, phase, base_network, player_regressor, player_classifier,
                 ball_classifier, n_frames=3, aggregation_type='attention',
                 max_player_detections=100, max_ball_detections=100,
                 player_threshold=0.0, ball_threshold=0.0):
        super().__init__(phase, base_network, player_regressor, player_classifier,
                        ball_classifier, max_player_detections, max_ball_detections,
                        player_threshold, ball_threshold)
        
        self.n_frames = n_frames
        self.aggregation_type = aggregation_type
        
        # Get number of channels from FPN
        lateral_channels = 32  # This should match your FPN configuration
        
        # Create temporal aggregators
        self.ball_temporal_agg = TemporalFeatureAggregator(
            lateral_channels, n_frames, aggregation_type
        )
        self.player_temporal_agg = TemporalFeatureAggregator(
            lateral_channels, n_frames, aggregation_type
        )
    
    def forward(self, x):
        """
        Args:
            x: Either single frame tensor (B, C, H, W) for compatibility
               or multi-frame tensor (B, T, C, H, W) for temporal processing
        """
        if x.dim() == 4:
            # Single frame input - replicate for compatibility
            x = x.unsqueeze(1).repeat(1, self.n_frames, 1, 1, 1)
        
        assert x.dim() == 5, f"Expected 5D input, got {x.dim()}D"
        B, T, C, H, W = x.shape
        assert T == self.n_frames, f"Expected {self.n_frames} frames, got {T}"
        
        # Process each frame through base network
        frame_features = []
        for t in range(T):
            frame = x[:, t]  # (B, C, H, W)
            features = self.base_network(frame)
            frame_features.append(features)
        
        # Separate ball and player features
        ball_features = [f[0] for f in frame_features]  # Higher resolution
        player_features = [f[1] for f in frame_features]  # Lower resolution
        
        # Aggregate temporal features
        ball_features_agg = self.ball_temporal_agg(ball_features)
        player_features_agg = self.player_temporal_agg(player_features)
        
        # Continue with detection heads
        ball_feature_map = self.ball_classifier(ball_features_agg)
        player_feature_map = self.player_classifier(player_features_agg)
        player_bbox = self.player_regressor(player_features_agg)
        
        # Rest of the forward pass remains the same
        if self.phase == 'eval' or self.phase == 'detect':
            player_feature_map = self.softmax(player_feature_map)
            ball_feature_map = self.softmax(ball_feature_map)
        
        if self.phase == 'train' or self.phase == 'eval':
            ball_feature_map = ball_feature_map.permute(0, 2, 3, 1).contiguous()
            player_feature_map = player_feature_map.permute(0, 2, 3, 1).contiguous()
            player_bbox = player_bbox.permute(0, 2, 3, 1).contiguous()
            output = (player_bbox, player_feature_map, ball_feature_map)
        elif self.phase == 'detect':
            output = self.detect(player_feature_map, player_bbox, ball_feature_map)
        
        return output


def build_temporal_footandball_detector(phase='train', n_frames=3, 
                                      aggregation_type='attention',
                                      max_player_detections=100, 
                                      max_ball_detections=100,
                                      player_threshold=0.0, 
                                      ball_threshold=0.0):
    """Build temporal FootAndBall detector"""
    
    assert phase in ['train', 'test', 'detect']
    
    # Build base components (same as original)
    layers, out_channels = fpn.make_modules(fpn.cfg['X'], batch_norm=True)
    lateral_channels = 32
    i_channels = 32
    
    base_net = fpn.FPN(layers, out_channels=out_channels, 
                      lateral_channels=lateral_channels, 
                      return_layers=[1, 3])
    
    ball_classifier = nn.Sequential(
        nn.Conv2d(lateral_channels, out_channels=i_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(i_channels, out_channels=2, kernel_size=3, padding=1)
    )
    
    player_classifier = nn.Sequential(
        nn.Conv2d(lateral_channels, out_channels=i_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(i_channels, out_channels=2, kernel_size=3, padding=1)
    )
    
    player_regressor = nn.Sequential(
        nn.Conv2d(lateral_channels, out_channels=i_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(i_channels, out_channels=4, kernel_size=3, padding=1)
    )
    
    detector = TemporalFootAndBall(
        phase, base_net, 
        player_regressor=player_regressor,
        player_classifier=player_classifier,
        ball_classifier=ball_classifier,
        n_frames=n_frames,
        aggregation_type=aggregation_type,
        ball_threshold=ball_threshold,
        player_threshold=player_threshold,
        max_ball_detections=max_ball_detections,
        max_player_detections=max_player_detections
    )
    
    return detector


# Example usage
if __name__ == '__main__':
    # Create temporal detector
    model = build_temporal_footandball_detector(
        phase='train',
        n_frames=3,
        aggregation_type='attention'
    )
    
    # Test with multi-frame input
    batch_size = 2
    n_frames = 3
    height, width = 720, 1280
    
    # Simulate multi-frame input
    x = torch.randn(batch_size, n_frames, 3, height, width)
    
    # Forward pass
    output = model(x)
    
    print(f"Model created successfully!")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
