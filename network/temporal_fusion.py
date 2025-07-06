import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalFusionModule(nn.Module):
    """
    MODIFICATION: New module to fuse temporal features from multiple frames
    Uses attention mechanism to weight different temporal frames
    """
    def __init__(self, in_channels, temporal_frames=3):
        super(TemporalFusionModule, self).__init__()
        self.temporal_frames = temporal_frames
        self.in_channels = in_channels
        
        # Temporal attention mechanism
        self.temporal_attention = nn.Sequential(
            nn.Conv2d(in_channels * temporal_frames, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, temporal_frames, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # Feature refinement after fusion
        self.refinement = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        x: Tensor of shape (B, T*C, H, W) where T is temporal frames, C is channels
        """
        batch_size, tc, h, w = x.shape
        assert tc == self.temporal_frames * self.in_channels
        
        # Reshape to separate temporal dimension: (B, T, C, H, W)
        x_reshaped = x.view(batch_size, self.temporal_frames, self.in_channels, h, w)
        
        # Compute temporal attention weights: (B, T, H, W)
        attention_weights = self.temporal_attention(x)
        
        # Apply attention weights to each temporal frame
        weighted_features = []
        for t in range(self.temporal_frames):
            weight = attention_weights[:, t:t+1, :, :]  # (B, 1, H, W)
            feature = x_reshaped[:, t, :, :, :]  # (B, C, H, W)
            weighted_feature = feature * weight
            weighted_features.append(weighted_feature)
        
        # Sum weighted features
        fused_features = torch.stack(weighted_features, dim=0).sum(dim=0)
        
        # Refine fused features
        output = self.refinement(fused_features)
        
        return output


class MotionAwareModule(nn.Module):
    """
    MODIFICATION: Module to explicitly capture motion between frames
    Useful for ball detection as balls move distinctively
    """
    def __init__(self, in_channels, temporal_frames=3):
        super(MotionAwareModule, self).__init__()
        self.temporal_frames = temporal_frames
        self.in_channels = in_channels
        
        # Motion extraction using frame differences
        self.motion_conv = nn.Sequential(
            nn.Conv2d((temporal_frames - 1) * in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # Combine motion and appearance
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + in_channels // 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        x: Tensor of shape (B, T*C, H, W)
        """
        batch_size, tc, h, w = x.shape
        
        # Reshape to separate temporal dimension
        x_reshaped = x.view(batch_size, self.temporal_frames, self.in_channels, h, w)
        
        # Compute frame differences for motion
        motion_features = []
        for t in range(1, self.temporal_frames):
            diff = x_reshaped[:, t] - x_reshaped[:, t-1]
            motion_features.append(diff)
        
        # Concatenate motion features
        motion_concat = torch.cat(motion_features, dim=1)  # (B, (T-1)*C, H, W)
        
        # Process motion features
        motion_processed = self.motion_conv(motion_concat)
        
        # Use current frame as appearance reference
        current_frame = x_reshaped[:, -1]  # Last frame (B, C, H, W)
        
        # Combine motion and appearance
        combined = torch.cat([current_frame, motion_processed], dim=1)
        output = self.fusion(combined)
        
        return output
