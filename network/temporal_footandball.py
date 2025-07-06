import torch
import torch.nn as nn

import network.fpn as fpn
import network.nms as nms
from data.augmentation import BALL_LABEL, PLAYER_LABEL, BALL_BBOX_SIZE
# NEW IMPORT
from network.temporal_fusion import TemporalFusionModule, MotionAwareModule


class TemporalFootAndBall(nn.Module):
    """
    MODIFICATION: Extended FootAndBall with temporal processing capabilities
    """
    def __init__(self, phase, base_network: nn.Module, player_regressor: nn.Module, 
                 player_classifier: nn.Module, ball_classifier: nn.Module, 
                 temporal_frames=3, use_motion_module=True,
                 max_player_detections=100, max_ball_detections=100, 
                 player_threshold=0.0, ball_threshold=0.0):
        super(TemporalFootAndBall, self).__init__()

        assert phase in ['train', 'eval', 'detect']

        self.phase = phase
        self.base_network = base_network
        self.ball_classifier = ball_classifier
        self.player_classifier = player_classifier
        self.player_regressor = player_regressor
        self.temporal_frames = temporal_frames
        self.use_motion_module = use_motion_module
        self.max_player_detections = max_player_detections
        self.max_ball_detections = max_ball_detections
        self.player_threshold = player_threshold
        self.ball_threshold = ball_threshold

        # Downsampling factors
        self.ball_downsampling_factor = 4
        self.player_downsampling_factor = 16
        self.ball_delta = 3
        self.player_delta = 3

        # NEW: Temporal fusion modules
        # Assuming base network outputs have 32 channels (lateral_channels)
        self.temporal_fusion_ball = TemporalFusionModule(32, temporal_frames)
        self.temporal_fusion_player = TemporalFusionModule(32, temporal_frames)
        
        # NEW: Motion-aware module for ball detection
        if use_motion_module:
            self.motion_module_ball = MotionAwareModule(32, temporal_frames)

        self.softmax = nn.Softmax(dim=1)
        self.nms_kernel_size = (3, 3)
        self.nms = nms.NonMaximaSuppression2d(self.nms_kernel_size)

    def forward(self, x):
        """
        MODIFICATION: Process temporal input
        x: Tensor of shape (B, T*C, H, W) where T is temporal frames
        """
        height, width = x.shape[2], x.shape[3]
        batch_size = x.shape[0]
        
        # Split temporal frames and process through base network
        # Reshape to process each frame separately through base network
        x_reshaped = x.view(batch_size, self.temporal_frames, 3, height, width)
        
        # Process each frame through base network
        frame_features = []
        for t in range(self.temporal_frames):
            frame = x_reshaped[:, t]
            features = self.base_network(frame)
            frame_features.append(features)
        
        # Concatenate temporal features for each scale
        # features[0] is ball scale, features[1] is player scale
        ball_features_temporal = []
        player_features_temporal = []
        
        for t in range(self.temporal_frames):
            ball_features_temporal.append(frame_features[t][0])
            player_features_temporal.append(frame_features[t][1])
        
        # Stack along channel dimension
        ball_features_concat = torch.cat(ball_features_temporal, dim=1)
        player_features_concat = torch.cat(player_features_temporal, dim=1)
        
        # Apply temporal fusion
        ball_features_fused = self.temporal_fusion_ball(ball_features_concat)
        player_features_fused = self.temporal_fusion_player(player_features_concat)
        
        # Apply motion module for ball if enabled
        if self.use_motion_module:
            ball_features_motion = self.motion_module_ball(ball_features_concat)
            # Combine temporal and motion features
            ball_features_fused = ball_features_fused + ball_features_motion
        
        # Apply classifiers and regressors
        ball_feature_map = self.ball_classifier(ball_features_fused)
        player_feature_map = self.player_classifier(player_features_fused)
        player_bbox = self.player_regressor(player_features_fused)

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
    
    # Rest of the methods remain the same as original FootAndBall
    def detect_from_map(self, confidence_map, downscale_factor, max_detections, bbox_map=None):
        # Same as original
        confidence_map = self.nms(confidence_map)[:, 1]
        batch_size, h, w = confidence_map.shape[0], confidence_map.shape[1], confidence_map.shape[2]
        confidence_map = confidence_map.view(batch_size, -1)

        values, indices = torch.sort(confidence_map, dim=-1, descending=True)
        if max_detections < indices.shape[1]:
            indices = indices[:, :max_detections]

        xc = indices % w
        xc = xc.float() * downscale_factor + (downscale_factor - 1.) / 2.

        yc = torch.div(indices, w, rounding_mode='trunc')
        yc = yc.float() * downscale_factor + (downscale_factor - 1.) / 2.

        if bbox_map is not None:
            bbox_map = bbox_map.view(batch_size, 4, -1)
            bbox_map[:, 0] *= w * downscale_factor
            bbox_map[:, 2] *= w * downscale_factor
            bbox_map[:, 1] *= h * downscale_factor
            bbox_map[:, 3] *= h * downscale_factor
        else:
            batch_size, h, w = confidence_map.shape[0], confidence_map.shape[-2], confidence_map.shape[-1]
            bbox_map = torch.zeros((batch_size, 4, h * w), dtype=torch.float).to(confidence_map.device)
            bbox_map[:, [2, 3]] = BALL_BBOX_SIZE

        detections = torch.zeros((batch_size, max_detections, 5), dtype=float).to(confidence_map.device)

        for n in range(batch_size):
            temp = bbox_map[n, :, indices[n]]
            bx = xc[n] + temp[0]
            by = yc[n] + temp[1]

            detections[n, :, 0] = bx - 0.5 * temp[2]
            detections[n, :, 2] = bx + 0.5 * temp[2]
            detections[n, :, 1] = by - 0.5 * temp[3]
            detections[n, :, 3] = by + 0.5 * temp[3]
            detections[n, :, 4] = values[n, :max_detections]

        return detections

    def detect(self, player_feature_map, player_bbox, ball_feature_map):
        # Same as original
        player_detections = self.detect_from_map(player_feature_map, self.player_downsampling_factor,
                                                 self.max_player_detections, player_bbox)

        ball_detections = self.detect_from_map(ball_feature_map, self.ball_downsampling_factor,
                                               self.max_ball_detections)

        output = []
        for player_det, ball_det in zip(player_detections, ball_detections):
            player_det = player_det[player_det[..., 4] >= self.player_threshold]
            player_boxes = player_det[..., 0:4]
            player_scores = player_det[..., 4]
            player_labels = torch.tensor([PLAYER_LABEL] * len(player_det), dtype=torch.int64)
            ball_det = ball_det[ball_det[..., 4] >= self.ball_threshold]
            ball_boxes = ball_det[..., 0:4]
            ball_scores = ball_det[..., 4]
            ball_labels = torch.tensor([BALL_LABEL] * len(ball_det), dtype=torch.int64)

            boxes = torch.cat([player_boxes, ball_boxes], dim=0)
            scores = torch.cat([player_scores, ball_scores], dim=0)
            labels = torch.cat([player_labels, ball_labels], dim=0)

            temp = {'boxes': boxes, 'labels': labels, 'scores': scores}
            output.append(temp)

        return output

    def groundtruth_maps(self, boxes, labels, img_shape):
        # Same as original
        from network.footandball import create_groundtruth_maps
        player_loc_t, player_conf_t, ball_conf_t = create_groundtruth_maps(
            boxes, labels, img_shape,
            self.player_downsampling_factor,
            self.ball_downsampling_factor,
            self.player_delta, self.ball_delta)

        return player_loc_t, player_conf_t, ball_conf_t

    def print_summary(self, show_architecture=True):
        # Extended to show temporal modules
        if show_architecture:
            print('Base network:')
            print(self.base_network)
            print('\nTemporal Fusion (Ball):')
            print(self.temporal_fusion_ball)
            print('\nTemporal Fusion (Player):')
            print(self.temporal_fusion_player)
            if self.use_motion_module:
                print('\nMotion Module (Ball):')
                print(self.motion_module_ball)
            if self.ball_classifier is not None:
                print('\nBall classifier:')
                print(self.ball_classifier)
            if self.player_classifier is not None:
                print('\nPlayer classifier:')
                print(self.player_classifier)

        from network.footandball import count_parameters
        ap, tp = count_parameters(self.base_network)
        print('\nBase network parameters (all/trainable): {}/{}'.format(ap, tp))

        ap, tp = count_parameters(self.temporal_fusion_ball)
        print('Temporal Fusion Ball parameters (all/trainable): {}/{}'.format(ap, tp))

        ap, tp = count_parameters(self.temporal_fusion_player)
        print('Temporal Fusion Player parameters (all/trainable): {}/{}'.format(ap, tp))

        if self.use_motion_module:
            ap, tp = count_parameters(self.motion_module_ball)
            print('Motion Module parameters (all/trainable): {}/{}'.format(ap, tp))

        if self.ball_classifier is not None:
            ap, tp = count_parameters(self.ball_classifier)
            print('Ball classifier parameters (all/trainable): {}/{}'.format(ap, tp))

        if self.player_classifier is not None:
            ap, tp = count_parameters(self.player_classifier)
            print('Player classifier parameters (all/trainable): {}/{}'.format(ap, tp))

        if self.player_regressor is not None:
            ap, tp = count_parameters(self.player_regressor)
            print('Player regressor parameters (all/trainable): {}/{}'.format(ap, tp))

        ap, tp = count_parameters(self)
        print('Total (all/trainable): {} / {}'.format(ap, tp))
        print('')


def build_temporal_footandball_detector(phase='train', temporal_frames=3, use_motion_module=True,
                                       max_player_detections=100, max_ball_detections=100,
                                       player_threshold=0.0, ball_threshold=0.0):
    """
    MODIFICATION: Build temporal FootAndBall detector
    """
    assert phase in ['train', 'test', 'detect']

    layers, out_channels = fpn.make_modules(fpn.cfg['X'], batch_norm=True)
    lateral_channels = 32
    i_channels = 32

    base_net = fpn.FPN(layers, out_channels=out_channels, lateral_channels=lateral_channels, return_layers=[1, 3])
    
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
        temporal_frames=temporal_frames,
        use_motion_module=use_motion_module,
        ball_threshold=ball_threshold,
        player_threshold=player_threshold, 
        max_ball_detections=max_ball_detections,
        max_player_detections=max_player_detections
    )
    
    return detector
