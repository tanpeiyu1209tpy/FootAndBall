# FootAndBall: Integrated Player and Ball Detector - Multi-frame Version
# 支持多帧Early Fusion的足球和球员检测器

import torch
import torch.nn as nn

import network.multiframe_fpn as fpn
import network.nms as nms
from data.multiframe_augmentation import BALL_LABEL, PLAYER_LABEL, BALL_BBOX_SIZE

# Import multi-frame FPN (assuming it's in the same fpn module)
from network.multiframe_fpn import MultiFrameFPN, create_multiframe_fpn


class MultiFrameFootAndBall(nn.Module):
    def __init__(self, phase, base_network: nn.Module, player_regressor: nn.Module, player_classifier: nn.Module,
                 ball_classifier: nn.Module, num_frames=3, max_player_detections=100, max_ball_detections=100, 
                 player_threshold=0.0, ball_threshold=0.0):
        """
        Multi-frame FootAndBall detector with Early Fusion
        
        Args:
            phase: 'train', 'eval', or 'detect'
            base_network: Backbone network (should accept multi-frame input)
            player_regressor: Player bounding box regressor
            player_classifier: Player classifier
            ball_classifier: Ball classifier  
            num_frames: Number of input frames
            max_player_detections: Max number of player detections in detect mode
            max_ball_detections: Max number of ball detections in detect mode
            player_threshold: Confidence threshold for player detection
            ball_threshold: Confidence threshold for ball detection
        """
        super(MultiFrameFootAndBall, self).__init__()

        assert phase in ['train', 'eval', 'detect']

        self.phase = phase
        self.num_frames = num_frames
        self.base_network = base_network
        self.ball_classifier = ball_classifier
        self.player_classifier = player_classifier
        self.player_regressor = player_regressor
        self.max_player_detections = max_player_detections
        self.max_ball_detections = max_ball_detections
        self.player_threshold = player_threshold
        self.ball_threshold = ball_threshold

        # Downsampling factors remain the same
        self.ball_downsampling_factor = 4
        self.player_downsampling_factor = 16
        self.ball_delta = 3
        self.player_delta = 3

        self.softmax = nn.Softmax(dim=1)
        self.nms_kernel_size = (3, 3)
        self.nms = nms.NonMaximaSuppression2d(self.nms_kernel_size)

    def detect_from_map(self, confidence_map, downscale_factor, max_detections, bbox_map=None):
        """Same as original FootAndBall"""
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

            detections[n, :, 0] = bx - 0.5 * temp[2]  # x1
            detections[n, :, 2] = bx + 0.5 * temp[2]  # x2
            detections[n, :, 1] = by - 0.5 * temp[3]  # y1
            detections[n, :, 3] = by + 0.5 * temp[3]  # y2
            detections[n, :, 4] = values[n, :max_detections]

        return detections

    def detect(self, player_feature_map, player_bbox, ball_feature_map):
        """Same as original FootAndBall"""
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
        """Same as original FootAndBall - ground truth is still based on target frame"""
        from network.footandball import create_groundtruth_maps
        
        player_loc_t, player_conf_t, ball_conf_t = create_groundtruth_maps(
            boxes, labels, img_shape,
            self.player_downsampling_factor,
            self.ball_downsampling_factor,
            self.player_delta, self.ball_delta
        )

        return player_loc_t, player_conf_t, ball_conf_t

    def forward(self, x):
        """
        Forward pass with multi-frame input
        
        Args:
            x: Input tensor of shape (batch_size, channels*num_frames, height, width)
        """
        height, width = x.shape[2], x.shape[3]
        
        # Verify input has correct number of channels
        expected_channels = 3 * self.num_frames
        assert x.shape[1] == expected_channels, f"Expected {expected_channels} channels, got {x.shape[1]}"

        # Pass through base network
        x = self.base_network(x)
        
        # Same assertions as original
        assert len(x) == 2
        assert x[0].shape[0] == x[1].shape[0]
        assert x[0].shape[1] == x[1].shape[1]
        assert x[0].shape[2] == height // self.ball_downsampling_factor
        assert x[0].shape[3] == width // self.ball_downsampling_factor
        assert x[1].shape[2] == height // self.player_downsampling_factor
        assert x[1].shape[3] == width // self.player_downsampling_factor

        ball_feature_map = self.ball_classifier(x[0])
        player_feature_map = self.player_classifier(x[1])
        player_bbox = self.player_regressor(x[1])

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

    def print_summary(self, show_architecture=True):
        """Print network statistics"""
        if show_architecture:
            print('Multi-frame Base network (num_frames={}):'.format(self.num_frames))
            print(self.base_network)
            if self.ball_classifier is not None:
                print('Ball classifier:')
                print(self.ball_classifier)
            if self.player_classifier is not None:
                print('Player classifier:')
                print(self.player_classifier)

        from network.footandball import count_parameters
        
        ap, tp = count_parameters(self.base_network)
        print('Base network parameters (all/trainable): {}/{}'.format(ap, tp))

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


def build_multiframe_footandball_detector(phase='train', num_frames=3, max_player_detections=100, 
                                         max_ball_detections=100, player_threshold=0.0, ball_threshold=0.0):
    """
    Build multi-frame FootAndBall detector
    """
    assert phase in ['train', 'eval', 'detect']

    # Create multi-frame FPN backbone
    lateral_channels = 32
    i_channels = 32
    
    base_net = create_multiframe_fpn(
        cfg_name='X', 
        num_frames=num_frames, 
        batch_norm=True, 
        lateral_channels=lateral_channels, 
        return_layers=[1, 3]
    )
    
    # Classification and regression heads remain the same
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
    
    detector = MultiFrameFootAndBall(
        phase, base_net, 
        player_regressor=player_regressor, 
        player_classifier=player_classifier,
        ball_classifier=ball_classifier, 
        num_frames=num_frames,
        ball_threshold=ball_threshold,
        player_threshold=player_threshold, 
        max_ball_detections=max_ball_detections,
        max_player_detections=max_player_detections
    )
    
    return detector


def multiframe_model_factory(model_name, phase, num_frames=3, max_player_detections=100, 
                            max_ball_detections=100, player_threshold=0.0, ball_threshold=0.0):
    """Factory function for creating multi-frame models"""
    if model_name == 'mf_fb1':
        model_fn = build_multiframe_footandball_detector
    else:
        print('Multi-frame model not implemented: {}'.format(model_name))
        raise NotImplementedError

    return model_fn(phase, num_frames=num_frames, ball_threshold=ball_threshold, 
                   player_threshold=player_threshold, max_ball_detections=max_ball_detections, 
                   max_player_detections=max_player_detections)
