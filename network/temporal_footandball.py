# Modified data/issia_dataset.py with temporal support

import torch
from PIL import Image
import numpy as np
import os

import data.augmentation as augmentation
import data.issia_utils as issia_utils
from data.augmentation import BALL_BBOX_SIZE, BALL_LABEL, PLAYER_LABEL

from torch.utils.data import Sampler


class TemporalIssiaDataset(torch.utils.data.Dataset):
    """
    MODIFICATION: Extended IssiaDataset to return multiple consecutive frames for temporal processing
    """
    def __init__(self, dataset_path, cameras, transform, only_ball_frames=False, temporal_frames=3):
        """
        Args:
            root_dir: Directory with all the images
            camera_id: Camera id between 1 and 6
            transform: Optional transform to be applied on a sample
            temporal_frames: Number of consecutive frames to return (default: 3)
        """
        for camera_id in cameras:
            assert 1 <= camera_id <= 6, 'Unknown camera id: {}'.format(camera_id)

        self.dataset_path = dataset_path
        self.cameras = cameras
        self.transform = transform
        self.only_ball_frames = only_ball_frames
        self.temporal_frames = temporal_frames  # NEW: Number of frames to use
        self.image_extension = '.jpg'
        # Dictionary with ground truth annotations per camera
        self.gt_annotations = {}
        # list of images as tuples (image_path, camera_id, image index)
        self.image_list = []

        unpack_path = os.path.join(dataset_path, 'unpacked')
        if not os.path.exists(unpack_path):
            os.mkdir(unpack_path)

        for camera_id in cameras:
            # Extract frames from the sequence if needed
            frames_path = os.path.join(dataset_path, 'unpacked', str(camera_id))
            self.frames_path = frames_path
            if not os.path.exists(frames_path):
                os.mkdir(frames_path)
                issia_utils.extract_frames(dataset_path, camera_id, frames_path)

            # Read ground truth data for the sequence
            self.gt_annotations[camera_id] = issia_utils.read_issia_ground_truth(camera_id, dataset_path)

            # Create a list with ids of all images with any annotation
            if self.only_ball_frames:
                annotated_frames = set(self.gt_annotations[camera_id].ball_pos)
            else:
                annotated_frames = set(self.gt_annotations[camera_id].ball_pos) and set(
                    self.gt_annotations[camera_id].persons)

            min_annotated_frame = min(annotated_frames)
            # Skip the first 50 annotated frames - as they may contain wrong annotations
            # MODIFICATION: Also ensure we have enough frames for temporal context
            annotated_frames = [e for e in list(annotated_frames) 
                              if e > min_annotated_frame + 50 + self.temporal_frames]

            for e in annotated_frames:
                # Verify if the image file exists and previous temporal frames exist
                all_exist = True
                for t in range(self.temporal_frames):
                    file_path = os.path.join(frames_path, str(e - t) + self.image_extension)
                    if not os.path.exists(file_path):
                        all_exist = False
                        break
                
                if all_exist:
                    file_path = os.path.join(frames_path, str(e) + self.image_extension)
                    self.image_list.append((file_path, camera_id, e))

        self.n_images = len(self.image_list)
        self.ball_images_ndx = set(self.get_elems_with_ball())
        self.no_ball_images_ndx = set([ndx for ndx in range(self.n_images) if ndx not in self.ball_images_ndx])
        print('Temporal ISSIA CNR: {} frames with the ball'.format(len(self.ball_images_ndx)))
        print('Temporal ISSIA CNR: {} frames without the ball'.format(len(self.no_ball_images_ndx)))
        print('Using {} temporal frames'.format(self.temporal_frames))

    def __len__(self):
        return self.n_images

    def __getitem__(self, ndx):
        # MODIFICATION: Returns multiple frames as a temporal sequence
        image_path, camera_id, image_ndx = self.image_list[ndx]
        
        # Load temporal frames (current + previous frames)
        temporal_images = []
        for t in range(self.temporal_frames):
            frame_idx = image_ndx - t
            frame_path = os.path.join(os.path.dirname(image_path), str(frame_idx) + self.image_extension)
            image = Image.open(frame_path)
            temporal_images.append(image)
        
        # Reverse to have chronological order
        temporal_images = temporal_images[::-1]
        
        # Get annotations only for the current (last) frame
        boxes, labels = self.get_annotations(camera_id, image_ndx)
        
        # Apply transform to all frames
        if self.transform:
            # For temporal sequences, we need consistent augmentation across frames
            # Store the first transform parameters and apply to all frames
            transformed_images = []
            for i, img in enumerate(temporal_images):
                if i == 0:
                    # First frame - apply full transform and save parameters
                    transformed_img, boxes, labels = self.transform((img, boxes, labels))
                    transformed_images.append(transformed_img)
                else:
                    # Subsequent frames - apply same spatial transform but allow different color augmentation
                    # This is a simplified approach - in production you'd want more sophisticated handling
                    transformed_img = augmentation.image2tensor(img)
                    transformed_images.append(transformed_img)
        
        # Stack temporal frames along channel dimension: (T*C, H, W)
        temporal_tensor = torch.cat(transformed_images, dim=0)
        
        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.int64)
        return temporal_tensor, boxes, labels

    def get_annotations(self, camera_id, image_ndx):
        # Same as original
        boxes = []
        labels = []

        ball_pos = self.gt_annotations[camera_id].ball_pos[image_ndx]
        for (x, y) in ball_pos:
            x1 = x - BALL_BBOX_SIZE // 2
            x2 = x1 + BALL_BBOX_SIZE
            y1 = y - BALL_BBOX_SIZE // 2
            y2 = y1 + BALL_BBOX_SIZE
            boxes.append((x1, y1, x2, y2))
            labels.append(BALL_LABEL)

        for (player_id, player_height, player_width, player_x, player_y) in self.gt_annotations[camera_id].persons[image_ndx]:
            boxes.append((player_x, player_y, player_x + player_width, player_y + player_height))
            labels.append(PLAYER_LABEL)

        return np.array(boxes, dtype=float), np.array(labels, dtype=np.int64)

    def get_elems_with_ball(self):
        # Same as original
        ball_images_ndx = []
        for ndx, (_, camera_id, image_ndx) in enumerate(self.image_list):
            ball_pos = self.gt_annotations[camera_id].ball_pos[image_ndx]
            if len(ball_pos) > 0:
                ball_images_ndx.append(ndx)

        return ball_images_ndx


def create_temporal_issia_dataset(dataset_path, cameras, mode, only_ball_frames=False, temporal_frames=3):
    """
    MODIFICATION: Create temporal ISSIA dataset
    """
    assert mode == 'train' or mode == 'val'
    assert os.path.exists(dataset_path), 'Cannot find dataset: ' + str(dataset_path)

    train_image_size = (720, 1280)
    val_image_size = (1080, 1920)
    if mode == 'train':
        transform = augmentation.TrainAugmentation(size=train_image_size)
    elif mode == 'val':
        transform = augmentation.NoAugmentation(size=val_image_size)

    dataset = TemporalIssiaDataset(dataset_path, cameras, transform, 
                                   only_ball_frames=only_ball_frames, 
                                   temporal_frames=temporal_frames)
    return dataset


# ============================================================================
# Modified network/temporal_fusion.py - NEW FILE
# ============================================================================

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


# ============================================================================
# Modified network/footandball.py with temporal support
# ============================================================================

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
