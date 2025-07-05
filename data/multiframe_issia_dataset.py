import torch
from PIL import Image
import numpy as np
import os

import data.multiframe_augmentation as augmentation
import data.issia_utils as issia_utils
from data.multiframe_augmentation import BALL_BBOX_SIZE, BALL_LABEL, PLAYER_LABEL

from torch.utils.data import Sampler


class MultiFrameIssiaDataset(torch.utils.data.Dataset):
    # Read multiple consecutive frames from the ISSIA dataset for Early Fusion
    def __init__(self, dataset_path, cameras, transform, only_ball_frames=False, 
                 num_frames=3, frame_interval=1, temporal_strategy='center'):
        """
        Args:
            dataset_path: Directory with all the images
            cameras: List of camera ids between 1 and 6
            transform: Optional transform to be applied on samples
            only_ball_frames: Whether to only use frames with ball annotations
            num_frames: Number of consecutive frames to use (should be odd for center strategy)
            frame_interval: Interval between selected frames
            temporal_strategy: 'center' - center frame is the target, 'last' - last frame is target
        """
        for camera_id in cameras:
            assert 1 <= camera_id <= 6, 'Unknown camera id: {}'.format(camera_id)
        
        assert temporal_strategy in ['center', 'last'], "Strategy must be 'center' or 'last'"
        if temporal_strategy == 'center':
            assert num_frames % 2 == 1, "For center strategy, num_frames must be odd"

        self.dataset_path = dataset_path
        self.cameras = cameras
        self.transform = transform
        self.only_ball_frames = only_ball_frames
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.temporal_strategy = temporal_strategy
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
            annotated_frames = [e for e in list(annotated_frames) if e > min_annotated_frame+50]
            
            # Filter frames to ensure we can get enough temporal context
            valid_frames = []
            for frame_idx in annotated_frames:
                if self._can_get_temporal_sequence(camera_id, frame_idx):
                    valid_frames.append(frame_idx)
            
            for frame_idx in valid_frames:
                # Verify if the target image file exists
                file_path = os.path.join(frames_path, str(frame_idx) + self.image_extension)
                if os.path.exists(file_path):
                    self.image_list.append((file_path, camera_id, frame_idx))

        self.n_images = len(self.image_list)
        self.ball_images_ndx = set(self.get_elems_with_ball())
        self.no_ball_images_ndx = set([ndx for ndx in range(self.n_images) if ndx not in self.ball_images_ndx])
        print('Multi-frame ISSIA CNR: {} frames with the ball'.format(len(self.ball_images_ndx)))
        print('Multi-frame ISSIA CNR: {} frames without the ball'.format(len(self.no_ball_images_ndx)))
        print('Using {} frames per sample with interval {}'.format(self.num_frames, self.frame_interval))

    def _can_get_temporal_sequence(self, camera_id, target_frame_idx):
        """Check if we can get a complete temporal sequence for the given target frame"""
        frames_path = os.path.join(self.dataset_path, 'unpacked', str(camera_id))
        
        if self.temporal_strategy == 'center':
            offset = (self.num_frames - 1) // 2
            start_frame = target_frame_idx - offset * self.frame_interval
            end_frame = target_frame_idx + offset * self.frame_interval
        else:  # 'last'
            start_frame = target_frame_idx - (self.num_frames - 1) * self.frame_interval
            end_frame = target_frame_idx
            
        # Check if all required frames exist
        for i in range(self.num_frames):
            frame_idx = start_frame + i * self.frame_interval
            file_path = os.path.join(frames_path, str(frame_idx) + self.image_extension)
            if not os.path.exists(file_path):
                return False
        return True

    def _get_temporal_sequence(self, camera_id, target_frame_idx):
        """Get temporal sequence of frames for the given target frame"""
        frames_path = os.path.join(self.dataset_path, 'unpacked', str(camera_id))
        
        if self.temporal_strategy == 'center':
            offset = (self.num_frames - 1) // 2
            start_frame = target_frame_idx - offset * self.frame_interval
        else:  # 'last'
            start_frame = target_frame_idx - (self.num_frames - 1) * self.frame_interval
            
        frame_indices = []
        image_paths = []
        
        for i in range(self.num_frames):
            frame_idx = start_frame + i * self.frame_interval
            frame_indices.append(frame_idx)
            file_path = os.path.join(frames_path, str(frame_idx) + self.image_extension)
            image_paths.append(file_path)
            
        return image_paths, frame_indices

    def __len__(self):
        return self.n_images

    def __getitem__(self, ndx):
        # Returns multi-frame concatenated image as a normalized tensor
        image_path, camera_id, target_image_ndx = self.image_list[ndx]
        
        # Get temporal sequence of images
        image_paths, frame_indices = self._get_temporal_sequence(camera_id, target_image_ndx)
        
        # Load all frames
        images = []
        for img_path in image_paths:
            img = Image.open(img_path)
            images.append(img)
        
        # Get annotations for the target frame only
        boxes, labels = self.get_annotations(camera_id, target_image_ndx)
        
        # Apply transformations (need to handle multi-frame case)
        images, boxes, labels = self.transform((images, boxes, labels))
        
        # Concatenate images along channel dimension
        # images should be a list of tensors, each with shape (C, H, W)
        if isinstance(images, list):
            multi_frame_image = torch.cat(images, dim=0)  # Shape: (C*num_frames, H, W)
        else:
            multi_frame_image = images  # Assume already concatenated by transform

        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        return multi_frame_image, boxes, labels

    def get_annotations(self, camera_id, image_ndx):
        # Same as original, get annotations for a single frame
        boxes = []
        labels = []

        # Add annotations for the ball position
        ball_pos = self.gt_annotations[camera_id].ball_pos[image_ndx]
        for (x, y) in ball_pos:
            x1 = x - BALL_BBOX_SIZE // 2
            x2 = x1 + BALL_BBOX_SIZE
            y1 = y - BALL_BBOX_SIZE // 2
            y2 = y1 + BALL_BBOX_SIZE
            boxes.append((x1, y1, x2, y2))
            labels.append(BALL_LABEL)

        # Add annotations for the player position
        for (player_id, player_height, player_width, player_x, player_y) in self.gt_annotations[camera_id].persons[image_ndx]:
            boxes.append((player_x, player_y, player_x + player_width, player_y + player_height))
            labels.append(PLAYER_LABEL)

        return np.array(boxes, dtype=float), np.array(labels, dtype=np.int64)

    def get_elems_with_ball(self):
        # Get indexes of images with ball ground truth
        ball_images_ndx = []
        for ndx, (_, camera_id, image_ndx) in enumerate(self.image_list):
            ball_pos = self.gt_annotations[camera_id].ball_pos[image_ndx]
            if len(ball_pos) > 0:
                ball_images_ndx.append(ndx)
        return ball_images_ndx


def create_multiframe_issia_dataset(dataset_path, cameras, mode, only_ball_frames=False, 
                                   num_frames=3, frame_interval=1, temporal_strategy='center', transform=None):
    # Get multi-frame ISSIA datasets for multiple cameras
    assert mode == 'train' or mode == 'val'
    assert os.path.exists(dataset_path), 'Cannot find dataset: ' + str(dataset_path)

    train_image_size = (720, 1280)
    val_image_size = (1080, 1920)
    
    if transform is None:
        if mode == 'train':
            transform = augmentation.MultiFrameTrainAugmentation(size=train_image_size, num_frames=num_frames)
        elif mode == 'val':
            transform = augmentation.MultiFrameNoAugmentation(size=val_image_size, num_frames=num_frames)

    dataset = MultiFrameIssiaDataset(dataset_path, cameras, transform=transform, only_ball_frames=only_ball_frames,
                                   num_frames=num_frames, frame_interval=frame_interval, 
                                   temporal_strategy=temporal_strategy, transform=transform)
    return dataset
