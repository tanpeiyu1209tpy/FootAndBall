# Temporal ISSIA Dataset for multi-frame input

import torch
from PIL import Image
import numpy as np
import os
from collections import deque

import data.augmentation as augmentation
import data.issia_utils as issia_utils
from data.augmentation import BALL_BBOX_SIZE, BALL_LABEL, PLAYER_LABEL
from data.issia_dataset import IssiaDataset


class TemporalIssiaDataset(IssiaDataset):
    """ISSIA dataset that returns multiple consecutive frames"""
    
    def __init__(self, dataset_path, cameras, transform, n_frames=3, 
                 frame_interval=1, only_ball_frames=False):
        """
        Args:
            n_frames: Number of consecutive frames to return
            frame_interval: Interval between frames (1 = consecutive, 2 = every other frame, etc.)
        """
        super().__init__(dataset_path, cameras, transform, only_ball_frames)
        
        self.n_frames = n_frames
        self.frame_interval = frame_interval
        self.temporal_radius = (n_frames - 1) * frame_interval // 2
        
        # Filter out frames that don't have enough temporal context
        self.valid_indices = self._get_valid_temporal_indices()
    
    def _get_valid_temporal_indices(self):
        """Get indices that have enough temporal context"""
        valid_indices = []
        
        for idx in range(len(self.image_list)):
            if self._has_temporal_context(idx):
                valid_indices.append(idx)
        
        return valid_indices
    
    def _has_temporal_context(self, center_idx):
        """Check if an index has enough temporal context"""
        _, camera_id, frame_idx = self.image_list[center_idx]
        
        # Check if we can get all required frames
        for offset in range(-self.temporal_radius, self.temporal_radius + 1, self.frame_interval):
            target_frame = frame_idx + offset
            frame_path = os.path.join(self.frames_path, str(target_frame) + self.image_extension)
            
            if not os.path.exists(frame_path):
                return False
        
        return True
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """Returns multiple frames and annotations for the center frame"""
        
        # Get the actual index from valid indices
        actual_idx = self.valid_indices[idx]
        _, camera_id, center_frame_idx = self.image_list[actual_idx]
        
        # Load multiple frames
        frames = []
        for offset in range(-self.temporal_radius, self.temporal_radius + 1, self.frame_interval):
            frame_idx = center_frame_idx + offset
            frame_path = os.path.join(self.frames_path, str(frame_idx) + self.image_extension)
            image = Image.open(frame_path)
            frames.append(image)
        
        # Get annotations for the center frame
        boxes, labels = self.get_annotations(camera_id, center_frame_idx)
        
        # Apply augmentation to all frames consistently
        if hasattr(self.transform, 'augment_temporal'):
            # Custom temporal augmentation
            frames, boxes, labels = self.transform.augment_temporal(frames, boxes, labels)
        else:
            # Apply same spatial augmentation to all frames
            # First, apply spatial augmentations
            if hasattr(self.transform.augment, 'transforms'):
                for t in self.transform.augment.transforms[:-1]:  # Skip ToTensorAndNormalize
                    if hasattr(t, '__call__'):
                        # Apply same random parameters to all frames
                        if hasattr(t, 'get_params'):
                            # For RandomAffine, RandomCrop etc.
                            h, w = frames[0].height, frames[0].width
                            params = t.get_params(h, w)
                            
                            # Apply with same parameters
                            temp_frames = []
                            for frame in frames:
                                frame, boxes_temp, labels_temp = t((frame, boxes.copy(), labels.copy()))
                                temp_frames.append(frame)
                                if frame == frames[0]:  # Only keep boxes from first application
                                    boxes, labels = boxes_temp, labels_temp
                            frames = temp_frames
                        else:
                            # For transforms without get_params (like ColorJitter)
                            frame0, boxes, labels = t((frames[0], boxes, labels))
                            frames = [frame0] + [t((f, [], []))[0] for f in frames[1:]]
        
        # Convert all frames to tensors
        to_tensor = augmentation.ToTensorAndNormalize()
        tensor_frames = []
        for i, frame in enumerate(frames):
            if i == 0:
                tensor_frame, boxes, labels = to_tensor((frame, boxes, labels))
            else:
                tensor_frame, _, _ = to_tensor((frame, [], []))
            tensor_frames.append(tensor_frame)
        
        # Stack frames along temporal dimension
        frames_tensor = torch.stack(tensor_frames, dim=0)  # (T, C, H, W)
        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        return frames_tensor, boxes, labels


class TemporalBatchSampler(torch.utils.data.Sampler):
    """Sampler that ensures temporal consistency in batches"""
    
    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
    
    def __iter__(self):
        # Get all valid indices
        indices = list(range(len(self.dataset)))
        
        # Shuffle indices
        torch.manual_seed(torch.randint(0, 2**32, (1,)).item())
        perm = torch.randperm(len(indices)).tolist()
        indices = [indices[i] for i in perm]
        
        # Create batches
        for i in range(0, len(indices), self.batch_size):
            if i + self.batch_size <= len(indices):
                yield indices[i:i + self.batch_size]
            elif not self.drop_last:
                yield indices[i:]
    
    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def temporal_collate_fn(batch):
    """Custom collate function for temporal data"""
    
    # frames: List of (T, C, H, W) tensors
    # boxes: List of box tensors
    # labels: List of label tensors
    
    frames = torch.stack([item[0] for item in batch], dim=0)  # (B, T, C, H, W)
    boxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    
    return frames, boxes, labels


def create_temporal_issia_dataset(dataset_path, cameras, mode, n_frames=3,
                                frame_interval=1, only_ball_frames=False):
    """Create temporal ISSIA dataset"""
    
    assert mode in ['train', 'val']
    assert os.path.exists(dataset_path), f'Cannot find dataset: {dataset_path}'
    
    train_image_size = (720, 1280)
    val_image_size = (1080, 1920)
    
    if mode == 'train':
        transform = augmentation.TrainAugmentation(size=train_image_size)
    else:
        transform = augmentation.NoAugmentation(size=val_image_size)
    
    dataset = TemporalIssiaDataset(
        dataset_path, cameras, transform,
        n_frames=n_frames,
        frame_interval=frame_interval,
        only_ball_frames=only_ball_frames
    )
    
    return dataset


# Example usage in training script modification
def make_temporal_dataloaders(params, n_frames=3, frame_interval=1):
    """Create dataloaders for temporal training"""
    
    from torch.utils.data import DataLoader
    
    train_dataset = create_temporal_issia_dataset(
        params.issia_path,
        params.issia_train_cameras,
        mode='train',
        n_frames=n_frames,
        frame_interval=frame_interval,
        only_ball_frames=False
    )
    
    val_dataset = None
    if len(params.issia_val_cameras) > 0:
        val_dataset = create_temporal_issia_dataset(
            params.issia_path,
            params.issia_val_cameras,
            mode='val',
            n_frames=n_frames,
            frame_interval=frame_interval,
            only_ball_frames=True
        )
    
    dataloaders = {}
    
    # Training dataloader
    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        pin_memory=True,
        collate_fn=temporal_collate_fn
    )
    
    # Validation dataloader
    if val_dataset is not None:
        dataloaders['val'] = DataLoader(
            val_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=params.num_workers,
            pin_memory=True,
            collate_fn=temporal_collate_fn
        )
    
    return dataloaders


if __name__ == '__main__':
    # Test the temporal dataset
    from misc.config import Params
    
    # Create a mock params object
    class MockParams:
        issia_path = '/path/to/issia/dataset'
        issia_train_cameras = [1]
        issia_val_cameras = []
        batch_size = 4
        num_workers = 2
    
    params = MockParams()
    
    # Create dataset
    dataset = create_temporal_issia_dataset(
        params.issia_path,
        params.issia_train_cameras,
        mode='train',
        n_frames=3,
        frame_interval=1
    )
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Test loading a sample
    if len(dataset) > 0:
        frames, boxes, labels = dataset[0]
        print(f"Frames shape: {frames.shape}")  # Should be (3, 3, 720, 1280)
        print(f"Boxes shape: {boxes.shape}")
        print(f"Labels shape: {labels.shape}")
