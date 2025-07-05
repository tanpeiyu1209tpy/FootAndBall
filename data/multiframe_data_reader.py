# Multi-frame Data Reader for Early Fusion
# 支持多帧Early Fusion的数据加载器

import random
import torch
from torch.utils.data import Sampler, DataLoader, ConcatDataset
from data.issia_utils import make_multiframe_transform
# Import multi-frame dataset (assuming it's in the same module)
from data.multiframe_issia_dataset import create_multiframe_issia_dataset, MultiFrameIssiaDataset
from misc.config import Params


def make_multiframe_dataloaders(params: Params, num_frames=3, frame_interval=1, temporal_strategy='center'):
    """
    Create data loaders for multi-frame training
    
    Args:
        params: Configuration parameters
        num_frames: Number of frames per sample
        frame_interval: Interval between frames  
        temporal_strategy: 'center' or 'last' - which frame to use as target
    
    Returns:
        Dictionary containing train and validation data loaders
    """
    transform = make_multiframe_transform()
    
    if params.issia_path is None:
        train_issia_dataset = None
    else:
        train_issia_dataset = create_multiframe_issia_dataset(
            params.issia_path, 
            params.issia_train_cameras, 
            mode='train',
            only_ball_frames=False,
            num_frames=num_frames,
            frame_interval=frame_interval,
            temporal_strategy=temporal_strategy
            transform=transform
        )
        
        if len(params.issia_val_cameras) == 0:
            val_issia_dataset = None
        else:
            val_issia_dataset = create_multiframe_issia_dataset(
                params.issia_path, 
                params.issia_val_cameras, 
                mode='val',
                only_ball_frames=True,
                num_frames=num_frames,
                frame_interval=frame_interval,
                temporal_strategy=temporal_strategy
            )
            
    dataloaders = {}
    
    if val_issia_dataset is not None:
        dataloaders['val'] = DataLoader(
            val_issia_dataset, 
            batch_size=2, 
            num_workers=params.num_workers,
            pin_memory=True, 
            collate_fn=multiframe_collate
        )
    
    train_dataset = train_issia_dataset
    batch_sampler = MultiFrameBalancedSampler(train_dataset)
    dataloaders['train'] = DataLoader(
        train_dataset, 
        sampler=batch_sampler, 
        batch_size=params.batch_size,
        num_workers=params.num_workers, 
        pin_memory=True, 
        collate_fn=multiframe_collate
    )

    return dataloaders


def multiframe_collate(batch):
    """
    Custom collate function for multi-frame data
    
    Args:
        batch: List of (multi_frame_image, boxes, labels) tuples
    
    Returns:
        Tuple of (stacked_images, list_of_boxes, list_of_labels)
    """
    images = torch.stack([e[0] for e in batch], dim=0)
    boxes = [e[1] for e in batch]
    labels = [e[2] for e in batch]
    return images, boxes, labels


class MultiFrameBalancedSampler(Sampler):
    """Balanced sampler for multi-frame datasets"""
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source
        self.sample_ndx = []
        self.generate_samples()
    
    def generate_samples(self):
        """Generate balanced samples for multi-frame dataset"""
        assert isinstance(self.data_source, MultiFrameIssiaDataset), 'Training data must be MultiFrameIssiaDataset.'
        
        issia_ds = self.data_source
        n_ball_images = len(issia_ds.ball_images_ndx)
        # Use fewer no-ball images to maintain balance
        n_no_ball_images = min(len(issia_ds.no_ball_images_ndx), int(0.5 * n_ball_images))
        
        self.sample_ndx = list(issia_ds.ball_images_ndx) + random.sample(
            list(issia_ds.no_ball_images_ndx), n_no_ball_images
        )
        random.shuffle(self.sample_ndx)

    def __iter__(self):
        self.generate_samples()  # Re-generate samples every epoch
        for ndx in self.sample_ndx:
            yield ndx

    def __len__(self):
        return len(self.sample_ndx)
