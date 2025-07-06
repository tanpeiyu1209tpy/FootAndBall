import random
import torch
from torch.utils.data import Sampler, DataLoader, ConcatDataset

from data.issia_dataset import create_issia_dataset, IssiaDataset
from data.temporal_dataset import TemporalDatasetWrapper, create_temporal_issia_dataset  # 新增import
from misc.config import Params


def make_dataloaders(params: Params, use_temporal=False, temporal_window=3, fusion_method='difference'):
    """
    Args:
        use_temporal: whether to use temporal fusion
        temporal_window: temporal window size
        fusion_method: fusion options
    """
    if params.issia_path is None:
        train_issia_dataset = None
    else:
        if use_temporal:
            # use temporal dataset
            train_issia_dataset = create_temporal_issia_dataset(
                params.issia_path, params.issia_train_cameras, mode='train',
                temporal_window=temporal_window, temporal_mode='repeat', only_ball_frames=False)
            
            if len(params.issia_val_cameras) == 0:
                val_issia_dataset = None
            else:
                val_issia_dataset = create_temporal_issia_dataset(
                    params.issia_path, params.issia_val_cameras, mode='val',
                    temporal_window=temporal_window, temporal_mode='repeat', only_ball_frames=True)
        else:
            # use original single frame dataset
            train_issia_dataset = create_issia_dataset(params.issia_path, params.issia_train_cameras, mode='train',
                                                       only_ball_frames=False)
            if len(params.issia_val_cameras) == 0:
                val_issia_dataset = None
            else:
                val_issia_dataset = create_issia_dataset(params.issia_path, params.issia_val_cameras, mode='val',
                                                         only_ball_frames=True)
            
    dataloaders = {}
    
    # choose the suitable collate function
    collate_fn = temporal_collate if use_temporal else my_collate
    
    if val_issia_dataset is not None:
        dataloaders['val'] = DataLoader(val_issia_dataset, batch_size=2, num_workers=params.num_workers,
                                        pin_memory=True, collate_fn=collate_fn)
    
    train_dataset = train_issia_dataset
    batch_sampler = BalancedSampler(train_dataset)
    dataloaders['train'] = DataLoader(train_dataset, sampler=batch_sampler, batch_size=params.batch_size,
                                      num_workers=params.num_workers, pin_memory=True, collate_fn=collate_fn)

    return dataloaders


def my_collate(batch):
    #original collate function - single frame
    images = torch.stack([e[0] for e in batch], dim=0)
    boxes = [e[1] for e in batch]
    labels = [e[2] for e in batch]
    return images, boxes, labels


def temporal_collate(batch):
    #new collate function - multiframe temporal
    temporal_images = torch.stack([e[0] for e in batch], dim=0)  # [B, T, 3, H, W]
    boxes = [e[1] for e in batch]
    labels = [e[2] for e in batch]
    return temporal_images, boxes, labels


class BalancedSampler(Sampler):
    # for single frame
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source
        self.sample_ndx = []
        self.generate_samples()
    
    def generate_samples(self):
        # process single frame dataset
        if isinstance(self.data_source, TemporalDatasetWrapper):
            issia_ds = self.data_source.dataset  # get single frame dataset
        elif isinstance(self.data_source, IssiaDataset):
            issia_ds = self.data_source
        else:
            raise ValueError(f"Unsupported dataset type: {type(self.data_source)}")
        
        n_ball_images = len(issia_ds.ball_images_ndx)
        n_no_ball_images = min(len(issia_ds.no_ball_images_ndx), int(0.5 * n_ball_images))
        
        self.sample_ndx = list(issia_ds.ball_images_ndx) + random.sample(list(issia_ds.no_ball_images_ndx),
                                                                         n_no_ball_images)
        random.shuffle(self.sample_ndx)

    def __iter__(self):
        self.generate_samples()
        for ndx in self.sample_ndx:
            yield ndx

    def __len(self):
        return len(self.sample_ndx)


def collate_fn(batch):
    """备用collate function"""
    images, targets = zip(*batch)
    return torch.stack(images, 0), targets
