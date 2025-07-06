import random
import torch
from torch.utils.data import Sampler, DataLoader, ConcatDataset
from misc.config import Params
from data.temporal_issia_dataset import TemporalIssiaDataset, create_temporal_issia_dataset

def make_temporal_dataloaders(params, temporal_frames=3):
    """
    MODIFICATION: Create dataloaders for temporal dataset
    """
    if params.issia_path is None:
        train_issia_dataset = None
    else:
        train_issia_dataset = create_temporal_issia_dataset(
            params.issia_path, params.issia_train_cameras, mode='train',
            only_ball_frames=False, temporal_frames=temporal_frames
        )
        if len(params.issia_val_cameras) == 0:
            val_issia_dataset = None
        else:
            val_issia_dataset = create_temporal_issia_dataset(
                params.issia_path, params.issia_val_cameras, mode='val',
                only_ball_frames=True, temporal_frames=temporal_frames
            )
            
    dataloaders = {}
    if val_issia_dataset is not None:
        dataloaders['val'] = DataLoader(
            val_issia_dataset, batch_size=2, num_workers=params.num_workers,
            pin_memory=True, collate_fn=my_collate
        )
    
    train_dataset = train_issia_dataset
    batch_sampler = TemporalBalancedSampler(train_dataset)
    dataloaders['train'] = DataLoader(
        train_dataset, sampler=batch_sampler, batch_size=params.batch_size,
        num_workers=params.num_workers, pin_memory=True, collate_fn=my_collate
    )

    return dataloaders

def my_collate(batch):
    images = torch.stack([e[0] for e in batch], dim=0)
    boxes = [e[1] for e in batch]
    labels = [e[2] for e in batch]
    return images, boxes, labels

class TemporalBalancedSampler(Sampler):
    """
    MODIFICATION: Sampler for temporal dataset
    """
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source
        self.sample_ndx = []
        self.generate_samples()

    def generate_samples(self):
        assert isinstance(self.data_source, TemporalIssiaDataset), 'Training data must be Temporal ISSIA CNR dataset.'
        
        issia_ds = self.data_source
        n_ball_images = len(issia_ds.ball_images_ndx)
        # no_ball_images = 0.5 * ball_images
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
    images, targets = zip(*batch)
    return torch.stack(images, 0), targets
