import torch
from data.issia_dataset import IssiaDataset
from data.issia_dataset import create_issia_dataset

class TemporalDatasetWrapper:
    # Wrap the original dataset into a temporal dataset
    def __init__(self, original_dataset, temporal_window=3, mode='repeat'):
        """
        Args:
            original_dataset: original IssiaDataset
            temporal_window: temporal window size
            mode: repeat current frame
        """
        self.dataset = original_dataset
        self.temporal_window = temporal_window
        self.mode = mode
        print(f"Temporal dataset wrapper: {mode} mode, window={temporal_window}")
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        if self.mode == 'repeat':
            # Simple strategy: repeat the current frame T times (for quick verification of fusion effect)
            image, boxes, labels = self.dataset[idx]
            
            # Repeat the frame and add a little noise to simulate slight variations
            temporal_images = []
            for t in range(self.temporal_window):
                if t == 0:
                    # The first frame remains as is
                    temporal_images.append(image)
                else:
                    # Add tiny random noise (standard deviation = 0.01)
                    noise = torch.randn_like(image) * 0.01
                    noisy_image = image + noise
                    noisy_image = torch.clamp(noisy_image, 0, 1)  # make sure range between [0,1]
                    temporal_images.append(noisy_image)
            
            temporal_images = torch.stack(temporal_images, dim=0)  # [T, 3, H, W]
            return temporal_images, boxes, labels
    
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
    def get_annotations(self, *args, **kwargs):
        # Forward to the original dataset
        return self.dataset.get_annotations(*args, **kwargs)
        
    def get_elems_with_ball(self):
        # Forward to the original dataset
        return self.dataset.get_elems_with_ball()
        
    @property
    def ball_images_ndx(self):
        # Forward to the original dataset
        return self.dataset.ball_images_ndx
        
    @property
    def no_ball_images_ndx(self):
        # Forward to the original dataset
        return self.dataset.no_ball_images_ndx


def create_temporal_issia_dataset(dataset_path, cameras, mode, temporal_window=3, 
                                  temporal_mode='repeat', only_ball_frames=False):
    # create temporal ISSIA dataset
    # create original dataset first
    original_dataset = create_issia_dataset(dataset_path, cameras, mode, only_ball_frames)
    
    # pack it as a temporal dataset
    temporal_dataset = TemporalDatasetWrapper(original_dataset, temporal_window, temporal_mode)
    
    return temporal_dataset
