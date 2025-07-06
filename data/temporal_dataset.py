import torch
from data.issia_dataset import IssiaDataset


class TemporalDatasetWrapper:
    """将原始dataset包装成temporal dataset"""
    def __init__(self, original_dataset, temporal_window=3, mode='repeat'):
        """
        Args:
            original_dataset: 原始的IssiaDataset
            temporal_window: 时间窗口大小
            mode: 'repeat' - 重复当前帧, 'consecutive' - 连续帧 (待实现)
        """
        self.dataset = original_dataset
        self.temporal_window = temporal_window
        self.mode = mode
        print(f"🎬 Temporal dataset wrapper: {mode} mode, window={temporal_window}")
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        if self.mode == 'repeat':
            # 简单策略：重复当前帧T次 (用于快速验证fusion效果)
            image, boxes, labels = self.dataset[idx]
            
            # 重复帧，添加微小的噪声模拟slight variations
            temporal_images = []
            for t in range(self.temporal_window):
                if t == 0:
                    # 第一帧保持原样
                    temporal_images.append(image)
                else:
                    # 添加微小随机噪声 (标准差=0.01)
                    noise = torch.randn_like(image) * 0.01
                    noisy_image = image + noise
                    noisy_image = torch.clamp(noisy_image, 0, 1)  # 确保在[0,1]范围内
                    temporal_images.append(noisy_image)
            
            temporal_images = torch.stack(temporal_images, dim=0)  # [T, 3, H, W]
            return temporal_images, boxes, labels
            
        elif self.mode == 'consecutive':
            # TODO: 实现真正的连续帧读取
            # 这需要更复杂的逻辑来处理sequence boundaries
            raise NotImplementedError("Consecutive frame mode not implemented yet")
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
    def get_annotations(self, *args, **kwargs):
        """转发到原始dataset"""
        return self.dataset.get_annotations(*args, **kwargs)
        
    def get_elems_with_ball(self):
        """转发到原始dataset"""
        return self.dataset.get_elems_with_ball()
        
    @property
    def ball_images_ndx(self):
        """转发到原始dataset"""
        return self.dataset.ball_images_ndx
        
    @property
    def no_ball_images_ndx(self):
        """转发到原始dataset"""
        return self.dataset.no_ball_images_ndx


def create_temporal_issia_dataset(dataset_path, cameras, mode, temporal_window=3, 
                                  temporal_mode='repeat', only_ball_frames=False):
    """创建temporal ISSIA dataset"""
    # 先创建原始dataset
    from data.issia_dataset import create_issia_dataset
    original_dataset = create_issia_dataset(dataset_path, cameras, mode, only_ball_frames)
    
    # 包装成temporal dataset
    temporal_dataset = TemporalDatasetWrapper(original_dataset, temporal_window, temporal_mode)
    
    return temporal_dataset
