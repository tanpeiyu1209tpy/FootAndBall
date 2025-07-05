# Multi-frame data augmentation for Early Fusion
# 多帧数据增强，确保时序一致性

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import random

BALL_BBOX_SIZE = 20
BALL_LABEL = 1
PLAYER_LABEL = 2


class MultiFrameTrainAugmentation:
    """训练时的多帧数据增强，保持时序一致性"""
    def __init__(self, size, num_frames=3):
        self.size = size  # (height, width)
        self.num_frames = num_frames
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
    def __call__(self, sample):
        images, boxes, labels = sample
        
        if not isinstance(images, list):
            images = [images]
            
        # Ensure we have the right number of frames
        assert len(images) == self.num_frames, f"Expected {self.num_frames} frames, got {len(images)}"
        
        # Convert PIL images to numpy arrays
        np_images = []
        for img in images:
            if isinstance(img, Image.Image):
                np_images.append(np.array(img))
            else:
                np_images.append(img)
        
        # Apply consistent transformations across all frames
        # 1. Resize all frames to the same size
        resized_images = []
        original_size = np_images[0].shape[:2]  # (H, W)
        scale_y = self.size[0] / original_size[0]
        scale_x = self.size[1] / original_size[1]
        
        for img in np_images:
            resized_img = cv2.resize(img, (self.size[1], self.size[0]))
            resized_images.append(resized_img)
        
        # Scale bounding boxes accordingly
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale_x  # x coordinates
            boxes[:, [1, 3]] *= scale_y  # y coordinates
        
        # 2. Random horizontal flip (applied to all frames consistently)
        if random.random() > 0.5:
            for i in range(len(resized_images)):
                resized_images[i] = np.fliplr(resized_images[i]).copy()
            
            # Flip bounding boxes
            if len(boxes) > 0:
                width = self.size[1]
                old_x1 = boxes[:, 0].copy()
                old_x2 = boxes[:, 2].copy()
                boxes[:, 0] = width - old_x2
                boxes[:, 2] = width - old_x1
        
        # 3. Random brightness/contrast adjustments (applied consistently)
        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        
        for i in range(len(resized_images)):
            img = resized_images[i].astype(np.float32) / 255.0
            # Apply brightness
            img = img * brightness_factor
            # Apply contrast
            img = (img - 0.5) * contrast_factor + 0.5
            img = np.clip(img, 0, 1)
            resized_images[i] = img
        
        # 4. Normalize and convert to tensors
        tensor_images = []
        for img in resized_images:
            # Normalize
            img = (img - self.mean) / self.std
            # Convert to tensor and change from HWC to CHW
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
            tensor_images.append(img_tensor)
        
        return tensor_images, boxes, labels


class MultiFrameNoAugmentation:
    """验证时的多帧数据处理，不进行数据增强"""
    def __init__(self, size, num_frames=3):
        self.size = size  # (height, width)
        self.num_frames = num_frames
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
    def __call__(self, sample):
        images, boxes, labels = sample
        
        if not isinstance(images, list):
            images = [images]
            
        # Ensure we have the right number of frames
        assert len(images) == self.num_frames, f"Expected {self.num_frames} frames, got {len(images)}"
        
        # Convert PIL images to numpy arrays
        np_images = []
        for img in images:
            if isinstance(img, Image.Image):
                np_images.append(np.array(img))
            else:
                np_images.append(img)
        
        # Resize all frames to the same size
        resized_images = []
        original_size = np_images[0].shape[:2]  # (H, W)
        scale_y = self.size[0] / original_size[0]
        scale_x = self.size[1] / original_size[1]
        
        for img in np_images:
            resized_img = cv2.resize(img, (self.size[1], self.size[0]))
            resized_images.append(resized_img)
        
        # Scale bounding boxes accordingly
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale_x  # x coordinates
            boxes[:, [1, 3]] *= scale_y  # y coordinates
        
        # Normalize and convert to tensors
        tensor_images = []
        for img in resized_images:
            img = img.astype(np.float32) / 255.0
            # Normalize
            img = (img - self.mean) / self.std
            # Convert to tensor and change from HWC to CHW
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
            tensor_images.append(img_tensor)
        
        return tensor_images, boxes, labels
