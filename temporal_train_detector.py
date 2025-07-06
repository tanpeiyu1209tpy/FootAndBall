
import tqdm
import argparse
import pickle
import numpy as np
import os
import time

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from network import footandball
from data.data_reader import make_dataloaders, make_temporal_dataloaders
from network.ssd_loss import SSDLoss
from misc.config import Params

MODEL_FOLDER = 'models'

def train(params: Params, override_model=None):
    """
    修改：支持通过override_model参数覆盖config中的model设置
    """
    if not os.path.exists(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)

    assert os.path.exists(MODEL_FOLDER), 'Cannot create folder to save trained model: {}'.format(MODEL_FOLDER)

    # 如果提供了override_model，使用它替代config中的model
    if override_model:
        print(f'Overriding model from config ({params.model}) with command line argument ({override_model})')
        model_name = override_model
    else:
        model_name = params.model

    # 根据model类型选择数据加载器
    if model_name == 'temporal_fb1':
        # 时序模型
        temporal_frames = getattr(params, 'temporal_frames', 3)
        use_motion_module = getattr(params, 'use_motion_module', True)
        dataloaders = make_temporal_dataloaders(params, temporal_frames=temporal_frames)
        
        # 创建时序模型
        device = "cuda" if torch.cuda.is_available() else 'cpu'
        model = footandball.model_factory(
            model_name, 'train', 
            temporal_frames=temporal_frames,
            use_motion_module=use_motion_module
        )
    else:
        # 单帧模型
        dataloaders = make_dataloaders(params)
        
        # 创建单帧模型
        device = "cuda" if torch.cuda.is_available() else 'cpu'
        model = footandball.model_factory(model_name, 'train')
    
    print(f'Using model: {model_name}')
    print('Training set: Dataset size: {}'.format(len(dataloaders['train'].dataset)))
    if 'val' in dataloaders:
        print('Validation set: Dataset size: {}'.format(len(dataloaders['val'].dataset)))

    model.print_summary(show_architecture=True)
    model = model.to(device)

    save_model_name = 'model_{}_{}'.format(model_name, time.strftime("%Y%m%d_%H%M"))
    print('Model save name: {}'.format(save_model_name))

    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    scheduler_milestones = [int(params.epochs * 0.75)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, scheduler_milestones, gamma=0.1)
    
    return train_model(model, optimizer, scheduler, params.epochs, dataloaders, device, save_model_name)


if __name__ == '__main__':
    print('Train FootAndBall detector on ISSIA dataset')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to the configuration file', type=str, default='config.txt')
    parser.add_argument('--model', help='Override model type (fb1 or temporal_fb1)', type=str, default=None)
    parser.add_argument('--temporal-frames', help='Number of temporal frames (for temporal models)', type=int, default=None)
    parser.add_argument('--use-motion-module', help='Use motion module (for temporal models)', action='store_true')
    parser.add_argument('--no-motion-module', help='Disable motion module (for temporal models)', action='store_true')
    parser.add_argument('--debug', dest='debug', help='debug mode', action='store_true')
    args = parser.parse_args()

    print('Config path: {}'.format(args.config))
    print('Debug mode: {}'.format(args.debug))

    params = Params(args.config)
    
    # 如果命令行提供了temporal参数，覆盖config中的设置
    if args.temporal_frames is not None:
        params.temporal_frames = args.temporal_frames
        print(f'Overriding temporal_frames to: {args.temporal_frames}')
    
    if args.use_motion_module:
        params.use_motion_module = True
        print('Enabling motion module')
    elif args.no_motion_module:
        params.use_motion_module = False
        print('Disabling motion module')
    
    params.print()

    train(params, override_model=args.model)
