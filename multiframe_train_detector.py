# FootAndBall: Multi-frame Training Script for Early Fusion
# 支持多帧Early Fusion的训练脚本

import tqdm
import argparse
import pickle
import numpy as np
import os
import time

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# Import both original and multi-frame components
from network import multiframe_footandball
from network.multiframe_footandball import multiframe_model_factory
from data.multiframe_data_reader import make_multiframe_dataloaders
from data.data_reader import make_dataloaders  # Original for backward compatibility
from network.ssd_loss import SSDLoss
from misc.config import Params

MODEL_FOLDER = 'models'


def plot_losses(training_stats, model_name):
    """Plot training losses"""
    for loss_key in ['loss', 'loss_ball_c', 'loss_player_c', 'loss_player_l']:
        plt.figure()
        for phase in training_stats:
            values = [e[loss_key] for e in training_stats[phase]]
            plt.plot(values, label=phase)
        plt.title(loss_key)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{model_name}_{loss_key}.png')
        plt.close()


def train_model(model, optimizer, scheduler, num_epochs, dataloaders, device, model_name, use_multiframe=False):
    """
    Train the model with support for both single-frame and multi-frame inputs
    
    Args:
        model: Model to train
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs
        dataloaders: Dictionary of data loaders
        device: Training device
        model_name: Name for saving model
        use_multiframe: Whether using multi-frame input
    """
    
    # Loss function weights
    alpha_l_player = 0.01
    alpha_c_player = 1.
    alpha_c_ball = 5.

    # Normalize weights
    total = alpha_l_player + alpha_c_player + alpha_c_ball
    alpha_l_player /= total
    alpha_c_player /= total
    alpha_c_ball /= total

    # Loss function
    criterion = SSDLoss(neg_pos_ratio=3)

    is_validation_set = 'val' in dataloaders
    phases = ['train', 'val'] if is_validation_set else ['train']

    # Training statistics
    training_stats = {'train': [], 'val': []}

    print(f'Training {"multi-frame" if use_multiframe else "single-frame"} model...')
    
    for epoch in tqdm.tqdm(range(num_epochs)):
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            batch_stats = {'loss': [], 'loss_ball_c': [], 'loss_player_c': [], 'loss_player_l': []}
            count_batches = 0

            for ndx, (images, boxes, labels) in enumerate(dataloaders[phase]):
                images = images.to(device)

                h, w = images.shape[-2], images.shape[-1]
                
                # For multi-frame models, we need to handle the input differently
                if use_multiframe:
                    # Verify input shape for multi-frame
                    expected_channels = 3 * getattr(model, 'num_frames', 1)
                    assert images.shape[1] == expected_channels, \
                        f"Expected {expected_channels} channels, got {images.shape[1]}"
                
                gt_maps = model.groundtruth_maps(boxes, labels, (h, w))
                gt_maps = [e.to(device) for e in gt_maps]
                count_batches += 1

                with torch.set_grad_enabled(phase == 'train'):
                    predictions = model(images)
                    optimizer.zero_grad()
                    loss_l_player, loss_c_player, loss_c_ball = criterion(predictions, gt_maps)

                    loss = alpha_l_player * loss_l_player + alpha_c_player * loss_c_player + alpha_c_ball * loss_c_ball

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                batch_stats['loss'].append(loss.item())
                batch_stats['loss_ball_c'].append(loss_c_ball.item())
                batch_stats['loss_player_c'].append(loss_c_player.item())
                batch_stats['loss_player_l'].append(loss_l_player.item())

            # Average stats per batch
            avg_batch_stats = {}
            for e in batch_stats:
                avg_batch_stats[e] = np.mean(batch_stats[e])

            training_stats[phase].append(avg_batch_stats)
            s = '{} Avg. loss total / ball conf. / player conf. / player loc.: {:.4f} / {:.4f} / {:.4f} / {:.4f}'
            print(s.format(phase, avg_batch_stats['loss'], avg_batch_stats['loss_ball_c'],
                           avg_batch_stats['loss_player_c'], avg_batch_stats['loss_player_l']))

        scheduler.step()

    # Save model
    model_filepath = os.path.join(MODEL_FOLDER, model_name + '_final' + '.pth')
    torch.save(model.state_dict(), model_filepath)

    # Save training statistics
    with open('training_stats_{}.pickle'.format(model_name), 'wb') as handle:
        pickle.dump(training_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Plot losses
    plot_losses(training_stats, model_name)
    
    return training_stats


def train_multiframe(params: Params, num_frames=3, frame_interval=1, temporal_strategy='center'):
    """
    Train multi-frame model
    
    Args:
        params: Configuration parameters
        num_frames: Number of frames per sample
        frame_interval: Interval between frames
        temporal_strategy: Temporal sampling strategy
    """
    if not os.path.exists(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)

    assert os.path.exists(MODEL_FOLDER), f'Cannot create folder to save trained model: {MODEL_FOLDER}'

    # Create multi-frame data loaders
    dataloaders = make_multiframe_dataloaders(
        params, 
        num_frames=num_frames, 
        frame_interval=frame_interval,
        temporal_strategy=temporal_strategy
    )
    
    print(f'Multi-frame training set: Dataset size: {len(dataloaders["train"].dataset)}')
    if 'val' in dataloaders:
        print(f'Multi-frame validation set: Dataset size: {len(dataloaders["val"].dataset)}')

    # Create multi-frame model
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = multiframe_model_factory(
        'mf_fb1', 'train',
        num_frames=num_frames
    )
    
    model.print_summary(show_architecture=True)
    model = model.to(device)

    model_name = f'multiframe_model_{num_frames}f_' + time.strftime("%Y%m%d_%H%M")
    print(f'Multi-frame model name: {model_name}')

    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    scheduler_milestones = [int(params.epochs * 0.75)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, scheduler_milestones, gamma=0.1)
    
    return train_model(model, optimizer, scheduler, params.epochs, dataloaders, device, model_name, use_multiframe=True)


def train(params: Params):
    """Original single-frame training for backward compatibility"""
    if not os.path.exists(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)

    assert os.path.exists(MODEL_FOLDER), f'Cannot create folder to save trained model: {MODEL_FOLDER}'

    dataloaders = make_dataloaders(params)
    print(f'Training set: Dataset size: {len(dataloaders["train"].dataset)}')
    if 'val' in dataloaders:
        print(f'Validation set: Dataset size: {len(dataloaders["val"].dataset)}')

    # Create single-frame model
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = footandball.model_factory(params.model, 'train')
    model.print_summary(show_architecture=True)
    model = model.to(device)

    model_name = 'model_' + time.strftime("%Y%m%d_%H%M")
    print(f'Model name: {model_name}')

    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    scheduler_milestones = [int(params.epochs * 0.75)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, scheduler_milestones, gamma=0.1)
    
    return train_model(model, optimizer, scheduler, params.epochs, dataloaders, device, model_name, use_multiframe=False)


def compare_models(params: Params, num_frames_list=[1, 3, 5], frame_interval=1, temporal_strategy='center'):
    """
    Compare performance of different temporal configurations
    
    Args:
        params: Configuration parameters
        num_frames_list: List of frame counts to test
        frame_interval: Interval between frames
        temporal_strategy: Temporal sampling strategy
    """
    results = {}
    
    for num_frames in num_frames_list:
        print(f"\n{'='*50}")
        print(f"Training with {num_frames} frames")
        print(f"{'='*50}")
        
        if num_frames == 1:
            # Use original single-frame training
            training_stats = train(params)
            model_type = 'single_frame'
        else:
            # Use multi-frame training
            training_stats = train_multiframe(
                params, 
                num_frames=num_frames, 
                frame_interval=frame_interval,
                temporal_strategy=temporal_strategy
            )
            model_type = f'multi_frame_{num_frames}f'
        
        results[model_type] = training_stats
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    for model_type, stats in results.items():
        train_losses = [epoch['loss'] for epoch in stats['train']]
        plt.plot(train_losses, label=f'{model_type} (train)')
        
        if 'val' in stats and len(stats['val']) > 0:
            val_losses = [epoch['loss'] for epoch in stats['val']]
            plt.plot(val_losses, label=f'{model_type} (val)', linestyle='--')
    
    plt.title('Model Comparison: Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('model_comparison.png')
    plt.close()
    
    # Save comparison results
    with open('comparison_results.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return results


if __name__ == '__main__':
    print('Train FootAndBall detector with multi-frame support')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to the configuration file', type=str, default='config.txt')
    parser.add_argument('--debug', dest='debug', help='Debug mode', action='store_true')
    parser.add_argument('--multiframe', dest='multiframe', help='Use multi-frame training', action='store_true')
    parser.add_argument('--num_frames', help='Number of frames for multi-frame training', type=int, default=3)
    parser.add_argument('--frame_interval', help='Interval between frames', type=int, default=1)
    parser.add_argument('--temporal_strategy', help='Temporal strategy (center/last)', type=str, default='center')
    parser.add_argument('--compare', dest='compare', help='Compare different frame configurations', action='store_true')
    
    args = parser.parse_args()

    print(f'Config path: {args.config}')
    print(f'Debug mode: {args.debug}')
    print(f'Multi-frame mode: {args.multiframe}')
    if args.multiframe:
        print(f'Number of frames: {args.num_frames}')
        print(f'Frame interval: {args.frame_interval}')
        print(f'Temporal strategy: {args.temporal_strategy}')

    params = Params(args.config)
    params.print()

    if args.compare:
        # Compare different configurations
        compare_models(
            params, 
            num_frames_list=[1, 3, 5], 
            frame_interval=args.frame_interval,
            temporal_strategy=args.temporal_strategy
        )
    elif args.multiframe:
        # Train multi-frame model
        train_multiframe(
            params, 
            num_frames=args.num_frames, 
            frame_interval=args.frame_interval,
            temporal_strategy=args.temporal_strategy
        )
    else:
        # Train original single-frame model
        train(params)
