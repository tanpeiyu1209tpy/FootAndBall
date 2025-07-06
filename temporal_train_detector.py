# Training script for Temporal FootAndBall detector

import torch
import torch.optim as optim
import tqdm
import argparse
import pickle
import numpy as np
import os
import time
import matplotlib.pyplot as plt

from network.temporal_footandball import build_temporal_footandball_detector
from temporal_dataset import make_temporal_dataloaders
from network.ssd_loss import SSDLoss
from misc.config import Params


MODEL_FOLDER = 'models'


def plot_losses(training_stats, model_name):
    """Plot training losses"""
    for loss_key in ['loss', 'loss_ball_c', 'loss_player_c', 'loss_player_l']:
        plt.figure()
        for phase in training_stats:
            if training_stats[phase]:  # Check if not empty
                values = [e[loss_key] for e in training_stats[phase]]
                plt.plot(values, label=phase)
        plt.title(loss_key)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{model_name}_{loss_key}.png')
        plt.close()


def train_temporal_model(model, optimizer, scheduler, num_epochs, dataloaders, 
                        device, model_name, n_frames=3):
    """Train temporal FootAndBall model"""
    
    # Loss weights
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
    
    phases = ['train']
    if 'val' in dataloaders:
        phases.append('val')
    
    # Training statistics
    training_stats = {'train': [], 'val': []}
    
    print(f'Training temporal model with {n_frames} frames...')
    
    for epoch in tqdm.tqdm(range(num_epochs)):
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            batch_stats = {
                'loss': [], 
                'loss_ball_c': [], 
                'loss_player_c': [], 
                'loss_player_l': []
            }
            
            # Iterate over data
            for batch_idx, (frames, boxes, labels) in enumerate(dataloaders[phase]):
                # frames shape: (B, T, C, H, W)
                frames = frames.to(device)
                B, T, C, H, W = frames.shape
                
                # Get ground truth for center frame (assuming center frame annotations)
                gt_maps = model.groundtruth_maps(boxes, labels, (H, W))
                gt_maps = [e.to(device) for e in gt_maps]
                
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass with temporal frames
                    predictions = model(frames)
                    
                    # Compute losses
                    optimizer.zero_grad()
                    loss_l_player, loss_c_player, loss_c_ball = criterion(predictions, gt_maps)
                    
                    loss = (alpha_l_player * loss_l_player + 
                           alpha_c_player * loss_c_player + 
                           alpha_c_ball * loss_c_ball)
                    
                    # Backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                batch_stats['loss'].append(loss.item())
                batch_stats['loss_ball_c'].append(loss_c_ball.item())
                batch_stats['loss_player_c'].append(loss_c_player.item())
                batch_stats['loss_player_l'].append(loss_l_player.item())
            
            # Average stats
            avg_batch_stats = {k: np.mean(v) for k, v in batch_stats.items()}
            training_stats[phase].append(avg_batch_stats)
            
            print(f'{phase} - Epoch {epoch+1}/{num_epochs} - '
                  f'Loss: {avg_batch_stats["loss"]:.4f} '
                  f'(Ball: {avg_batch_stats["loss_ball_c"]:.4f}, '
                  f'Player C: {avg_batch_stats["loss_player_c"]:.4f}, '
                  f'Player L: {avg_batch_stats["loss_player_l"]:.4f})')
        
        # Step scheduler
        scheduler.step()
    
    # Save final model
    model_filepath = os.path.join(MODEL_FOLDER, f'{model_name}_final.pth')
    torch.save(model.state_dict(), model_filepath)
    print(f'Model saved to {model_filepath}')
    
    # Save training stats
    with open(f'training_stats_{model_name}.pickle', 'wb') as f:
        pickle.dump(training_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Plot losses
    plot_losses(training_stats, model_name)
    
    return training_stats


def train_temporal(params: Params, n_frames=3, aggregation_type='attention'):
    """Main training function for temporal model"""
    
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)
    
    # Create temporal dataloaders
    dataloaders = make_temporal_dataloaders(params, n_frames=n_frames)
    print(f'Training set size: {len(dataloaders["train"].dataset)}')
    if 'val' in dataloaders:
        print(f'Validation set size: {len(dataloaders["val"].dataset)}')
    
    # Create temporal model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_temporal_footandball_detector(
        phase='train',
        n_frames=n_frames,
        aggregation_type=aggregation_type
    )
    
    # Print model summary
    print(f'\nTemporal FootAndBall Model Summary:')
    print(f'Number of frames: {n_frames}')
    print(f'Aggregation type: {aggregation_type}')
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Device: {device}\n')
    
    model = model.to(device)
    
    # Model name with timestamp
    model_name = f'temporal_model_{n_frames}frames_{aggregation_type}_' + time.strftime("%Y%m%d_%H%M")
    print(f'Model name: {model_name}')
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    scheduler_milestones = [int(params.epochs * 0.75)]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, scheduler_milestones, gamma=0.1
    )
    
    # Train the model
    return train_temporal_model(
        model, optimizer, scheduler, params.epochs, 
        dataloaders, device, model_name, n_frames
    )


# Modified run_detector for temporal model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.txt',
                       help='Path to configuration file')
    parser.add_argument('--n_frames', type=int, default=3,
                       help='Number of temporal frames')
    parser.add_argument('--aggregation', type=str, default='attention',
                       choices=['attention', 'mean', 'max', 'conv3d'],
                       help='Temporal aggregation method')
    args = parser.parse_args()
    
    print('Training Temporal FootAndBall Detector')
    print(f'Config: {args.config}')
    print(f'Number of frames: {args.n_frames}')
    print(f'Aggregation: {args.aggregation}')
    
    params = Params(args.config)
    params.print()
    
    train_temporal(params, n_frames=args.n_frames, 
                  aggregation_type=args.aggregation)
