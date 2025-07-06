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
from data.data_reader import make_dataloaders
from network.ssd_loss import SSDLoss
from misc.config import Params

MODEL_FOLDER = 'models'

def plot_losses(training_stats, model_name):
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
            
def train_model(model, optimizer, scheduler, num_epochs, dataloaders, device, model_name):
    
    alpha_l_player = 0.01
    alpha_c_player = 1.
    alpha_c_ball = 5.

    total = alpha_l_player + alpha_c_player + alpha_c_ball
    alpha_l_player /= total
    alpha_c_player /= total
    alpha_c_ball /= total

    criterion = SSDLoss(neg_pos_ratio=3)

    is_validation_set = 'val' in dataloaders
    if is_validation_set:
        phases = ['train', 'val']
    else:
        phases = ['train']

    training_stats = {'train': [], 'val': []}

    print('Training...')
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
                
                # 处理temporal input的shape
                if images.dim() == 5:  # [B, T, 3, H, W]
                    h, w = images.shape[-2], images.shape[-1]
                else:  # [B, 3, H, W]
                    h, w = images.shape[-2], images.shape[-1]
                
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
                torch.cuda.empty_cache()

                count_batches += 1
                batch_stats['loss'].append(loss.item())
                batch_stats['loss_ball_c'].append(loss_c_ball.item())
                batch_stats['loss_player_c'].append(loss_c_player.item())
                batch_stats['loss_player_l'].append(loss_l_player.item())

            avg_batch_stats = {}
            for e in batch_stats:
                avg_batch_stats[e] = np.mean(batch_stats[e])

            training_stats[phase].append(avg_batch_stats)
            s = '{} Avg. loss total / ball conf. / player conf. / player loc.: {:.4f} / {:.4f} / {:.4f} / {:.4f}'
            print(s.format(phase, avg_batch_stats['loss'], avg_batch_stats['loss_ball_c'],
                           avg_batch_stats['loss_player_c'], avg_batch_stats['loss_player_l']))

        scheduler.step()

    model_filepath = os.path.join(MODEL_FOLDER, model_name + '_final' + '.pth')
    torch.save(model.state_dict(), model_filepath)

    with open('training_stats_{}.pickle'.format(model_name), 'wb') as handle:
        pickle.dump(training_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plot_losses(training_stats, model_name)
    
    return training_stats


def train(params: Params, model_name='fb1', use_temporal_fusion=False, temporal_window=3, fusion_method='difference'):
    """
    Args:
        model_name: 模型名称 (从命令行传入)
        use_temporal_fusion: 是否使用temporal fusion
        temporal_window: 时间窗口大小  
        fusion_method: fusion方法
    """
    if not os.path.exists(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)

    assert os.path.exists(MODEL_FOLDER), ' Cannot create folder to save trained model: {}'.format(MODEL_FOLDER)

    dataloaders = make_dataloaders(params, use_temporal=use_temporal_fusion, 
                                   temporal_window=temporal_window, fusion_method=fusion_method)
    print('Training set: Dataset size: {}'.format(len(dataloaders['train'].dataset)))
    if 'val' in dataloaders:
        print('Validation set: Dataset size: {}'.format(len(dataloaders['val'].dataset)))

    # 使用命令行传入的model_name
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = footandball.model_factory(model_name, 'train',  # ←使用传入的model_name
                                      use_temporal_fusion=use_temporal_fusion,
                                      temporal_window=temporal_window,
                                      fusion_method=fusion_method)
    model.print_summary(show_architecture=True)
    model = model.to(device)

    # Model name包含所有关键信息
    suffix = f"_temporal_{fusion_method}_w{temporal_window}" if use_temporal_fusion else ""
    model_name = f'model_{model_name}_' + time.strftime("%Y%m%d_%H%M") + suffix
    print('Model name: {}'.format(model_name))

    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    scheduler_milestones = [int(params.epochs * 0.75)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, scheduler_milestones, gamma=0.1)
    
    return train_model(model, optimizer, scheduler, params.epochs, dataloaders, device, model_name)


if __name__ == '__main__':
    print('Train FootAndBall detector on ISSIA dataset')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to the configuration file', type=str, default='config.txt')
    parser.add_argument('--debug', dest='debug', help='debug mode', action='store_true')
    
    # Model参数
    parser.add_argument('--model', help='model name', type=str, default='fb1')
    
    # Temporal fusion参数
    parser.add_argument('--temporal', dest='use_temporal', help='use temporal fusion', action='store_true')
    parser.add_argument('--temporal-window', help='temporal window size', type=int, default=3)
    parser.add_argument('--fusion-method', help='fusion method', type=str, default='difference',
                        choices=['difference', 'variance', 'weighted_avg', 'attention'])
    
    args = parser.parse_args()

    print('Config path: {}'.format(args.config))
    print('Model: {}'.format(args.model))
    print('Debug mode: {}'.format(args.debug))
    print('Temporal fusion: {}'.format(args.use_temporal))
    if args.use_temporal:
        print('Temporal window: {}'.format(args.temporal_window))
        print('Fusion method: {}'.format(args.fusion_method))

    params = Params(args.config)
    params.print()

    # 传递所有命令行参数
    train(params, model_name=args.model, use_temporal_fusion=args.use_temporal, 
          temporal_window=args.temporal_window, fusion_method=args.fusion_method)
