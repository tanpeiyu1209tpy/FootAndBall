# FootAndBall: Integrated Player and Ball Detector
# Jacek Komorowski, Grzegorz Kurzejamski, Grzegorz Sarwas
# Copyright (c) 2020 Sport Algorithmics and Gaming

#
# Train FootAndBall detector on ISSIA-CNR Soccer and SoccerPlayerDetection dataset
#

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
from evaluate import eval_model

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

def plot_eval_history(history, model_name):
    import matplotlib.pyplot as plt
    keys = ['Ball AP', 'Player AP', 'mAP']
    for key in keys:
        values = [e[key] for e in history if key in e]
        plt.figure()
        plt.plot(values, label=key)
        plt.xlabel('Epoch')
        plt.ylabel('AP')
        plt.title(key)
        plt.grid(True)
        plt.savefig(f'{model_name}_{key.replace(" ", "_")}.png')
        plt.close()
            
def train_model(model, optimizer, scheduler, num_epochs, dataloaders, device, model_name):
    eval_history = []
    best_map = -1.0
    best_epoch = None
    best_model_path = None
    
    # Weight for components of the loss function.
    # Ball-related loss and player-related loss are mean losses (loss per one positive example)
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
    if is_validation_set:
        phases = ['train', 'val']
    else:
        phases = ['train']

    # Training statistics
    training_stats = {'train': [], 'val': []}

    print('Training...')
    for epoch in tqdm.tqdm(range(num_epochs)):
        # Each epoch has a training and validation phase
        # for phase in ['train', 'val']:
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            batch_stats = {'loss': [], 'loss_ball_c': [], 'loss_player_c': [], 'loss_player_l': []}

            #count_batches = 0
            #print("📦 Preparing to load first batch...")
            # Iterate over data.
            for ndx, (images, boxes, labels) in enumerate(dataloaders[phase]):
                #print(f"✅ Batch {ndx} loaded")
                images = images.to(device)
                h, w = images.shape[-2], images.shape[-1]
                gt_maps = model.groundtruth_maps(boxes, labels, (h, w))
                gt_maps = [e.to(device) for e in gt_maps]
                #count_batches += 1

                with torch.set_grad_enabled(phase == 'train'):
                    predictions = model(images)
                    # Backpropagation
                    optimizer.zero_grad()
                    loss_l_player, loss_c_player, loss_c_ball = criterion(predictions, gt_maps)

                    loss = alpha_l_player * loss_l_player + alpha_c_player * loss_c_player + alpha_c_ball * loss_c_ball

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                #count_batches += 1
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

        # Scheduler step
        scheduler.step()

        if is_validation_set:
            print("Running evaluation on validation set...")
            model.phase = 'detect'  # 切换到 detect 模式
            eval_results = eval_model(model, dataloaders['val'], device)
            model.phase = 'train'   # 还原回来
            eval_history.append(eval_results)

            if eval_results['mAP'] > best_map:
                best_map = eval_results['mAP']
                best_epoch = epoch + 1  # epoch 是从 0 开始的，加 1 更直观
                best_model_path = os.path.join(MODEL_FOLDER, model_name + '_best.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved new best model: {best_model_path} with mAP {best_map:.4f}")
    
                with open(f'best_model_info_{model_name}.txt', 'w') as f:
                    f.write(f"Best mAP: {best_map:.4f} at epoch {best_epoch}\n")

    model_filepath = os.path.join(MODEL_FOLDER, model_name + '_final' + '.pth')
    torch.save(model.state_dict(), model_filepath)

    with open('training_stats_{}.pickle'.format(model_name), 'wb') as handle:
        pickle.dump(training_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Draw and save loss graphs
    plot_losses(training_stats, model_name)

    if is_validation_set:
        with open(f'eval_history_{model_name}.pickle', 'wb') as f:
            pickle.dump(eval_history, f)
        plot_eval_history(eval_history, model_name)
        
    return training_stats


def train(params: Params):
    if not os.path.exists(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)

    assert os.path.exists(MODEL_FOLDER), ' Cannot create folder to save trained model: {}'.format(MODEL_FOLDER)

    dataloaders = make_dataloaders(params)
    print('Training set: Dataset size: {}'.format(len(dataloaders['train'].dataset)))
    if 'val' in dataloaders:
        print('Validation set: Dataset size: {}'.format(len(dataloaders['val'].dataset)))

    # Create model
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = footandball.model_factory(params.model, 'train')
    model.print_summary(show_architecture=True)
    model = model.to(device)

    model_name = 'model_' + time.strftime("%Y%m%d_%H%M")
    print('Model name: {}'.format(model_name))

    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    scheduler_milestones = [int(params.epochs * 0.75)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, scheduler_milestones, gamma=0.1)
    
    return train_model(model, optimizer, scheduler, params.epochs, dataloaders, device, model_name)


if __name__ == '__main__':
    print('Train FoootAndBall detector on ISSIA dataset')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to the configuration file', type=str, default='config.txt')
    parser.add_argument('--debug', dest='debug', help='debug mode', action='store_true')
    args = parser.parse_args()

    print('Config path: {}'.format(args.config))
    print('Debug mode: {}'.format(args.debug))

    params = Params(args.config)
    params.print()

    train(params)


'''
NOTES:
In camera 5 some of the player bboxes are moved by a few pixels from the true position.
When evaluating mean precision use smaller IoU ratio, otherwise detection results are poor.
Alternatively add some margin to ISSIA ground truth bboxes.
'''
