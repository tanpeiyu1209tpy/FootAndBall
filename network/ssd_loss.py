import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    Args:
        loss (N, n): the loss for each example.
        labels (N, n): the labels.
        neg_pos_ratio: the ratio between the negative and positive examples.
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_pos[num_pos == 0] = 1  # avoid zero division
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf  # ignore positives for mining negatives
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask, torch.sum(num_pos).item()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits, shape (N, C)
        # targets: class indices, shape (N,)
        logpt = -F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(logpt)  # pt is probability of true class
        focal_loss = -self.alpha * (1 - pt) ** self.gamma * logpt

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SSDLoss(nn.Module):
    def __init__(self, neg_pos_ratio, gamma=2.0, alpha=0.25):
        super(SSDLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha, reduction='sum')

    def forward(self, predictions, targets):
        """
        Args:
            predictions: tuple of (player_loc, player_conf, ball_conf)
                player_loc: (batch, n_player, 4)
                player_conf: (batch, n_player, 2)
                ball_conf: (batch, n_ball, 2)
            targets: tuple of (player_loc_t, player_conf_t, ball_conf_t)
                player_loc_t: (batch, n_player, 4)
                player_conf_t: (batch, n_player)
                ball_conf_t: (batch, n_ball)
        Returns:
            player_loss_l: localization loss (Smooth L1)
            player_loss_c: classification loss (Focal Loss) for players
            ball_loss_c: classification loss (Focal Loss) for balls
        """
        player_loc, player_conf, ball_conf = predictions
        player_loc_t, player_conf_t, ball_conf_t = targets
        batch_size = player_loc.shape[0]

        # reshape as needed
        player_loc = player_loc.view(batch_size, -1, 4)
        player_conf = player_conf.view(batch_size, -1, 2)
        ball_conf = ball_conf.view(batch_size, -1, 2)
        player_loc_t = player_loc_t.view(batch_size, -1, 4)
        player_conf_t = player_conf_t.view(batch_size, -1)
        ball_conf_t = ball_conf_t.view(batch_size, -1)

        with torch.no_grad():
            loss_player = -F.log_softmax(player_conf, dim=2)[:, :, 0]
            mask_player, num_pos_player = hard_negative_mining(loss_player, player_conf_t, self.neg_pos_ratio)
            loss_ball = -F.log_softmax(ball_conf, dim=2)[:, :, 0]
            mask_ball, num_pos_ball = hard_negative_mining(loss_ball, ball_conf_t, self.neg_pos_ratio)

        player_conf = player_conf[mask_player, :]
        ball_conf = ball_conf[mask_ball, :]

        player_loss_c = self.focal_loss(player_conf, player_conf_t[mask_player])
        ball_loss_c = self.focal_loss(ball_conf, ball_conf_t[mask_ball])

        pos_mask_player = player_conf_t > 0
        player_loc = player_loc[pos_mask_player, :].view(-1, 4)
        player_loc_t = player_loc_t[pos_mask_player, :].view(-1, 4)
        player_loss_l = F.smooth_l1_loss(player_loc, player_loc_t, reduction='sum')

        return player_loss_l / num_pos_player, player_loss_c / num_pos_player, ball_loss_c / num_pos_ball
