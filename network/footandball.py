import torch
import torch.nn as nn

import network.fpn as fpn
import network.nms as nms
from data.augmentation import BALL_LABEL, PLAYER_LABEL, BALL_BBOX_SIZE

'''
class TemporalFusion(nn.Module):
    """简单的temporal fusion模块"""
    def __init__(self, channels=32, method='difference'):
        super().__init__()
        self.method = method
        self.channels = channels
        
    def forward(self, features, batch_size, temporal_window):
        """
        features: [B*T, C, H, W] - FPN输出的特征
        batch_size: B 
        temporal_window: T
        Returns: [B, C, H, W]
        """
        if temporal_window == 1:
            return features
            
        # Reshape to [B, T, C, H, W]
        BT, C, H, W = features.shape
        features = features.view(batch_size, temporal_window, C, H, W)
        
        if self.method == 'difference':
            # 帧间差异 - 突出运动，抑制静态广告牌等
            diff_maps = []
            for i in range(1, temporal_window):
                diff = torch.abs(features[:, i] - features[:, i-1])
                diff_maps.append(diff)
            
            if len(diff_maps) == 0:
                fused = features[:, 0]  # 单帧情况
            else:
                # 平均所有差异 + 保留一定的原始特征
                diff_avg = torch.mean(torch.stack(diff_maps, dim=1), dim=1)
                current_frame = features[:, -1]  # 最新帧
                fused = 0.7 * diff_avg + 0.3 * current_frame
                
        elif self.method == 'variance':
            # 时间方差 - 运动区域方差大，静态区域方差小
            fused = torch.var(features, dim=1)
            
        elif self.method == 'weighted_avg':
            # 加权平均 - 最新帧权重最高
            weights = torch.linspace(0.5, 1.0, temporal_window).to(features.device)
            weights = weights.view(1, temporal_window, 1, 1, 1)
            weighted_features = features * weights
            fused = torch.mean(weighted_features, dim=1)
            
        else:
            # 默认：简单平均
            fused = torch.mean(features, dim=1)
            
        return fused  # [B, C, H, W]
'''
class TemporalFusion(nn.Module):
    def __init__(self, channels=32, method='difference'):
        super().__init__()
        self.method = method
        self.channels = channels

        if self.method == 'attention':
            # 简化的 attention 融合
            self.query = nn.Linear(channels, channels)
            self.key = nn.Linear(channels, channels)
            self.value = nn.Linear(channels, channels)
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, features, batch_size, temporal_window):
        """
        features: [B*T, C, H, W]
        Returns: [B, C, H, W]
        """
        if temporal_window == 1:
            return features.view(batch_size, *features.shape[1:])

        BT, C, H, W = features.shape
        features = features.view(batch_size, temporal_window, C, H, W)

        if self.method == 'difference':
            diff_maps = []
            for i in range(1, temporal_window):
                diff = torch.abs(features[:, i] - features[:, i-1])
                diff_maps.append(diff)
            if len(diff_maps) == 0:
                fused = features[:, 0]
            else:
                diff_avg = torch.mean(torch.stack(diff_maps, dim=1), dim=1)
                current_frame = features[:, -1]
                fused = 0.7 * diff_avg + 0.3 * current_frame

        elif self.method == 'variance':
            fused = torch.var(features, dim=1)

        elif self.method == 'weighted_avg':
            weights = torch.linspace(0.5, 1.0, temporal_window).to(features.device)
            weights = weights.view(1, temporal_window, 1, 1, 1)
            weighted_features = features * weights
            fused = torch.mean(weighted_features, dim=1)

        elif self.method == 'attention':
            # reshape to [B, T, C, H*W]
            features_flat = features.view(batch_size, temporal_window, C, -1)  # [B, T, C, HW]
            features_flat = features_flat.permute(0, 3, 1, 2)  # [B, HW, T, C]

            Q = self.query(features_flat)  # [B, HW, T, C]
            K = self.key(features_flat)
            V = self.value(features_flat)

            attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.channels ** 0.5)  # [B, HW, T, T]
            attention_weights = self.softmax(attention_scores)  # [B, HW, T, T]
            attended = torch.matmul(attention_weights, V)  # [B, HW, T, C]
            attended = attended.sum(dim=2)  # [B, HW, C]

            fused = attended.permute(0, 2, 1).view(batch_size, C, H, W)  # [B, C, H, W]

        else:
            fused = torch.mean(features, dim=1)

        return fused

# 保持原有的辅助函数不变...
def get_active_cells(bbox_center_x, bbox_center_y, downsampling_factor, conf_width, conf_height, delta):
    cell_x = int(bbox_center_x / downsampling_factor)
    cell_y = int(bbox_center_y / downsampling_factor)
    x1 = max(cell_x - delta // 2, 0)
    x2 = min(cell_x + delta // 2, conf_width - 1)
    y1 = max(cell_y - delta // 2, 0)
    y2 = min(cell_y + delta // 2, conf_height - 1)
    return x1, y1, x2, y2


def cell2pixel(cell_x, cell_y, downsampling_factor):
    x1 = cell_x * downsampling_factor
    x2 = cell_x * downsampling_factor + downsampling_factor - 1
    y1 = cell_y * downsampling_factor
    y2 = cell_y * downsampling_factor + downsampling_factor - 1
    return x1, y1, x2, y2


def create_groundtruth_maps(bboxes, blabels, img_shape, player_downsampling_factor, ball_downsampling_factor,
                            player_delta, ball_delta):
    # 保持原有实现不变...
    num = len(bboxes)
    h, w = img_shape
    ball_conf_height = h // ball_downsampling_factor
    ball_conf_width = w // ball_downsampling_factor
    player_conf_height = h // player_downsampling_factor
    player_conf_width = w // player_downsampling_factor

    player_loc_t = torch.zeros([num, player_conf_height, player_conf_width, 4], dtype=torch.float)
    player_conf_t = torch.zeros([num, player_conf_height, player_conf_width], dtype=torch.long)
    ball_conf_t = torch.zeros([num, ball_conf_height, ball_conf_width], dtype=torch.long)

    for idx, (boxes, labels) in enumerate(zip(bboxes, blabels)):
        for box, label in zip(boxes, labels):
            bbox_center_x = (box[0] + box[2]) / 2.
            bbox_center_y = (box[1] + box[3]) / 2.
            bbox_width = box[2] - box[0]
            bbox_height = box[3] - box[1]

            if label == BALL_LABEL:
                x1, y1, x2, y2 = get_active_cells(bbox_center_x, bbox_center_y, ball_downsampling_factor,
                                                  ball_conf_width, ball_conf_height, ball_delta)
                ball_conf_t[idx, y1:y2 + 1, x1:x2 + 1] = 1
            elif label == PLAYER_LABEL:
                x1, y1, x2, y2 = get_active_cells(bbox_center_x, bbox_center_y, player_downsampling_factor,
                                                  player_conf_width, player_conf_height, player_delta)
                player_conf_t[idx, y1:y2 + 1, x1:x2 + 1] = 1

                temp_x = torch.tensor(range(x1, x2 + 1)).float() * player_downsampling_factor + \
                         (player_downsampling_factor - 1) / 2
                temp_y = torch.tensor(range(y1, y2 + 1)).float() * player_downsampling_factor + \
                         (player_downsampling_factor - 1) / 2

                temp_x = (bbox_center_x - temp_x) / w
                temp_y = (bbox_center_y - temp_y) / h

                player_loc_t[idx, y1:y2 + 1, x1: x2+1, 0] = temp_x.unsqueeze(0)
                player_loc_t[idx, y1:y2 + 1, x1: x2+1, 1] = temp_y.unsqueeze(1)
                player_loc_t[idx, y1:y2 + 1, x1: x2+1, 2] = bbox_width / w
                player_loc_t[idx, y1:y2 + 1, x1: x2+1, 3] = bbox_height / h

    return player_loc_t, player_conf_t, ball_conf_t


def count_parameters(model):
    if model is None:
        return 0, 0
    else:
        ap = sum(p.numel() for p in model.parameters())
        tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return ap, tp


class FootAndBall(nn.Module):
    def __init__(self, phase, base_network: nn.Module, player_regressor: nn.Module, player_classifier: nn.Module,
                 ball_classifier: nn.Module, max_player_detections=100, max_ball_detections=100, player_threshold=0.0,
                 ball_threshold=0.0, use_temporal_fusion=False, temporal_window=3, fusion_method='difference'):
        super(FootAndBall, self).__init__()

        assert phase in ['train', 'eval', 'detect']

        self.phase = phase
        self.base_network = base_network
        self.ball_classifier = ball_classifier
        self.player_classifier = player_classifier
        self.player_regressor = player_regressor
        self.max_player_detections = max_player_detections
        self.max_ball_detections = max_ball_detections
        self.player_threshold = player_threshold
        self.ball_threshold = ball_threshold

        # Temporal fusion 参数
        self.use_temporal_fusion = use_temporal_fusion
        self.temporal_window = temporal_window
        
        if self.use_temporal_fusion:
            # 为球和球员特征分别创建temporal fusion模块
            self.temporal_fusion_ball = TemporalFusion(channels=32, method=fusion_method)
            self.temporal_fusion_player = TemporalFusion(channels=32, method=fusion_method)
            print(f"✅ Temporal fusion enabled: {fusion_method}, window={temporal_window}")

        self.ball_downsampling_factor = 4
        self.player_downsampling_factor = 16
        self.ball_delta = 3
        self.player_delta = 3

        self.softmax = nn.Softmax(dim=1)
        self.nms_kernel_size = (3, 3)
        self.nms = nms.NonMaximaSuppression2d(self.nms_kernel_size)

    # 保持原有的detect_from_map, detect, groundtruth_maps方法不变...
    def detect_from_map(self, confidence_map, downscale_factor, max_detections, bbox_map=None):
        confidence_map = self.nms(confidence_map)[:, 1]
        batch_size, h, w = confidence_map.shape[0], confidence_map.shape[1], confidence_map.shape[2]
        confidence_map = confidence_map.view(batch_size, -1)

        values, indices = torch.sort(confidence_map, dim=-1, descending=True)
        if max_detections < indices.shape[1]:
            indices = indices[:, :max_detections]

        xc = indices % w
        xc = xc.float() * downscale_factor + (downscale_factor - 1.) / 2.

        yc = torch.div(indices, w, rounding_mode='trunc')
        yc = yc.float() * downscale_factor + (downscale_factor - 1.) / 2.

        if bbox_map is not None:
            bbox_map = bbox_map.view(batch_size, 4, -1)
            bbox_map[:, 0] *= w * downscale_factor
            bbox_map[:, 2] *= w * downscale_factor
            bbox_map[:, 1] *= h * downscale_factor
            bbox_map[:, 3] *= h * downscale_factor
        else:
            batch_size, h, w = confidence_map.shape[0], confidence_map.shape[-2], confidence_map.shape[-1]
            bbox_map = torch.zeros((batch_size, 4, h * w), dtype=torch.float).to(confidence_map.device)
            bbox_map[:, [2, 3]] = BALL_BBOX_SIZE

        detections = torch.zeros((batch_size, max_detections, 5), dtype=float).to(confidence_map.device)

        for n in range(batch_size):
            temp = bbox_map[n, :, indices[n]]
            bx = xc[n] + temp[0]
            by = yc[n] + temp[1]

            detections[n, :, 0] = bx - 0.5 * temp[2]
            detections[n, :, 2] = bx + 0.5 * temp[2]
            detections[n, :, 1] = by - 0.5 * temp[3]
            detections[n, :, 3] = by + 0.5 * temp[3]
            detections[n, :, 4] = values[n, :max_detections]

        return detections

    def detect(self, player_feature_map, player_bbox, ball_feature_map):
        player_detections = self.detect_from_map(player_feature_map, self.player_downsampling_factor,
                                                 self.max_player_detections, player_bbox)

        ball_detections = self.detect_from_map(ball_feature_map, self.ball_downsampling_factor,
                                               self.max_ball_detections)

        output = []
        for player_det, ball_det in zip(player_detections, ball_detections):
            player_det = player_det[player_det[..., 4] >= self.player_threshold]
            player_boxes = player_det[..., 0:4]
            player_scores = player_det[..., 4]
            player_labels = torch.tensor([PLAYER_LABEL] * len(player_det), dtype=torch.int64)
            ball_det = ball_det[ball_det[..., 4] >= self.ball_threshold]
            ball_boxes = ball_det[..., 0:4]
            ball_scores = ball_det[..., 4]
            ball_labels = torch.tensor([BALL_LABEL] * len(ball_det), dtype=torch.int64)

            boxes = torch.cat([player_boxes, ball_boxes], dim=0)
            scores = torch.cat([player_scores, ball_scores], dim=0)
            labels = torch.cat([player_labels, ball_labels], dim=0)

            temp = {'boxes': boxes, 'labels': labels, 'scores': scores}
            output.append(temp)

        return output

    def groundtruth_maps(self, boxes, labels, img_shape):
        player_loc_t, player_conf_t, ball_conf_t = create_groundtruth_maps(boxes, labels, img_shape,
                                                                           self.player_downsampling_factor,
                                                                           self.ball_downsampling_factor,
                                                                           self.player_delta, self.ball_delta)
        return player_loc_t, player_conf_t, ball_conf_t

    def forward(self, x):
        # 处理temporal input: x可以是[B, 3, H, W]或[B, T, 3, H, W]
        if x.dim() == 5:  # Temporal input [B, T, 3, H, W]
            B, T = x.shape[:2]
            temporal_mode = True
            x = x.view(B*T, *x.shape[2:])  # [B*T, 3, H, W]
        else:  # Single frame [B, 3, H, W]
            B, T = x.shape[0], 1
            temporal_mode = False

        height, width = x.shape[2], x.shape[3]

        # FPN backbone处理
        x = self.base_network(x)
        
        # 在这里应用temporal fusion
        if self.use_temporal_fusion and temporal_mode and T > 1:
            # x[0]: 球特征图 [B*T, 32, H/4, W/4]
            # x[1]: 球员特征图 [B*T, 32, H/16, W/16]
            x[0] = self.temporal_fusion_ball(x[0], B, T)     # → [B, 32, H/4, W/4]
            x[1] = self.temporal_fusion_player(x[1], B, T)   # → [B, 32, H/16, W/16]
        
        # 如果是单帧模式，确保输出维度正确
        if not temporal_mode or T == 1:
            if temporal_mode:  # [B*1, ...] → [B, ...]
                x[0] = x[0].view(B, *x[0].shape[1:])
                x[1] = x[1].view(B, *x[1].shape[1:])

        # 验证dimensions
        assert len(x) == 2
        assert x[0].shape[0] == x[1].shape[0] == B  # batch size一致
        assert x[0].shape[1] == x[1].shape[1]       # channels一致
        assert x[0].shape[2] == height // self.ball_downsampling_factor
        assert x[0].shape[3] == width // self.ball_downsampling_factor
        assert x[1].shape[2] == height // self.player_downsampling_factor
        assert x[1].shape[3] == width // self.player_downsampling_factor

        # 分类和回归
        ball_feature_map = self.ball_classifier(x[0])
        player_feature_map = self.player_classifier(x[1])
        player_bbox = self.player_regressor(x[1])

        if self.phase == 'eval' or self.phase == 'detect':
            player_feature_map = self.softmax(player_feature_map)
            ball_feature_map = self.softmax(ball_feature_map)

        if self.phase == 'train' or self.phase == 'eval':
            ball_feature_map = ball_feature_map.permute(0, 2, 3, 1).contiguous()
            player_feature_map = player_feature_map.permute(0, 2, 3, 1).contiguous()
            player_bbox = player_bbox.permute(0, 2, 3, 1).contiguous()
            output = (player_bbox, player_feature_map, ball_feature_map)
        elif self.phase == 'detect':
            output = self.detect(player_feature_map, player_bbox, ball_feature_map)

        return output

    def print_summary(self, show_architecture=True):
        if show_architecture:
            print('Base network:')
            print(self.base_network)
            if self.ball_classifier is not None:
                print('Ball classifier:')
                print(self.ball_classifier)
            if self.player_classifier is not None:
                print('Player classifier:')
                print(self.player_classifier)

        ap, tp = count_parameters(self.base_network)
        print('Base network parameters (all/trainable): {}/{}'.format(ap, tp))

        if self.ball_classifier is not None:
            ap, tp = count_parameters(self.ball_classifier)
            print('Ball classifier parameters (all/trainable): {}/{}'.format(ap, tp))

        if self.player_classifier is not None:
            ap, tp = count_parameters(self.player_classifier)
            print('Player classifier parameters (all/trainable): {}/{}'.format(ap, tp))

        if self.player_regressor is not None:
            ap, tp = count_parameters(self.player_regressor)
            print('Player regressor parameters (all/trainable): {}/{}'.format(ap, tp))

        if hasattr(self, 'temporal_fusion_ball') and self.temporal_fusion_ball is not None:
            ap, tp = count_parameters(self.temporal_fusion_ball)
            print('Temporal fusion ball parameters (all/trainable): {}/{}'.format(ap, tp))

        if hasattr(self, 'temporal_fusion_player') and self.temporal_fusion_player is not None:
            ap, tp = count_parameters(self.temporal_fusion_player)
            print('Temporal fusion player parameters (all/trainable): {}/{}'.format(ap, tp))

        ap, tp = count_parameters(self)
        print('Total (all/trainable): {} / {}'.format(ap, tp))
        print('')


def build_footandball_detector1(phase='train', max_player_detections=100, max_ball_detections=100,
                                player_threshold=0.0, ball_threshold=0.0, use_temporal_fusion=False, 
                                temporal_window=3, fusion_method='difference'):
    assert phase in ['train', 'test', 'detect']

    layers, out_channels = fpn.make_modules(fpn.cfg['X'], batch_norm=True)
    lateral_channels = 32
    i_channels = 32

    base_net = fpn.FPN(layers, out_channels=out_channels, lateral_channels=lateral_channels, return_layers=[1, 3])
    ball_classifier = nn.Sequential(nn.Conv2d(lateral_channels, out_channels=i_channels, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(i_channels, out_channels=2, kernel_size=3, padding=1))
    player_classifier = nn.Sequential(nn.Conv2d(lateral_channels, out_channels=i_channels, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(i_channels, out_channels=2, kernel_size=3, padding=1))
    player_regressor = nn.Sequential(nn.Conv2d(lateral_channels, out_channels=i_channels, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(i_channels, out_channels=4, kernel_size=3, padding=1))
    
    detector = FootAndBall(phase, base_net, player_regressor=player_regressor, player_classifier=player_classifier,
                           ball_classifier=ball_classifier, ball_threshold=ball_threshold,
                           player_threshold=player_threshold, max_ball_detections=max_ball_detections,
                           max_player_detections=max_player_detections, use_temporal_fusion=use_temporal_fusion,
                           temporal_window=temporal_window, fusion_method=fusion_method)
    return detector


def model_factory(model_name, phase, max_player_detections=100, max_ball_detections=100, player_threshold=0.0,
                  ball_threshold=0.0, use_temporal_fusion=False, temporal_window=3, fusion_method='difference'):
    if model_name == 'fb1':
        model_fn = build_footandball_detector1
    else:
        print('Model not implemented: {}'.format(model_name))
        raise NotImplementedError

    return model_fn(phase, ball_threshold=ball_threshold, player_threshold=player_threshold,
                    max_ball_detections=max_ball_detections, max_player_detections=max_player_detections,
                    use_temporal_fusion=use_temporal_fusion, temporal_window=temporal_window, 
                    fusion_method=fusion_method)


if __name__ == '__main__':
    net = model_factory('fb1', 'train', use_temporal_fusion=True, temporal_window=3)
    net.print_summary()

    # 测试单帧
    x = torch.zeros((2, 3, 1024, 1024))
    x = net(x)
    print("Single frame output shapes:")
    for t in x:
        print(t.shape)

    # 测试多帧
    x_temporal = torch.zeros((2, 3, 3, 1024, 1024))  # [B, T, 3, H, W]
    x_temporal = net(x_temporal)
    print("Temporal output shapes:")
    for t in x_temporal:
        print(t.shape)

    print('✅ Temporal FootAndBall test completed!')
