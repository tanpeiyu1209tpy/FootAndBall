import torch
import numpy as np
import torchvision
from torchvision.ops import box_iou
from collections import defaultdict
from data.augmentation import PLAYER_LABEL, BALL_LABEL

def average_precision(pred_boxes, gt_boxes, iou_threshold=0.5):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0

    # Sort predictions by descending confidence
    pred_boxes = sorted(pred_boxes, key=lambda x: -x[1])
    pred_boxes_tensor = torch.tensor([p[0] for p in pred_boxes], dtype=torch.float32)
    scores = [p[1] for p in pred_boxes]
    pred_labels = [p[2] for p in pred_boxes]

    gt_boxes_tensor = torch.tensor([g[0] for g in gt_boxes], dtype=torch.float32)
    gt_labels = [g[1] for g in gt_boxes]

    # Compute IoU matrix
    iou_matrix = box_iou(pred_boxes_tensor, gt_boxes_tensor)  # shape: (N_pred, N_gt)

    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    matched_gt = set()

    for i in range(len(pred_boxes)):
        best_iou = 0.0
        best_j = -1
        for j in range(len(gt_boxes)):
            if j in matched_gt:
                continue
            if pred_labels[i] != gt_labels[j]:
                continue
            iou = iou_matrix[i, j].item()
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            tp[i] = 1
            matched_gt.add(best_j)
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-6)
    recalls = cum_tp / len(gt_boxes)
    
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0
        ap += p / 11
    return ap


def eval_model(model, dataloader, device):
    model.eval()
    model.phase = 'detect'

    all_pred = []
    all_gt = []

    with torch.no_grad():
        for images, boxes, labels in dataloader:
            images = images.to(device)
            detections = model(images)
            for i in range(len(detections)):
                preds = detections[i]
                pred_boxes = preds['boxes'].cpu().numpy()
                scores = preds['scores'].cpu().numpy()
                cls = preds['labels'].cpu().numpy()
                pred = [(pred_boxes[j], scores[j], cls[j]) for j in range(len(pred_boxes))]
                all_pred.append(pred)

                gt = [(boxes[i][j].numpy(), labels[i][j].item()) for j in range(len(labels[i]))]
                all_gt.append(gt)

                if i == 0:
                    print("第一张图的第一个预测框：", pred_boxes[0])
                    print("第一张图的第一个GT框：", boxes[i][0])

    results = {}
    for label_id, label_name in [(BALL_LABEL, 'Ball AP'), (PLAYER_LABEL, 'Player AP')]:
        pred_cls = [p for img_p in all_pred for p in img_p if p[2] == label_id]
        gt_cls = [g for img_g in all_gt for g in img_g if g[1] == label_id]
        ap = average_precision(pred_cls, gt_cls)
        results[label_name] = ap
        print(f"{label_name}: {ap:.4f}")

    aps = []
    for label_id in [BALL_LABEL, PLAYER_LABEL]:
        pred_cls = [p for img_p in all_pred for p in img_p if p[2] == label_id]
        gt_cls = [g for img_g in all_gt for g in img_g if g[1] == label_id]
        aps.append(average_precision(pred_cls, gt_cls))
    results['mAP'] = np.mean(aps)
    print(f"mAP: {results['mAP']:.4f}")
    return results
