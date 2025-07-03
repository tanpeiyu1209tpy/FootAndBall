import torch
import numpy as np
from collections import defaultdict
from data.augmentation import PLAYER_LABEL, BALL_LABEL
from network.nms import compute_iou


def average_precision(pred_boxes, gt_boxes, iou_threshold=0.5):
    if len(pred_boxes) == 0:
        return 0.0
    pred_boxes = sorted(pred_boxes, key=lambda x: -x[1])  # sort by confidence
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    matched = set()

    for i, (p_box, p_score, p_label) in enumerate(pred_boxes):
        found = False
        for j, (g_box, g_label) in enumerate(gt_boxes):
            if j in matched or p_label != g_label:
                continue
            iou = compute_iou(p_box, g_box)
            if iou >= iou_threshold:
                found = True
                matched.add(j)
                break
        tp[i] = 1 if found else 0
        fp[i] = 0 if found else 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-6)
    recalls = cum_tp / len(gt_boxes)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = np.max(precisions[recalls >= t]) if np.sum(recalls >= t) > 0 else 0
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
