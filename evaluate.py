import torch
import numpy as np
import torchvision
from torchvision.ops import box_iou
from collections import defaultdict
import sys
import typing as t
import xml.etree.ElementTree as ET
import numpy as np
import torch
from sklearn.metrics import average_precision_score
import numpy as np
from data.augmentation import PLAYER_LABEL, BALL_LABEL, BALL_BBOX_SIZE

def compute_ap_map(detections, ground_truths, iou_threshold=0.5, num_classes=2):
    """
    detections: list of dicts with keys ['boxes', 'scores', 'labels']
    ground_truths: list of dicts with keys ['boxes', 'labels']
    returns: dict {class_id: ap_value, ..., 'mAP': float}
    """
    aps = {}
    for class_id in [BALL_LABEL, PLAYER_LABEL]:  # [1, 2]
        y_true = []
        y_scores = []

        for det, gt in zip(detections, ground_truths):
            det_boxes = [b for b, l in zip(det["boxes"], det["labels"]) if l == class_id]
            det_scores = [s for s, l in zip(det["scores"], det["labels"]) if l == class_id]
            gt_boxes = [b for b, l in zip(gt["boxes"], gt["labels"]) if l == class_id]
            #print(f"Class {class_id}: {len(det_boxes)} detections vs {len(gt_boxes)} GT")
            matched = [False] * len(gt_boxes)

            for box, score in zip(det_boxes, det_scores):
                iou_max = 0.0
                matched_idx = -1
                for i, gt_box in enumerate(gt_boxes):
                    iou = IoU_box(box, gt_box)
                    if iou > iou_max:
                        iou_max = iou
                        matched_idx = i
                if iou_max >= iou_threshold and not matched[matched_idx]:
                    y_true.append(1)
                    matched[matched_idx] = True
                else:
                    y_true.append(0)
                y_scores.append(score)

            for m in matched:
                if not m:
                    y_true.append(1)
                    y_scores.append(0)

        if len(y_true) == 0 or len(set(y_true)) == 1:
            aps[class_id] = 0.0
        else:
            #aps[class_id] = average_precision_score(y_true, y_scores)
            aps[class_id] = average_precision_score(y_true, [s.cpu().item() if torch.is_tensor(s) else s for s in y_scores])

    aps['mAP'] = np.mean(list(aps.values()))
    return aps

def getGT(xgtf_path: str) -> t.List[t.Dict[str, torch.Tensor]]:
    """
    Parses the .xgtf ground truth file and returns a list of frame-wise dicts.
    Each dict contains 'boxes': Tensor[N,4], 'labels': Tensor[N]
    """
    tree = ET.parse(xgtf_path)
    root = tree.getroot()
    gt_by_frame = {}

    for obj in root.findall('.//{http://lamp.cfar.umd.edu/viper#}object'):
        name = obj.get('name')
        framespan = obj.get('framespan')
        if framespan is None:
            continue
        frame_start, frame_end = map(int, framespan.split(':'))

        if name == "BALL":
            for attr in obj.findall('.//{http://lamp.cfar.umd.edu/viper#}attribute[@name="BallPos"]'):
                for point in attr.findall('{http://lamp.cfar.umd.edu/viperdata#}point'):
                    frame_id = int(point.attrib['framespan'].split(':')[0])
                    x = int(point.attrib['x'])
                    y = int(point.attrib['y'])
                    half = BALL_BBOX_SIZE // 2
                    box = [x - half, y - half, x + half, y + half]

                    if frame_id not in gt_by_frame:
                        gt_by_frame[frame_id] = {'boxes': [], 'labels': []}
                    gt_by_frame[frame_id]['boxes'].append(box)
                    gt_by_frame[frame_id]['labels'].append(BALL_LABEL)

        elif name == "Person":
            for attr in obj.findall('.//{http://lamp.cfar.umd.edu/viper#}attribute[@name="LOCATION"]'):
                for bbox in attr.findall('{http://lamp.cfar.umd.edu/viperdata#}bbox'):
                    frame_id = int(bbox.attrib['framespan'].split(':')[0])
                    x = int(bbox.attrib['x'])
                    y = int(bbox.attrib['y'])
                    width = int(bbox.attrib['width'])
                    height = int(bbox.attrib['height'])
                    box = [x, y, x + width, y + height]

                    if frame_id not in gt_by_frame:
                        gt_by_frame[frame_id] = {'boxes': [], 'labels': []}
                    gt_by_frame[frame_id]['boxes'].append(box)
                    gt_by_frame[frame_id]['labels'].append(PLAYER_LABEL)

    ground_truths = []
    frame_ids = sorted(gt_by_frame.keys())
    #print(f"[DEBUG] Ground truth frame range: {frame_ids[0]} ~ {frame_ids[-1]}")

    for frame_id in sorted(gt_by_frame.keys()):
        boxes = [list(map(float, box)) for box in gt_by_frame[frame_id]['boxes']]
        labels = list(gt_by_frame[frame_id]['labels'])
        ground_truths.append({'boxes': boxes, 'labels': labels})
    return ground_truths, frame_ids[0]



def average_precision(pred_boxes, gt_boxes, iou_threshold=0.5):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        print("Warning: empty prediction or ground truth for this class")
        return 0.0

    pred_boxes = sorted(pred_boxes, key=lambda x: -x[1])
    pred_boxes_tensor = torch.as_tensor([np.array(p[0]) for p in pred_boxes], dtype=torch.float32)
    scores = [p[1] for p in pred_boxes]
    pred_labels = [p[2] for p in pred_boxes]

    gt_boxes_tensor = torch.as_tensor([np.array(g[0]) for g in gt_boxes], dtype=torch.float32)
    gt_labels = [g[1] for g in gt_boxes]

    iou_matrix = box_iou(pred_boxes_tensor, gt_boxes_tensor)
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    matched_gt = set()

    for i in range(len(pred_boxes)):
        best_iou = 0.0
        best_j = -1
        for j in range(len(gt_boxes)):
            if j in matched_gt or pred_labels[i] != gt_labels[j]:
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
    recalls = cum_tp / max(len(gt_boxes), 1)  # 修复点

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

    results = {}
    for label_id, label_name in [(BALL_LABEL, 'Ball AP'), (PLAYER_LABEL, 'Player AP')]:
        pred_cls = [p for img_p in all_pred for p in img_p if p[2] == label_id]
        gt_cls = [g for img_g in all_gt for g in img_g if g[1] == label_id]
        ap = average_precision(pred_cls, gt_cls)
        results[label_name] = ap
        print(f"{label_name}: {ap:.4f}")
    '''
    aps = []
    for label_id in [BALL_LABEL, PLAYER_LABEL]:
        pred_cls = [p for img_p in all_pred for p in img_p if p[2] == label_id]
        gt_cls = [g for img_g in all_gt for g in img_g if g[1] == label_id]
        aps.append(average_precision(pred_cls, gt_cls))
    results['mAP'] = np.mean(aps)
    '''
    aps = [results['Ball AP'], results['Player AP']]
    results['mAP'] = np.mean(aps)

    print(f"mAP: {results['mAP']:.4f}")
    return results
