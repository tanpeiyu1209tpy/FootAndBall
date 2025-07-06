import torch
import numpy as np
import torchvision
from torchvision.ops import box_iou
from collections import defaultdict
import typing as t
import xml.etree.ElementTree as ET
from sklearn.metrics import average_precision_score
from data.augmentation import PLAYER_LABEL, BALL_LABEL, BALL_BBOX_SIZE

def IoU_box(box1, box2):
    """Compute IoU between two boxes in [x1, y1, x2, y2] format"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = max(0, (box1[2] - box1[0])) * max(0, (box1[3] - box1[1]))
    box2_area = max(0, (box2[2] - box2[0])) * max(0, (box2[3] - box2[1]))
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area
    
def compute_ap_map(detections, ground_truths, iou_threshold=0.5):
    """
    detections: list of dicts with keys ['boxes', 'scores', 'labels']
    ground_truths: list of dicts with keys ['boxes', 'labels']
    returns: dict {class_id: ap_value, ..., 'mAP': float}
    """
    aps = {}
    
    for class_id in [BALL_LABEL, PLAYER_LABEL]:  # [1, 2]
        # 收集所有检测结果和对应的GT匹配信息
        all_detections = []
        total_gt_count = 0
        
        for frame_idx, (det, gt) in enumerate(zip(detections, ground_truths)):
            det_boxes = [b for b, l in zip(det["boxes"], det["labels"]) if l == class_id]
            det_scores = [s for s, l in zip(det["scores"], det["labels"]) if l == class_id]
            gt_boxes = [b for b, l in zip(gt["boxes"], gt["labels"]) if l == class_id]
            
            total_gt_count += len(gt_boxes)
            
            # 为每个检测结果找到最佳匹配的GT
            for box, score in zip(det_boxes, det_scores):
                score_val = score.cpu().item() if torch.is_tensor(score) else score
                
                best_iou = 0.0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_boxes):
                    iou = IoU_box(box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                all_detections.append({
                    'score': score_val,
                    'iou': best_iou,
                    'frame_idx': frame_idx,
                    'gt_idx': best_gt_idx
                })
        
        if total_gt_count == 0:
            aps[class_id] = 0.0
            print(f"Class {class_id}: No ground truth found")
            continue
        
        if len(all_detections) == 0:
            aps[class_id] = 0.0
            print(f"Class {class_id}: No detections found")
            continue
            
        # 按置信度降序排列
        all_detections.sort(key=lambda x: x['score'], reverse=True)
        
        # 跟踪已匹配的GT
        matched_gt = set()
        
        # 计算TP/FP
        tp_fp_labels = []
        scores = []
        
        for det in all_detections:
            scores.append(det['score'])
            
            if det['iou'] >= iou_threshold and det['gt_idx'] != -1:
                gt_key = (det['frame_idx'], det['gt_idx'])
                if gt_key not in matched_gt:
                    # True Positive
                    tp_fp_labels.append(1)
                    matched_gt.add(gt_key)
                else:
                    # False Positive (GT already matched)
                    tp_fp_labels.append(0)
            else:
                # False Positive (IoU too low or no GT)
                tp_fp_labels.append(0)
        
        # 计算AP
        if len(tp_fp_labels) == 0 or sum(tp_fp_labels) == 0:
            aps[class_id] = 0.0
        else:
            aps[class_id] = average_precision_score(tp_fp_labels, scores)
        
        # 打印调试信息
        tp_count = sum(tp_fp_labels)
        fp_count = len(tp_fp_labels) - tp_count
        recall = tp_count / total_gt_count if total_gt_count > 0 else 0
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        
        print(f"Class {class_id}: Detections={len(all_detections)}, GT={total_gt_count}")
        print(f"  TP={tp_count}, FP={fp_count}, Precision={precision:.4f}, Recall={recall:.4f}")
        print(f"  AP@{iou_threshold}={aps[class_id]:.4f}")

    aps['mAP'] = np.mean(list(aps.values()))
    return aps


def getGT(xgtf_path: str) -> t.Tuple[t.List[t.Dict[str, torch.Tensor]], t.List[int]]:
    """
    Parses the .xgtf ground truth file and returns:
    - List of frame-wise dicts with 'boxes' and 'labels'
    - List of frame indices that have annotations
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

    # Return both ground truths and the list of annotated frame indices
    ground_truths = []
    annotated_frames = sorted(gt_by_frame.keys())
    
    for frame_id in annotated_frames:
        boxes = [list(map(float, box)) for box in gt_by_frame[frame_id]['boxes']]
        labels = list(gt_by_frame[frame_id]['labels'])
        ground_truths.append({'boxes': boxes, 'labels': labels})
    
    return ground_truths, annotated_frames


# In your main detection code:
if args.metric_path:
    print("Loading ground truth from:", args.metric_path)
    gt_by_frame, annotated_frames = evaluate.getGT(args.metric_path)
    print(f"Loaded {len(gt_by_frame)} frames with annotations")
    print(f"Annotated frame indices: {annotated_frames[:5]}...{annotated_frames[-5:]}")
    
    # Only evaluate frames that have ground truth annotations
    filtered_detections = []
    for frame_idx in annotated_frames:
        if frame_idx < len(all_detections):
            filtered_detections.append(all_detections[frame_idx])
        else:
            # If detection is missing for this frame, add empty detection
            filtered_detections.append({'boxes': [], 'scores': [], 'labels': []})
    
    # Now both lists have the same length and correspond to the same frames
    print(f"Evaluating {len(filtered_detections)} frames with annotations")
    
    ap_results = evaluate.compute_ap_map(filtered_detections, gt_by_frame)
    
    print("\n===== Evaluation Results =====")
    print(f"Ball AP@0.5:   {ap_results.get(BALL_LABEL, 0.0):.4f}")
    print(f"Player AP@0.5: {ap_results.get(PLAYER_LABEL, 0.0):.4f}")
    print(f"mAP@0.5:       {ap_results.get('mAP', 0.0):.4f}")
