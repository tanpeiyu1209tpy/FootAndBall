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
        # 收集所有该类别的检测结果和GT
        all_detections = []
        total_gt_count = 0
        
        for frame_idx, (det, gt) in enumerate(zip(detections, ground_truths)):
            # 获取该类别的检测结果
            det_boxes = [b for b, l in zip(det["boxes"], det["labels"]) if l == class_id]
            det_scores = [s for s, l in zip(det["scores"], det["labels"]) if l == class_id]
            gt_boxes = [b for b, l in zip(gt["boxes"], gt["labels"]) if l == class_id]
            
            total_gt_count += len(gt_boxes)
            
            # 为每个检测结果计算与GT的最大IoU
            for box, score in zip(det_boxes, det_scores):
                max_iou = 0.0
                for gt_box in gt_boxes:
                    iou = IoU_box(box, gt_box)
                    max_iou = max(max_iou, iou)
                
                all_detections.append({
                    'score': score.cpu().item() if torch.is_tensor(score) else score,
                    'max_iou': max_iou,
                    'frame_idx': frame_idx
                })
        
        if total_gt_count == 0:
            aps[class_id] = 0.0
            continue
            
        # 按置信度降序排列
        all_detections.sort(key=lambda x: x['score'], reverse=True)
        
        # 计算精确率和召回率
        tp = 0
        fp = 0
        matched_gt = set()  # 记录已匹配的GT (frame_idx, gt_idx)
        
        precisions = []
        recalls = []
        
        for det in all_detections:
            if det['max_iou'] >= iou_threshold:
                # 检查是否已经匹配过相同的GT
                frame_idx = det['frame_idx']
                gt_boxes = [b for b, l in zip(ground_truths[frame_idx]["boxes"], 
                                            ground_truths[frame_idx]["labels"]) if l == class_id]
                
                # 找到最佳匹配的GT
                best_gt_idx = -1
                best_iou = 0.0
                det_boxes = [b for b, l in zip(detections[frame_idx]["boxes"], 
                                             detections[frame_idx]["labels"]) if l == class_id]
                
                # 重新计算IoU找到最佳匹配（这里可以优化，但为了清晰保持这样）
                for gt_idx, gt_box in enumerate(gt_boxes):
                    for det_idx, det_box in enumerate(det_boxes):
                        det_score = detections[frame_idx]["scores"][
                            [i for i, l in enumerate(detections[frame_idx]["labels"]) if l == class_id][det_idx]
                        ]
                        if torch.is_tensor(det_score):
                            det_score = det_score.cpu().item()
                        
                        if abs(det_score - det['score']) < 1e-6:  # 找到对应的检测框
                            iou = IoU_box(det_box, gt_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = gt_idx
                
                gt_key = (frame_idx, best_gt_idx)
                if gt_key not in matched_gt and best_gt_idx != -1:
                    tp += 1
                    matched_gt.add(gt_key)
                else:
                    fp += 1
            else:
                fp += 1
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / total_gt_count if total_gt_count > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # 计算AP
        if len(precisions) == 0:
            aps[class_id] = 0.0
        else:
            # 使用sklearn的average_precision_score
            # 但需要转换为二元分类格式
            y_true = [1 if det['max_iou'] >= iou_threshold else 0 for det in all_detections]
            y_scores = [det['score'] for det in all_detections]
            
            if len(set(y_true)) == 1:
                aps[class_id] = 0.0
            else:
                aps[class_id] = average_precision_score(y_true, y_scores)

    aps['mAP'] = np.mean(list(aps.values()))
    return aps


def getGT(xgtf_path: str) -> t.Tuple[t.List[t.Dict[str, t.List]], int]:
    """
    Parses the .xgtf ground truth file and returns a list of frame-wise dicts.
    Each dict contains 'boxes': List[List[float]], 'labels': List[int]
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
    print(f"[DEBUG] Ground truth frame range: {frame_ids[0]} ~ {frame_ids[-1]}")

    for frame_id in sorted(gt_by_frame.keys()):
        boxes = [list(map(float, box)) for box in gt_by_frame[frame_id]['boxes']]
        labels = list(gt_by_frame[frame_id]['labels'])
        ground_truths.append({'boxes': boxes, 'labels': labels})
    
    return ground_truths, frame_ids[0]
    for frame_id in sorted(gt_by_frame.keys()):
        boxes = [list(map(float, box)) for box in gt_by_frame[frame_id]['boxes']]
        labels = list(gt_by_frame[frame_id]['labels'])
        ground_truths.append({'boxes': boxes, 'labels': labels})
    return ground_truths, frame_ids[0]
