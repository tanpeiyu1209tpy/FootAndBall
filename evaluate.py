
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
    
def compute_ap_map_with_resize(detections, ground_truths, original_size, resized_size, iou_threshold=0.5):
    """
    便捷函数：当预测时使用了resize的视频时使用
    
    Args:
        detections: list of dicts with keys ['boxes', 'scores', 'labels']
        ground_truths: list of dicts with keys ['boxes', 'labels']
        original_size: (width, height) of original video
        resized_size: (width, height) of resized video used for prediction
        iou_threshold: IoU threshold for matching
    
    Returns:
        dict {class_id: ap_value, ..., 'mAP': float}
    """
    return compute_ap_map(detections, ground_truths, iou_threshold, 
                         original_size=original_size, resized_size=resized_size)


def compute_ap_map(detections, ground_truths, iou_threshold=0.5, scale_factor=None, original_size=None, resized_size=None):
    """
    detections: list of dicts with keys ['boxes', 'scores', 'labels']
    ground_truths: list of dicts with keys ['boxes', 'labels']
    scale_factor: (scale_x, scale_y) tuple for scaling detection boxes back to original coordinates
    original_size: (width, height) of original video
    resized_size: (width, height) of resized video used for prediction
    returns: dict {class_id: ap_value, ..., 'mAP': float}
    """
    aps = {}
    
    # Calculate scale factors if not provided
    if scale_factor is None and original_size is not None and resized_size is not None:
        scale_x = original_size[0] / resized_size[0]
        scale_y = original_size[1] / resized_size[1]
        scale_factor = (scale_x, scale_y)
    
    # Debug information
    total_frames = len(detections)
    frames_with_gt = sum(1 for gt in ground_truths if len(gt['boxes']) > 0)
    print(f"\n[Debug] Total frames: {total_frames}, Frames with GT: {frames_with_gt}")
    if scale_factor:
        print(f"[Debug] Scale factor: {scale_factor}")
    if original_size and resized_size:
        print(f"[Debug] Original size: {original_size}, Resized size: {resized_size}")
    
    for class_id in [BALL_LABEL, PLAYER_LABEL]:  # [1, 2]
        all_detections = []  # [(score, is_correct), ...]
        total_gt_count = 0
        
        class_name = "Ball" if class_id == BALL_LABEL else "Player"
        frames_evaluated = 0
        
        for det, gt in zip(detections, ground_truths):
            # 处理所有帧，包括没有ground truth的帧
            frames_evaluated += 1
                
            det_boxes = [b for b, l in zip(det["boxes"], det["labels"]) if l == class_id]
            det_scores = [s for s, l in zip(det["scores"], det["labels"]) if l == class_id]
            gt_boxes = [b for b, l in zip(gt["boxes"], gt["labels"]) if l == class_id]
            
            # Scale detection boxes back to original coordinates if needed
            if scale_factor is not None:
                scaled_det_boxes = []
                for box in det_boxes:
                    # box format: [x1, y1, x2, y2]
                    if torch.is_tensor(box):
                        box = box.cpu().numpy()
                    
                    scaled_box = [
                        box[0] * scale_factor[0],  # x1
                        box[1] * scale_factor[1],  # y1
                        box[2] * scale_factor[0],  # x2
                        box[3] * scale_factor[1]   # y2
                    ]
                    scaled_det_boxes.append(scaled_box)
                det_boxes = scaled_det_boxes
            
            total_gt_count += len(gt_boxes)
            
            # 如果没有ground truth，所有检测都是false positive
            if len(gt_boxes) == 0:
                for score in det_scores:
                    score_val = score.cpu().item() if torch.is_tensor(score) else score
                    all_detections.append((score_val, False))
                continue
            
            # 匹配检测框和ground truth
            matched = [False] * len(gt_boxes)

            for box, score in zip(det_boxes, det_scores):
                score_val = score.cpu().item() if torch.is_tensor(score) else score
                
                best_iou = 0.0
                best_match_idx = -1
                
                for i, gt_box in enumerate(gt_boxes):
                    if matched[i]:  # 已经匹配过的GT跳过
                        continue
                    iou = IoU_box(box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = i
                
                # 判断是否匹配成功
                if best_iou >= iou_threshold and best_match_idx != -1:
                    all_detections.append((score_val, True))  # True positive
                    matched[best_match_idx] = True
                else:
                    all_detections.append((score_val, False))  # False positive
        
        print(f"[Debug] {class_name} - Evaluated {frames_evaluated} frames")
        print(f"[Debug] {class_name} - Total GT: {total_gt_count}, Total detections: {len(all_detections)}")

        # 计算AP
        if len(all_detections) == 0 or total_gt_count == 0:
            aps[class_id] = 0.0
            print(f"[Debug] {class_name} - AP: 0.0 (no detections or no GT)")
        else:
            # 按置信度排序
            all_detections.sort(key=lambda x: x[0], reverse=True)
            
            # 计算precision和recall
            tp = 0
            fp = 0
            precisions = []
            recalls = []
            
            for score, is_correct in all_detections:
                if is_correct:
                    tp += 1
                else:
                    fp += 1
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / total_gt_count if total_gt_count > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
            
            # 使用sklearn计算AP
            y_true = [1 if is_correct else 0 for _, is_correct in all_detections]
            y_scores = [score for score, _ in all_detections]
            
            if len(set(y_true)) == 1:
                aps[class_id] = 0.0
            else:
                aps[class_id] = average_precision_score(y_true, y_scores)
            
            print(f"[Debug] {class_name} - AP: {aps[class_id]:.4f}")

    aps['mAP'] = np.mean(list(aps.values()))
    print(f"[Debug] mAP: {aps['mAP']:.4f}")
    return aps


def getGT(xgtf_path: str) -> t.Tuple[t.List[t.Dict[str, torch.Tensor]], int]:
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

    # Get total frames from the file info
    info = root.find('.//{http://lamp.cfar.umd.edu/viper#}file[@name="Information"]')
    num_frames_elem = info.find('.//{http://lamp.cfar.umd.edu/viperdata#}dvalue')
    total_frames = int(num_frames_elem.get('value'))
    
    ground_truths = []
    frame_ids = sorted(gt_by_frame.keys())
    
    print(f"[DEBUG] Total video frames: {total_frames}")
    print(f"[DEBUG] Ground truth frame range: {frame_ids[0] if frame_ids else 'N/A'} ~ {frame_ids[-1] if frame_ids else 'N/A'}")
    print(f"[DEBUG] Number of annotated frames: {len(frame_ids)}")

    # Create ground truth list for ALL frames (including empty ones)
    for frame_id in range(total_frames):
        if frame_id in gt_by_frame:
            boxes = [list(map(float, box)) for box in gt_by_frame[frame_id]['boxes']]
            labels = list(gt_by_frame[frame_id]['labels'])
            ground_truths.append({'boxes': boxes, 'labels': labels})
        else:
            # Empty frame (no annotations)
            ground_truths.append({'boxes': [], 'labels': []})
    
    # Return ground truths and first annotated frame
    return ground_truths, frame_ids[0] if frame_ids else 0
