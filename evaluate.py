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
    
    # Debug information
    total_frames = len(detections)
    frames_with_gt = sum(1 for gt in ground_truths if len(gt['boxes']) > 0)
    print(f"\n[Debug] Total frames: {total_frames}, Frames with GT: {frames_with_gt}")
    
    for class_id in [BALL_LABEL, PLAYER_LABEL]:  # [1, 2]
        y_true = []
        y_scores = []
        
        class_name = "Ball" if class_id == BALL_LABEL else "Player"
        frames_evaluated = 0
        
        for det, gt in zip(detections, ground_truths):
            # Skip frames that have no ground truth annotations at all
            if len(gt['boxes']) == 0:
                continue
                
            frames_evaluated += 1
                
            det_boxes = [b for b, l in zip(det["boxes"], det["labels"]) if l == class_id]
            det_scores = [s for s, l in zip(det["scores"], det["labels"]) if l == class_id]
            gt_boxes = [b for b, l in zip(gt["boxes"], gt["labels"]) if l == class_id]
            
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
        
        print(f"[Debug] {class_name} - Evaluated {frames_evaluated} frames, {len(y_true)} predictions")

        if len(y_true) == 0 or len(set(y_true)) == 1:
            aps[class_id] = 0.0
        else:
            aps[class_id] = average_precision_score(y_true, [s.cpu().item() if torch.is_tensor(s) else s for s in y_scores])

    aps['mAP'] = np.mean(list(aps.values()))
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
