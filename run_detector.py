import torch
import cv2
import os
import argparse
import tqdm
import json
import numpy as np
import random
from collections import deque

import network.footandball as footandball
import data.augmentation as augmentations
from data.augmentation import PLAYER_LABEL, BALL_LABEL
import evaluate


def set_deterministic(seed=42):
    """
    设置确定性随机种子，确保结果可复现
    """
    print(f"Setting deterministic mode with seed: {seed}")
    
    # Python随机数
    random.seed(seed)
    
    # NumPy随机数
    np.random.seed(seed)
    
    # PyTorch随机数
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保CUDA操作是确定性的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置环境变量以确保更好的确定性
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # PyTorch 1.8+的确定性设置
    try:
        torch.use_deterministic_algorithms(True)
    except:
        pass


def log_system_info():
    """
    记录系统信息，帮助调试差异
    """
    print("=== 系统信息 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  总内存: {props.total_memory / 1024**3:.1f} GB")
            print(f"  已分配: {torch.cuda.memory_allocated(i) / 1024**3:.1f} GB")
            print(f"  缓存: {torch.cuda.memory_reserved(i) / 1024**3:.1f} GB")


def clear_gpu_cache():
    """
    彻底清理GPU缓存
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def create_detection_hash(detections):
    """
    为检测结果创建哈希值，用于比较
    """
    # 将检测结果转换为可哈希的字符串
    detection_str = json.dumps(detections, sort_keys=True)
    return hash(detection_str)


def save_debug_info(detections, filename_prefix):
    """
    保存调试信息
    """
    debug_info = {
        "total_frames": len(detections),
        "total_ball_detections": sum(len([l for l in frame.get('labels', []) if l == BALL_LABEL]) for frame in detections),
        "total_player_detections": sum(len([l for l in frame.get('labels', []) if l == PLAYER_LABEL]) for frame in detections),
        "detection_hash": create_detection_hash(detections),
        "first_10_frames": detections[:10] if len(detections) >= 10 else detections,
        "system_info": {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        }
    }
    
    with open(f"{filename_prefix}_debug.json", "w") as f:
        json.dump(debug_info, f, indent=2)
    
    print(f"调试信息已保存到: {filename_prefix}_debug.json")
    print(f"检测结果哈希值: {debug_info['detection_hash']}")


def run_detector_deterministic(model: footandball.FootAndBall, args: argparse.Namespace):
    """
    确定性的检测器运行函数
    """
    # 设置确定性模式
    if args.deterministic:
        set_deterministic(args.seed)
    
    # 记录系统信息
    log_system_info()
    
    # 清理GPU缓存
    clear_gpu_cache()
    
    model.print_summary(show_architecture=False)
    model = model.to(args.device)

    try:
        state_dict = torch.load(args.weights, map_location=args.device)
        model.load_state_dict(state_dict)
        model.eval()
        
        # 确保模型处于评估模式且没有梯度计算
        for param in model.parameters():
            param.requires_grad = False
            
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None

    sequence = cv2.VideoCapture(args.path)
    if not sequence.isOpened():
        print(f"Error: Cannot open video file {args.path}")
        return None
        
    fps = sequence.get(cv2.CAP_PROP_FPS)
    frame_width = int(sequence.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(sequence.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(sequence.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {frame_width}x{frame_height}, {n_frames}帧, {fps:.2f} FPS")
    
    os.makedirs(os.path.dirname(args.out_video), exist_ok=True)
    out_sequence = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    print('Processing video: {}'.format(args.path))
    pbar = tqdm.tqdm(total=n_frames)

    frame_buffer = deque(maxlen=args.temporal_window if args.temporal else 1)
    all_detections = []
    frame_count = 0

    try:
        while sequence.isOpened():
            ret, frame = sequence.read()
            if not ret:
                break

            img_tensor = augmentations.numpy2tensor(frame)
            frame_buffer.append(img_tensor)

            if args.temporal:
                if len(frame_buffer) < args.temporal_window:
                    padded_buffer = list(frame_buffer)
                    while len(padded_buffer) < args.temporal_window:
                        padded_buffer.insert(0, frame_buffer[0])
                    input_tensor = torch.stack(padded_buffer).unsqueeze(0).to(args.device)
                else:
                    input_tensor = torch.stack(list(frame_buffer)).unsqueeze(0).to(args.device)
            else:
                input_tensor = img_tensor.unsqueeze(0).to(args.device)

            with torch.no_grad():
                # 确保推理是确定性的
                if args.deterministic and torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                detections = model(input_tensor)[0]

                filtered_boxes, filtered_scores, filtered_labels = [], [], []
                
                if len(detections["boxes"]) > 0:
                    for box, score, label in zip(detections["boxes"], detections["scores"], detections["labels"]):
                        score_val = score.item() if torch.is_tensor(score) else score
                        label_val = label.item() if torch.is_tensor(label) else label
                        
                        if (label_val == BALL_LABEL and score_val >= args.ball_threshold) or \
                           (label_val == PLAYER_LABEL and score_val >= args.player_threshold):
                            
                            if torch.is_tensor(box):
                                box_coords = box.tolist()
                            else:
                                box_coords = box
                            
                            x1, y1, x2, y2 = box_coords
                            x1 = max(0, min(x1, frame_width))
                            x2 = max(0, min(x2, frame_width))
                            y1 = max(0, min(y1, frame_height))
                            y2 = max(0, min(y2, frame_height))
                            
                            if x2 > x1 and y2 > y1:
                                # 确保数值精度一致
                                filtered_boxes.append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)])
                                filtered_scores.append(round(score_val, 4))
                                filtered_labels.append(int(label_val))

                all_detections.append({
                    "frame": frame_count,
                    "boxes": filtered_boxes,
                    "scores": filtered_scores,
                    "labels": filtered_labels,
                })

            # 绘制边界框
            detections_for_draw = {
                "boxes": filtered_boxes,
                "scores": filtered_scores,
                "labels": filtered_labels
            }
            frame = draw_bboxes(frame, detections_for_draw)
            out_sequence.write(frame)
            
            frame_count += 1
            pbar.update(1)
            
            # 定期清理内存（但在确定性模式下减少频率）
            if frame_count % 200 == 0:
                if args.deterministic:
                    clear_gpu_cache()

    except Exception as e:
        print(f"Error during processing: {e}")
        return None
    finally:
        pbar.close()
        sequence.release()
        out_sequence.release()
        clear_gpu_cache()
    
    # 保存调试信息
    if args.save_debug:
        save_debug_info(all_detections, args.debug_prefix)
    
    return all_detections


def draw_bboxes(image, detections):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
        if label == PLAYER_LABEL:
            x1, y1, x2, y2 = box
            color = (255, 0, 0)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(image, '{:0.2f}'.format(score), (int(x1), max(0, int(y1) - 10)), font, 1, color, 2)

        elif label == BALL_LABEL:
            x1, y1, x2, y2 = box
            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2)
            color = (0, 0, 255)
            
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            radius = max(5, min(25, int(min(bbox_width, bbox_height) / 2)))
            
            cv2.circle(image, (int(x), int(y)), radius, color, 2)
            cv2.putText(image, '{:0.2f}'.format(score), (max(0, int(x - radius)), max(0, y - radius - 10)), font, 1, color, 2)
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Path to input video')
    parser.add_argument('--model', type=str, default='fb1', help='Model name')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--ball_threshold', type=float, default=0.5, help='Ball confidence threshold')
    parser.add_argument('--player_threshold', type=float, default=0.5, help='Player confidence threshold')
    parser.add_argument('--out_video', type=str, required=True, help='Output video path')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run inference on')
    parser.add_argument('--metric-path', type=str, help='Path to .xgtf ground truth for evaluation')

    # Temporal arguments
    parser.add_argument('--temporal', action='store_true', help='Enable temporal fusion')
    parser.add_argument('--temporal-window', type=int, default=3, help='Number of frames in temporal window')
    parser.add_argument('--fusion-method', type=str, default='difference',
                        choices=['difference', 'variance', 'weighted_avg', 'attention'], help='Fusion method type')
    
    # 确定性和调试参数
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic mode for reproducible results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for deterministic mode')
    parser.add_argument('--save-debug', action='store_true', help='Save detailed debug information')
    parser.add_argument('--debug-prefix', type=str, default='run', help='Prefix for debug files')

    args = parser.parse_args()

    print('Video path: {}'.format(args.path))
    print('Model: {}'.format(args.model))
    print('Weights: {}'.format(args.weights))
    print('Output video: {}'.format(args.out_video))
    print('Device: {}'.format(args.device))
    print(f'Deterministic mode: {args.deterministic}')
    
    if args.temporal:
        print(f'Temporal mode: ON | Window: {args.temporal_window} | Method: {args.fusion_method}')

    if not os.path.exists(args.weights):
        print(f'Error: Weights file not found: {args.weights}')
        exit(1)
    if not os.path.exists(args.path):
        print(f'Error: Input video not found: {args.path}')
        exit(1)

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('Warning: CUDA not available, switching to CPU')
        args.device = 'cpu'

    try:
        model = footandball.model_factory(
            args.model, 'detect',
            ball_threshold=args.ball_threshold,
            player_threshold=args.player_threshold,
            use_temporal_fusion=args.temporal,
            temporal_window=args.temporal_window,
            fusion_method=args.fusion_method
        )
    except Exception as e:
        print(f'Error creating model: {e}')
        exit(1)

    all_detections = run_detector_deterministic(model, args)
    
    if all_detections is None:
        print("Detection failed")
        exit(1)

    print(f"Successfully processed {len(all_detections)} frames")

    # === Evaluation ===
    if args.metric_path:
        print("\nLoading ground truth from:", args.metric_path)
        try:
            gt_by_frame, gt_start_frame = evaluate.getGT(args.metric_path)
            print(f"Loaded {len(gt_by_frame)} frames of ground truth")

            if len(all_detections) < len(gt_by_frame):
                print(f"Warning: Video has fewer frames ({len(all_detections)}) than GT ({len(gt_by_frame)})")
                while len(all_detections) < len(gt_by_frame):
                    all_detections.append({'boxes': [], 'scores': [], 'labels': []})
            elif len(all_detections) > len(gt_by_frame):
                print(f"Warning: Video has more frames ({len(all_detections)}) than GT ({len(gt_by_frame)})")
                all_detections = all_detections[:len(gt_by_frame)]

            ap_results = evaluate.compute_ap_map(all_detections, gt_by_frame)

            print("\n===== Evaluation Results =====")
            print(f"Ball AP@0.5:   {ap_results.get(BALL_LABEL, 0.0):.4f}")
            print(f"Player AP@0.5: {ap_results.get(PLAYER_LABEL, 0.0):.4f}")
            print(f"mAP@0.5:       {ap_results.get('mAP', 0.0):.4f}")

            with open("ap_results.json", "w", encoding="utf-8") as f:
                json.dump({
                    "ball_ap": ap_results.get(BALL_LABEL, 0.0),
                    "player_ap": ap_results.get(PLAYER_LABEL, 0.0),
                    "mAP@0.5": ap_results.get('mAP', 0.0)
                }, f, indent=2)
                
            print("Results saved to ap_results.json")
        except Exception as e:
            print(f"Error during evaluation: {e}")
    else:
        print("No metric path provided. Skipping mAP evaluation.")
    
    print("Processing completed successfully!")
