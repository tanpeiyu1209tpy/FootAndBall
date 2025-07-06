import torch
import cv2
import os
import argparse
import tqdm
import json
import gc
from collections import deque

import network.footandball as footandball
import data.augmentation as augmentations
from data.augmentation import PLAYER_LABEL, BALL_LABEL
import evaluate


def get_memory_usage():

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        return allocated, cached
    return 0, 0


def clear_memory():

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def draw_bboxes(image, detections):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
        if label == PLAYER_LABEL:
            x1, y1, x2, y2 = box
            color = (255, 0, 0)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(image, '{:0.2f}'.format(score), (int(x1), max(0, int(y1)-10)), font, 1, color, 2)

        elif label == BALL_LABEL:
            x1, y1, x2, y2 = box
            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2)
            color = (0, 0, 255)
            radius = 25
            cv2.circle(image, (int(x), int(y)), radius, color, 2)
            cv2.putText(image, '{:0.2f}'.format(score), (max(0, int(x - radius)), max(0, (y - radius - 10))), font, 1,
                        color, 2)

    return image
    
def estimate_memory_usage(frame_width, frame_height, temporal_window, batch_size=1):

    # 单帧内存使用 (3 channels * height * width * 4 bytes for float32)
    single_frame_mb = (3 * frame_height * frame_width * 4) / (1024 * 1024)
    

    temporal_mb = single_frame_mb * temporal_window * batch_size
    

    model_mb = 200  
    
    total_mb = temporal_mb + model_mb
    return total_mb


def adaptive_resize_for_memory(frame_width, frame_height, max_memory_mb=4000):

    current_mb = estimate_memory_usage(frame_width, frame_height, 1)
    
    if current_mb <= max_memory_mb:
        return frame_width, frame_height, 1.0
    
    
    scale = (max_memory_mb / current_mb) ** 0.5
    new_width = int(frame_width * scale)
    new_height = int(frame_height * scale)
    

    new_width = (new_width // 2) * 2
    new_height = (new_height // 2) * 2
    
    return new_width, new_height, scale


def run_detector_memory_efficient(model: footandball.FootAndBall, args: argparse.Namespace):

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    model.print_summary(show_architecture=False)
    model = model.to(args.device)

    try:
        state_dict = torch.load(args.weights, map_location=args.device)
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None

    sequence = cv2.VideoCapture(args.path)
    if not sequence.isOpened():
        print(f"Error: Cannot open video file {args.path}")
        return None
        
    fps = sequence.get(cv2.CAP_PROP_FPS)
    orig_width = int(sequence.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(sequence.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(sequence.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Original video: {orig_width}x{orig_height}, {n_frames} frames")
    
  
    if args.auto_resize:
        proc_width, proc_height, scale = adaptive_resize_for_memory(
            orig_width, orig_height, args.max_memory_mb
        )
        print(f"Processing resolution: {proc_width}x{proc_height} (scale: {scale:.3f})")
        need_resize = scale < 0.99
    else:
        proc_width, proc_height = orig_width, orig_height
        scale = 1.0
        need_resize = False
    
    
    estimated_mb = estimate_memory_usage(proc_width, proc_height, args.temporal_window)
    print(f"Estimated memory usage: {estimated_mb:.1f} MB")
    
    os.makedirs(os.path.dirname(args.out_video), exist_ok=True)
    out_sequence = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (orig_width, orig_height))

    print('Processing video: {}'.format(args.path))
    pbar = tqdm.tqdm(total=n_frames)

    frame_buffer = deque(maxlen=args.temporal_window if args.temporal else 1)
    all_detections = []
    frame_count = 0
    
    
    initial_alloc, initial_cached = get_memory_usage()
    print(f"Initial GPU memory: {initial_alloc:.2f}GB allocated, {initial_cached:.2f}GB cached")

    try:
        while sequence.isOpened():
            ret, frame = sequence.read()
            if not ret:
                break

        
            if need_resize:
              
                processed_frame = cv2.resize(frame, (proc_width, proc_height))
            else:
                processed_frame = frame

            img_tensor = augmentations.numpy2tensor(processed_frame)
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
                            
                      
                            if need_resize:
                                x1 = x1 / scale
                                x2 = x2 / scale
                                y1 = y1 / scale
                                y2 = y2 / scale
                            
                      
                            x1 = max(0, min(x1, orig_width))
                            x2 = max(0, min(x2, orig_width))
                            y1 = max(0, min(y1, orig_height))
                            y2 = max(0, min(y2, orig_height))
                            
                            if x2 > x1 and y2 > y1:
                                filtered_boxes.append([x1, y1, x2, y2])
                                filtered_scores.append(score_val)
                                filtered_labels.append(label_val)

   
                del input_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                detections["boxes"] = filtered_boxes
                detections["scores"] = filtered_scores
                detections["labels"] = filtered_labels

                all_detections.append({
                    "boxes": filtered_boxes,
                    "scores": filtered_scores,
                    "labels": filtered_labels,
                })


            frame = draw_bboxes(frame, detections)
            out_sequence.write(frame)
            
            frame_count += 1
            pbar.update(1)
            
  
            if frame_count % args.memory_check_interval == 0:
                clear_memory()
                alloc, cached = get_memory_usage()
                if frame_count % (args.memory_check_interval * 10) == 0:  # 每1000帧打印一次
                    print(f"\nFrame {frame_count}: {alloc:.2f}GB allocated, {cached:.2f}GB cached")
                
    
                if alloc > args.max_gpu_memory_gb:
                    print(f"Warning: GPU memory usage ({alloc:.2f}GB) exceeds limit ({args.max_gpu_memory_gb}GB)")
                    clear_memory()

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\nOut of memory error at frame {frame_count}")
            print("Suggestions:")
            print("1. Use --auto-resize flag")
            print("2. Reduce --max-memory-mb")
            print("3. Disable temporal fusion (remove --temporal)")
            print("4. Use CPU instead of GPU")
        raise e
    except Exception as e:
        print(f"Error during processing: {e}")
        return None
    finally:
        pbar.close()
        sequence.release()
        out_sequence.release()
        clear_memory()
        
    final_alloc, final_cached = get_memory_usage()
    print(f"Final GPU memory: {final_alloc:.2f}GB allocated, {final_cached:.2f}GB cached")
    
    return all_detections


if __name__ == '__main__':
    import random
    import numpy as np
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except:
            pass
    
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
    
    parser.add_argument('--auto-resize', action='store_true', help='Automatically resize based on memory limit')
    parser.add_argument('--max-memory-mb', type=int, default=4000, help='Maximum memory usage in MB')
    parser.add_argument('--max-gpu-memory-gb', type=float, default=8.0, help='Maximum GPU memory in GB')
    parser.add_argument('--memory-check-interval', type=int, default=100, help='Memory check interval (frames)')
    
    parser.add_argument('--batch-process', action='store_true', help='Process video in batches')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size in frames')

    args = parser.parse_args()

    print('Video path: {}'.format(args.path))
    print('Model: {}'.format(args.model))
    print('Weights: {}'.format(args.weights))
    print('Output video: {}'.format(args.out_video))
    print('Device: {}'.format(args.device))
    print(f'Memory optimization: auto_resize={args.auto_resize}, max_memory={args.max_memory_mb}MB')
    
    if args.temporal:
        print(f'Temporal mode: ON | Window: {args.temporal_window} | Method: {args.fusion_method}')

    if not os.path.exists(args.weights):
        print(f'Error: Weights file not found: {args.weights}')
        exit(1)
    if not os.path.exists(args.path):
        print(f'Error: Input video not found: {args.path}')
        exit(1)

    # 检查设备可用性
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print('Warning: CUDA not available, switching to CPU')
            args.device = 'cpu'
        else:
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_memory_gb = gpu_props.total_memory / 1024**3
            print(f'GPU: {gpu_props.name}, Memory: {gpu_memory_gb:.1f} GB')

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

    all_detections = run_detector_memory_efficient(model, args)
    
    if all_detections is None:
        print("Detection failed")
        exit(1)

    print(f"Successfully processed {len(all_detections)} frames")

    debug_info = {
        "total_frames": len(all_detections),
        "ball_detections": sum(d.get('labels', []).count(BALL_LABEL) for d in all_detections),
        "player_detections": sum(d.get('labels', []).count(PLAYER_LABEL) for d in all_detections),
        "frames_with_ball": sum(1 for d in all_detections if BALL_LABEL in d.get('labels', [])),
        "frames_with_players": sum(1 for d in all_detections if PLAYER_LABEL in d.get('labels', [])),
        "settings": {
            "auto_resize": args.auto_resize,
            "max_memory_mb": args.max_memory_mb,
            "temporal": args.temporal,
            "ball_threshold": args.ball_threshold,
            "player_threshold": args.player_threshold
        }
    }
    
    with open("detection_debug.json", "w") as f:
        json.dump(debug_info, f, indent=2)

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
