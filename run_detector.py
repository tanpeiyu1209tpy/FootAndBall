import torch
import cv2
import os
import argparse
import tqdm
import json

import network.footandball as footandball
import data.augmentation as augmentations
from data.augmentation import PLAYER_LABEL, BALL_LABEL
import evaluate


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
            radius = 25
            cv2.circle(image, (int(x), int(y)), radius, color, 2)
            cv2.putText(image, '{:0.2f}'.format(score), (max(0, int(x - radius)), max(0, y - radius - 10)), font, 1, color, 2)
    return image


def run_detector(model: footandball.FootAndBall, args: argparse.Namespace):

    resize_width, resize_height = 640, 360

    model.print_summary(show_architecture=False)
    model = model.to(args.device)

    state_dict = torch.load(args.weights, map_location=args.device)
    model.load_state_dict(state_dict)
    model.eval()

    sequence = cv2.VideoCapture(args.path)
    fps = sequence.get(cv2.CAP_PROP_FPS)
    frame_width = int(sequence.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(sequence.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(sequence.get(cv2.CAP_PROP_FRAME_COUNT))
    out_sequence = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    print('Processing video: {}'.format(args.path))
    pbar = tqdm.tqdm(total=n_frames)

    scale_x = frame_width / resize_width
    scale_y = frame_height / resize_height
    frame_buffer = []
    all_detections = []

    while sequence.isOpened():
        ret, frame = sequence.read()
        if not ret:
            break


        resized_frame = cv2.resize(frame, (resize_width, resize_height))
        img_tensor = augmentations.numpy2tensor(resized_frame)
        frame_buffer.append(img_tensor)

        if args.temporal:
            if len(frame_buffer) < args.temporal_window:
                pbar.update(1)
                continue
            elif len(frame_buffer) > args.temporal_window:
                frame_buffer.pop(0)
            input_tensor = torch.stack(frame_buffer).unsqueeze(0).to(args.device)
        else:
            input_tensor = img_tensor.unsqueeze(0).to(args.device)

        with torch.no_grad():
            detections = model(input_tensor)[0]

        filtered_boxes, filtered_scores, filtered_labels = [], [], []
        for box, score, label in zip(detections["boxes"], detections["scores"], detections["labels"]):
            if (label == BALL_LABEL and score >= args.ball_threshold) or \
               (label == PLAYER_LABEL and score >= args.player_threshold):
                x1, y1, x2, y2 = box.tolist()
                x1 *= scale_x
                y1 *= scale_y
                x2 *= scale_x
                y2 *= scale_y
                box = torch.tensor([x1, y1, x2, y2])
                filtered_boxes.append(box)
                filtered_scores.append(score)
                filtered_labels.append(label)

        detections["boxes"] = filtered_boxes
        detections["scores"] = filtered_scores
        detections["labels"] = filtered_labels

        all_detections.append({
            "boxes": [b.tolist() for b in filtered_boxes],
            "scores": [s.item() for s in filtered_scores],
            "labels": [l.item() for l in filtered_labels],
        })

        frame = draw_bboxes(frame, detections)
        out_sequence.write(frame)
        pbar.update(1)

    pbar.close()
    sequence.release()
    out_sequence.release()
    return all_detections


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

    args = parser.parse_args()

    print('Video path: {}'.format(args.path))
    print('Model: {}'.format(args.model))
    print('Weights: {}'.format(args.weights))
    print('Output video: {}'.format(args.out_video))
    print('Device: {}'.format(args.device))
    if args.temporal:
        print(f'Temporal mode: ON | Window: {args.temporal_window} | Method: {args.fusion_method}')

    assert os.path.exists(args.weights), 'Weights not found'
    assert os.path.exists(args.path), 'Input video not found'

    model = footandball.model_factory(
        args.model, 'detect',
        ball_threshold=args.ball_threshold,
        player_threshold=args.player_threshold,
        use_temporal_fusion=args.temporal,
        temporal_window=args.temporal_window,
        fusion_method=args.fusion_method
    )

    all_detections = run_detector(model, args)

    # === Evaluation ===
    if args.metric_path:
        print("\nLoading ground truth from:", args.metric_path)
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
    else:
        print("No metric path provided. Skipping mAP evaluation.")

