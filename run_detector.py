# FootAndBall: Integrated Player and Ball Detector
# Jacek Komorowski, Grzegorz Kurzejamski, Grzegorz Sarwas
# Copyright (c) 2020 Sport Algorithmics and Gaming

#
# Run FootAndBall detector on ISSIA-CNR Soccer videos
#

import torch
import cv2
import os
import argparse
import tqdm
import json

import network.footandball as footandball
import data.augmentation as augmentations
from data.augmentation import PLAYER_LABEL, BALL_LABEL
import evaluate as metric  # Assumes metric.py is in same directory


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
            cv2.putText(image, '{:0.2f}'.format(score), (max(0, int(x - radius)), max(0, (y - radius - 10))), font, 1, color, 2)
    return image


def run_detector(model: footandball.FootAndBall, args: argparse.Namespace):
    model.print_summary(show_architecture=False)
    model = model.to(args.device)

    if args.device == 'cpu':
        print('Loading CPU weights...')
        state_dict = torch.load(args.weights, map_location=lambda storage, loc: storage)
    else:
        print('Loading GPU weights...')
        state_dict = torch.load(args.weights)

    model.load_state_dict(state_dict)
    model.eval()

    sequence = cv2.VideoCapture(args.path)
    fps = sequence.get(cv2.CAP_PROP_FPS)
    (frame_width, frame_height) = (int(sequence.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                   int(sequence.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    n_frames = int(sequence.get(cv2.CAP_PROP_FRAME_COUNT))
    out_sequence = cv2.VideoWriter(args.out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                   (frame_width, frame_height))

    print('Processing video: {}'.format(args.path))
    pbar = tqdm.tqdm(total=n_frames)

    all_detections = []
    frame_idx = 0

    while sequence.isOpened():
        ret, frame = sequence.read()
        if not ret:
            break

        img_tensor = augmentations.numpy2tensor(frame)
        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(dim=0).to(args.device)
            detections = model(img_tensor)[0]

            frame_detections = {
                "boxes": [box.tolist() for box in detections["boxes"]],
                "scores": [score.item() for score in detections["scores"]],
                "labels": [label.item() for label in detections["labels"]]
            }
            all_detections.append(frame_detections)

        frame = draw_bboxes(frame, detections)
        out_sequence.write(frame)
        pbar.update(1)
        frame_idx += 1

    pbar.close()
    sequence.release()
    out_sequence.release()

    return all_detections


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to video', type=str, required=True)
    parser.add_argument('--model', help='model name', type=str, default='fb1')
    parser.add_argument('--weights', help='path to model weights', type=str, required=True)
    parser.add_argument('--ball_threshold', help='ball confidence detection threshold', type=float, default=0.5)
    parser.add_argument('--player_threshold', help='player confidence detection threshold', type=float, default=0.5)
    parser.add_argument('--out_video', help='path to output video', type=str, required=True)
    parser.add_argument('--device', help='device (CPU or CUDA)', type=str, default='cuda:0')
    parser.add_argument('--metric-path', help='Path to ground truth annotation (.xgtf) for evaluation', type=str)
    args = parser.parse_args()

    print('Video path: {}'.format(args.path))
    print('Model: {}'.format(args.model))
    print('Weights: {}'.format(args.weights))
    print('Output: {}'.format(args.out_video))
    print('Device: {}'.format(args.device))

    assert os.path.exists(args.weights), 'Weights not found'
    assert os.path.exists(args.path), 'Input video not found'

    model = footandball.model_factory(args.model, 'detect',
                                      ball_threshold=args.ball_threshold,
                                      player_threshold=args.player_threshold)

    all_detections = run_detector(model, args)

    # Load ground truth
    if args.metric_path:
        print("Loading ground truth from:", args.metric_path)
        gt_by_frame, gt_start_frame = metric.getGT(args.metric_path)
        print(f"Loaded {len(gt_by_frame)} frames of ground truth. Start from frame {gt_start_frame}")

        all_detections = all_detections[gt_start_frame:]
        
        # Ensure detection and GT have same number of frames
        if len(all_detections) > len(gt_by_frame):
            all_detections = all_detections[:len(gt_by_frame)]
        elif len(gt_by_frame) > len(all_detections):
            gt_by_frame = gt_by_frame[:len(all_detections)]

        ap_results = metric.compute_ap_map(all_detections, gt_by_frame)

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
