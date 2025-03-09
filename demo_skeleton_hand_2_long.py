import argparse
import json
import cv2
import numpy as np
import torch
import mmengine
import pandas as pd
from collections import deque
from operator import itemgetter
from mmengine import Config, DictAction
#from mmaction.apis import inference_recognizer, init_recognizer
from pyskl.apis import inference_recognizer, init_recognizer
import mediapipe as mp
import csv
import matplotlib.pyplot as plt
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
#hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.8)
hands = mp_hands.Hands(min_detection_confidence=0.4, min_tracking_confidence=0.5, max_num_hands=1)

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
THICKNESS = 1
LINETYPE = 1

EXCLUED_STEPS = ['OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit', 'PyAVDecode', 'RawFrameDecode']

def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 long video demo with MediaPipe keypoint extraction')
    parser.add_argument('video_path', help='video file/url')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file/url')
    parser.add_argument('label', help=r"D:\pyskl_thesis\tools\data\label_map\hand_gesture.txt")
    parser.add_argument('out_file', help='output result file in video/json')
    parser.add_argument('--input-step', type=int, default=1, help='Step for sampling frames')
    parser.add_argument('csv_file', help='output CSV file for frame predictions')
    parser.add_argument('--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument('--threshold', type=float, default=0.01, help='Recognition score threshold')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, default={}, help='Override settings in config')
    return parser.parse_args()

def extract_hand_keypoints(frame, frame_width, frame_height):
    """Extract hand keypoints using MediaPipe. If no hand is detected, return Class 5 (Negative)."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    keypoints = np.zeros((1, 1, 21, 2))  # Default to Class 5: No hands detected
    keypoint_score = np.zeros((1, 1, 21))  # Confidence = 0 for Class 5

    if results.multi_hand_landmarks:
        print("✅ Hand detected")
        keypoint_score = np.ones((1, 1, 21))  # Confidence = 1 for detected hands
        for hand_landmarks in results.multi_hand_landmarks:
            for i, landmark in enumerate(hand_landmarks.landmark):
                keypoints[0, 0, i] = [landmark.x * frame_width, landmark.y * frame_height]
    else:
        print("⚠️ No hands detected, assigning Class 5 (Negative)")

    return keypoints, keypoint_score

import time
import csv
import cv2
import numpy as np
import torch
from mmengine import Config, DictAction
from pyskl.apis import inference_recognizer, init_recognizer
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
THICKNESS = 1
LINETYPE = 1

def show_results(model, data, label, args):
    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(args.out_file, fourcc, fps, (frame_width, frame_height))

    print(f"📹 Video FPS: {fps}, Width: {frame_width}, Height: {frame_height}, Total Frames: {num_frames}")

    # Prepare CSV files
    with open(args.csv_file, mode='w', newline='') as file, open(args.csv_file.replace(".csv", "_motion.csv"), mode='w', newline='') as motion_file:
        writer = csv.writer(file)
        motion_writer = csv.writer(motion_file)

        # Headers for CSVs
        writer.writerow(["Frame", "Predicted Action"])
        motion_writer.writerow(["No.", "Predicted Label", "Start Time (s)", "End Time (s)", "Duration (s)"])

        frame_idx = 0
        motion_counter = 1
        previous_label = None
        start_time = 0  # Start time for current motion
        total_inference_time =0
        while frame_idx < num_frames:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            frame_idx += 1
            current_time = round(frame_idx / fps, 3)  # Time in seconds
            print(f"🖼 Processing Frame {frame_idx}/{num_frames} (Time: {current_time}s)")

            # Extract hand keypoints using MediaPipe
            keypoints, keypoint_score = extract_hand_keypoints(frame, frame_width, frame_height)

            inference_time = 0  # ✅ Default value to avoid UnboundLocalError
            if np.all(keypoint_score == 0):  # No hand detected
                action_label = "Negative"
            else:
                cur_data = data.copy()
                cur_data.update(
                    dict(
                        keypoint=keypoints,
                        keypoint_score=keypoint_score,
                        modality='Pose',
                        total_frames=1,
                        start_index=0,
                        img_shape=(frame_height, frame_width),
                        original_shape=(frame_height, frame_width)
                    )
                )

                # Run inference for each frame
                start_inference = time.time()
                result = inference_recognizer(model, cur_data)
                end_inference = time.time()

                pred_class = int(result[0][0])  # Get predicted class index
                action_label = label[pred_class]  # Map to action name
                inference_time = round((end_inference - start_inference) * 1000, 3)  # Time in ms
                total_inference_time += inference_time
            print(f"📝 Frame {frame_idx}: Predicted Action = {action_label} (Inference Time: {inference_time} ms)")
            print(f"🕒 Total Inference Time for the clip: {total_inference_time} ms")

            # Write frame prediction to CSV
            writer.writerow([frame_idx, action_label])

            # Check for label transition to record motion duration
            if action_label != previous_label and previous_label is not None:
                end_time = round(frame_idx / fps, 3)  # End time of previous motion
                motion_duration = round(end_time - start_time, 3)  # Duration in seconds
                motion_writer.writerow([motion_counter, previous_label, start_time, end_time, motion_duration])
                motion_counter += 1
                start_time = end_time  # Reset start time for new motion

            previous_label = action_label  # Update previous label

            # Draw results on the frame
            cv2.putText(frame, f"Action: {action_label}", (10, 30), FONTFACE, FONTSCALE, (255, 255, 255), THICKNESS, LINETYPE)
            video_writer.write(frame)

        # Save the last motion duration
        if previous_label is not None:
            end_time = round(frame_idx / fps, 3)
            motion_duration = round(end_time - start_time, 3)
            motion_writer.writerow([motion_counter, previous_label, start_time, end_time, motion_duration])

    # Release resources
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    print(f"✅ CSV file saved at: {args.csv_file}")
    print(f"✅ Motion duration file saved at: {args.csv_file.replace('.csv', '_motion.csv')}")

    ground_truth = []
    motion_predictions = []
    previous_label = None
    start_time = None
    motion_counter = 1
    frame_idx = 0
    gt_motion_data = []
    previous_label = None
    start_time = None
    for i, row in enumerate(ground_truth):
        time_sec = round(i / fps, 3)
        gt_label = row[1]
        
        if gt_label != previous_label:
            if previous_label is not None:
                gt_motion_data.append([previous_label, start_time, time_sec, round(time_sec - start_time, 3)])
            start_time = time_sec
        previous_label = gt_label
    
    gt_df = pd.DataFrame(gt_motion_data, columns=["GT_Label", "Start Time", "End Time", "Duration"])
    gt_df.to_csv("75ground_truth_motion.csv", index=False)
    
    plt.figure(figsize=(10, 5))
        # Convert motion_predictions into a proper DataFrame
    #motion_df = pd.DataFrame(motion_predictions, columns=["No.", "Predicted Label", "Start Time", "End Time", "Duration"])
    if motion_predictions:  # ✅ Ensure motion_predictions is not empty
        motion_df = pd.DataFrame(motion_predictions, columns=["No.", "Predicted Label", "Start Time", "End Time", "Duration"])
        motion_df.to_csv("12predicted_motion.csv", index=False)
    else:
        print("⚠️ No valid motion predictions found, skipping CSV export.")
    motion_df.to_csv("12predicted_motion.csv", index=False)

# Ensure the ground truth motion data is formatted properly
    gt_df = pd.DataFrame(gt_motion_data, columns=["GT_Label", "Start Time", "End Time", "Duration"])
    gt_df.to_csv("12ground_truth_motion.csv", index=False)

    plt.figure(figsize=(10, 5))

# ✅ Plot Ground Truth in RED
    plt.step(gt_df["Start Time"], gt_df["GT_Label"], label="Ground Truth", color="red")

# ✅ Plot Inference Results in BLUE
    plt.step(motion_df["Start Time"], motion_df["Predicted Label"], label="Inference", color="blue")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Motion")
    plt.title("Ground Truth vs. Inference Motion Prediction")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    args = parse_args()
    args.device = torch.device(args.device)
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    model = init_recognizer(args.config, args.checkpoint, device=args.device)
    data = dict(img_shape=None, modality='Pose', label=-1)

    with open(args.label, 'r') as f:
        label = [line.strip() for line in f]

    show_results(model, data, label, args)

if __name__ == '__main__':
    main()
