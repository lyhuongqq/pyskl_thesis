import argparse
import json
import cv2
import numpy as np
import torch
import mmengine
import pandas as pd
import matplotlib.pyplot as plt
import csv
import time
from mmengine import Config, DictAction
from pyskl.apis import inference_recognizer, init_recognizer
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.4, min_tracking_confidence=0.5, max_num_hands=1)

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
THICKNESS = 1
LINETYPE = 1
#csv_file=r"D:\Hand-GCN-main\Hand-Gesture-GCN-main\mtm_augmented_data\Jabil\OneDrive_2_27-11-2024\70_mdc04_mtc05_Train_original.csv"
csv_file = r"D:\Hand-Gesture-Recognition-in-manual-assembly-tasks-using-GCN-main\Hand-Gesture-Recognition-in-manual-assembly-tasks-using-GCN-main\data\graphdata\0_mdc04_mtc05_Train_original.csv"
#csv_file=r"D:\Hand-GCN-main\Hand-Gesture-GCN-main\mtm_augmented_data\Jabil\OneDrive_2_27-11-2024\75_mdc04_mtc05_Train_original.csv"
#csv_file=r"D:\Hand-GCN-main\Hand-Gesture-GCN-main\mtm_augmented_data\Jabil\OneDrive_2_27-11-2024\66_mdc04_mtc05_Train_original.csv"
def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 long video demo with MediaPipe keypoint extraction')
    parser.add_argument('video_path', help='video file/url')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file/url')
    parser.add_argument('label', help='Path to label map file')
    parser.add_argument('out_file', help='Output result file in video/json')
    #parser.add_argument('csv_file', default=csv_file, help='Ground truth CSV file')
    parser.add_argument('--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument('--threshold', type=float, default=0.01, help='Recognition score threshold')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, default={}, help='Override settings in config')
    return parser.parse_args()

def extract_hand_keypoints(frame, frame_width, frame_height):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    keypoints = np.zeros((1, 1, 21, 2))
    keypoint_score = np.zeros((1, 1, 21))
    
    if results.multi_hand_landmarks:
        keypoint_score = np.ones((1, 1, 21))
        for hand_landmarks in results.multi_hand_landmarks:
            for i, landmark in enumerate(hand_landmarks.landmark):
                keypoints[0, 0, i] = [landmark.x * frame_width, landmark.y * frame_height]
    
    return keypoints, keypoint_score 

def progressBar(pil_im, bgcolor, color, x, y, w, h, progress):
    drawObject = ImageDraw.Draw(pil_im)
    drawObject.ellipse((x + w, y, x + h + w, y + h), fill=bgcolor)
    drawObject.ellipse((x, y, x + h, y + h), fill=bgcolor)
    drawObject.rectangle((x + (h / 2), y, x + w + (h / 2), y + h), fill=bgcolor)
    if progress <= 0:
        progress = 0.01
    if progress > 1:
        progress = 1
    w = w * progress
    drawObject.ellipse((x + w, y, x + h + w, y + h), fill=color)
    drawObject.ellipse((x, y, x + h, y + h), fill=color)
    drawObject.rectangle((x + (h / 2), y, x + w + (h / 2), y + h), fill=color)
    return pil_im

def show_results(model, data, label, args):
    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps_target = 24  # Adjust FPS to match CSV
    frame_skip_rate = fps / fps_target
    frame_counter = 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(args.out_file, fourcc, fps_target, (frame_width, frame_height))
    gt_data = pd.read_csv(csv_file) #args.csv_file
    gt_data.columns = gt_data.columns.str.lower()
    if 'label' not in gt_data.columns:
        raise ValueError("Error: 'label' column not found in ground truth CSV.")
    
    ground_truth = []
    motion_predictions = []
    previous_label = None
    start_time = None
    motion_counter = 1
    frame_idx = 0

    with open(csv_file.replace(".csv", "_motion.csv"), mode='w', newline='') as motion_file:
        motion_writer = csv.writer(motion_file)
        motion_writer.writerow(["No.", "Predicted Label", "Start Time (s)", "End Time (s)", "Duration (s)"])

        while frame_idx < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            if frame_counter % frame_skip_rate >= 1:
                continue  # Skip frames to match 24 FPS
            current_time = round(frame_idx / fps, 3)
            print(f"🖼 Processing Frame {frame_idx}/{num_frames} (Time: {current_time}s)")
            keypoints, keypoint_score = extract_hand_keypoints(frame, frame_width, frame_height)
            
            if np.all(keypoint_score == 0):
                action_label = "Negative"
                confidence_score = 0.0
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
                result = inference_recognizer(model, cur_data)
                pred_class = int(result[0][0])
                confidence_score = result[0][1]
                action_label = label[pred_class]
            
            gt_label = gt_data.iloc[min(frame_idx, len(gt_data) - 1)]['label']
            ground_truth.append([frame_idx, gt_label])
            
            #if start_time is not None:  # ✅ Ensure start_time is initialized before using it
            #    motion_predictions.append([motion_counter, action_label, start_time, current_time, round(current_time - start_time, 3)])
            #    motion_counter += 1
            #    start_time = current_time
            print(f"📝 Frame {frame_idx}: Predicted Action = {action_label} ")
            
            if previous_label is None:
                    start_time = current_time  # Ensure start_time is initialized properly
            motion_predictions.append([motion_counter, action_label, start_time, current_time, round(current_time - start_time, 3)])
            if action_label != previous_label:
                if previous_label is not None:
                    motion_writer.writerow([motion_counter, previous_label, start_time, current_time, round(current_time - start_time, 3)])
                    motion_predictions.append([motion_counter, previous_label, start_time, current_time, round(current_time - start_time, 3)])
                    motion_counter += 1
            #if action_label != previous_label and previous_label is not None:
                #if start_time is not None:  # ✅ Ensure start_time is valid before subtraction
                #    motion_writer.writerow([motion_counter, previous_label, start_time, current_time, round(current_time - start_time, 3)])
                #motion_writer.writerow([motion_counter, previous_label, start_time, current_time, round(current_time - start_time, 3)])
                #motion_counter += 1
                start_time = current_time
            previous_label = action_label
            
            cv2.putText(frame, f"Action: {action_label}", (10, 30), FONTFACE, FONTSCALE, (255, 255, 255), THICKNESS, LINETYPE)
            #video_writer.write(frame)

            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            font = ImageFont.truetype("arial.ttf", 24)
            draw.text((10, 10), f"Action: {action_label} ({confidence_score:.2f})", (255, 255, 255), font=font)
            pil_image = progressBar(pil_image, (50, 50, 50), (0, 255, 0), 200, 50, 300, 20, confidence_score)
            numpy_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            video_writer.write(numpy_image)
            cv2.imshow("Action Recognition", numpy_image)
        if previous_label is not None:
            motion_writer.writerow([motion_counter, previous_label, start_time, current_time, round(current_time - start_time, 3)])
            motion_predictions.append([motion_counter, previous_label, start_time, current_time, round(current_time - start_time, 3)])    

    cap.release()
    video_writer.release()
    
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
    
    if previous_label is not None:
        gt_motion_data.append([previous_label, start_time, time_sec, round(time_sec - start_time, 3)])

    gt_df = pd.DataFrame(gt_motion_data, columns=["GT_Label", "Start Time", "End Time", "Duration"])
    gt_df.to_csv("0_ground_truth_motion.csv", index=False)
    
    plt.figure(figsize=(10, 5))
        # Convert motion_predictions into a proper DataFrame
    #motion_df = pd.DataFrame(motion_predictions, columns=["No.", "Predicted Label", "Start Time", "End Time", "Duration"])
    if motion_predictions:  # ✅ Ensure motion_predictions is not empty
        motion_df = pd.DataFrame(motion_predictions, columns=["No.", "Predicted Label", "Start Time", "End Time", "Duration"])
        motion_df.to_csv("c3d_0_hybrid_predicted_motion3.csv", index=False)
    else:
        print("⚠️ No valid motion predictions found, skipping CSV export.")
    #motion_df.to_csv("70_c3d_Ja_frame_predicted_motion.csv", index=False)

# Ensure the ground truth motion data is formatted properly
    #gt_df = pd.DataFrame(gt_motion_data, columns=["GT_Label", "Start Time", "End Time", "Duration"])
    #gt_df.to_csv("70_Ja_ground_truth_motion.csv", index=False)

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
