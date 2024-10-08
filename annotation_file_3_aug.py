import cv2
import mediapipe as mp
import numpy as np
import pickle
from pathlib import Path
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# Initialize MediaPipe Hand Pose model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.4, min_tracking_confidence=0.5)

# Function to extract keypoints and scores with augmentation
def extract_hand_keypoints_and_scores(video_path, augmentation="original"):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    keypoints = []
    keypoints_scores = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply augmentation
        #if augmentation == "flip-vert":
        #    frame = cv2.flip(frame, 0)
        #elif augmentation == "flip-hor":
        #    frame = cv2.flip(frame, 1)
        #elif augmentation == "flip-hor-vert":
        #    frame = cv2.flip(frame, -1)
        
        # Augmentation and color conversion
        if augmentation == "original":
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif augmentation == "flip-vert":
            frame_rgb = cv2.cvtColor(cv2.flip(frame, 0), cv2.COLOR_BGR2RGB)
        elif augmentation == "flip-hor":
            frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        elif augmentation == "flip-hor-vert":
            frame_rgb = cv2.cvtColor(cv2.flip(frame, -1), cv2.COLOR_BGR2RGB)

        #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #result = hands.process(frame_rgb)
        frame_rgb.flags.writeable = False
        result = hands.process(frame_rgb)
        frame_rgb.flags.writeable = True
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]  # Single hand
            keypoint = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32)  # Shape [21, 2]
            keypoint_score = np.array([lm.visibility for lm in hand_landmarks.landmark], dtype=np.float32)  # Shape [21]
        else:
            keypoint = np.zeros((21, 2), dtype=np.float32)
            keypoint_score = np.zeros(21, dtype=np.float32)

        keypoints.append(keypoint)
        keypoints_scores.append(keypoint_score)
    
    cap.release()
    
    keypoints = np.array(keypoints, dtype=np.float32)  # Shape [T, 21, 2]
    keypoints_scores = np.array(keypoints_scores, dtype=np.float32)  # Shape [T, 21]
    
    return keypoints, keypoints_scores, frame_count, (img_height, img_width)

# Function to load split text files
def load_split_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

# Load video-label pairs from input text file
def load_video_label_pairs(txt_file):
    video_paths = []
    labels = []
    with open(txt_file, 'r') as f:
        for line in f:
            video, label = line.strip().split(' ')
            video_paths.append(video)
            labels.append(int(label))
    return video_paths, labels

# Main function with augmentation
def process_videos_and_save_pickle(video_label_txt, train_txt, eval_txt, output_pickle):
    video_paths, labels = load_video_label_pairs(video_label_txt)
    
    annotations = []
    augmentations = ["original", "flip-vert", "flip-hor", "flip-hor-vert"]
    
    # Process each video and its augmentations
    for idx, video_path in enumerate(video_paths):
        label = labels[idx]
        
        for aug in augmentations:
            aug_suffix = {
                "original": "ori",
                "flip-vert": "flip-vert",
                "flip-hor": "flip-hor",
                "flip-hor-vert": "flip-hor-vert"
            }[aug]
            
            keypoints, keypoint_scores, total_frames, img_shape = extract_hand_keypoints_and_scores(video_path, aug)
            print(f"Processing {video_path} with {aug_suffix}, keypoints shape: {keypoints.shape}, scores shape: {keypoint_scores.shape}")
            
            annotations.append({
                'frame_dir': Path(video_path).stem + f"_{aug_suffix}",
                'total_frames': total_frames,
                'img_shape': img_shape,
                'original_shape': img_shape,
                'label': label,
                'keypoint': keypoints[np.newaxis, ...],  # Add extra dimension for number of persons (M=1)
                'keypoint_score': keypoint_scores[np.newaxis, ...]  # Shape [1, T, V]
            })
    
    # Load split files for train and eval
    train_videos = load_split_file(train_txt)
    val_videos = load_split_file(eval_txt)
    
    split_dict = {
        'train': [Path(video).stem for video in train_videos],
        'val': [Path(video).stem for video in val_videos]
    }
    
    # Save everything to pickle
    data = {
        'split': split_dict,
        'annotations': annotations
    }
    
    with open(output_pickle, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Data with augmentations saved to {output_pickle}")

# Example usage
video_label_txt = r"D:\pyskl-main\pyskl-main\frame\video_label_1.txt"  # Input text file containing video paths and labels
train_txt = r"D:\pyskl-main\pyskl-main\frame\train.txt"  # File containing train video paths
eval_txt = r"D:\pyskl-main\pyskl-main\frame\eval.txt"  # File containing eval video paths
output_pickle = 'hand_pose_dataset_with_aug_5.pkl'  # Output pickle file

process_videos_and_save_pickle(video_label_txt, train_txt, eval_txt, output_pickle)
