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
    
    #hands = mp_hands.Hands(min_detection_confidence=0.4, min_tracking_confidence=0.5, max_num_hands=1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Augmentation and color conversion
        if augmentation == "original":
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif augmentation == "flip-vert":
            frame_rgb = cv2.cvtColor(cv2.flip(frame, 0), cv2.COLOR_BGR2RGB)
        elif augmentation == "flip-hor":
            frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        elif augmentation == "flip-hor-vert":
            frame_rgb = cv2.cvtColor(cv2.flip(frame, -1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        frame_rgb.flags.writeable = False
        results = hands.process(frame_rgb)
        frame_rgb.flags.writeable = True
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Assuming single hand
            keypoint = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32)  # Shape [21, 3]
            keypoint_score = np.array([lm.visibility for lm in hand_landmarks.landmark], dtype=np.float32)  # Shape [21]
            # Optionally draw hand landmarks
            #mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            # If no hand landmarks are detected, fill with zeros
            keypoint = np.zeros((21, 2), dtype=np.float32)
            keypoint_score = np.zeros(21, dtype=np.float32)
        
        keypoints.append(keypoint)
        keypoints_scores.append(keypoint_score)

        # Display the result (for debugging or visualization)
        #cv2.imshow('MediaPipe Hands', frame_bgr)
        #if cv2.waitKey(1) & 0xFF == 27:
        #    break
    
    cap.release()
    #cv2.destroyAllWindows()
    
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
            
            print(f"Processing {video_path} with {aug_suffix}, keypoints shape: {keypoints}, scores shape: {keypoint_scores}")
            
            annotations.append({
                'frame_dir': Path(video_path).stem + f"_{aug_suffix}",
                'total_frames': total_frames,
                'img_shape': img_shape,
                'original_shape': img_shape,
                'label': label,
                'keypoint': keypoints[np.newaxis, ...],  # Add extra dimension for number of persons (M=1)
                'keypoint_score': keypoint_scores[np.newaxis, ...]  # Shape [1, T, V]
            })
    
    # Load split files for train and val
    train_videos = load_split_file(train_txt)
    val_videos = load_split_file(eval_txt)  # Changed 'eval' to 'val' to match expected key
    
    split_dict = {
        'train': [Path(video).stem for video in train_videos],
        'val': [Path(video).stem for video in val_videos]  # Changed 'eval' to 'val'
    }
    
    # Save everything to pickle
    data = {
        'split': split_dict,
        'annotations': annotations
    }
    
    with open(output_pickle, 'wb') as f:
        pickle.dump(data, f) #, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Data with augmentations saved to {output_pickle}")


# Example usage
video_label_txt = r"D:\pyskl-main\pyskl-main\frame\video_label_2.txt"  # Input text file containing video paths and labels
train_txt = r"D:\pyskl-main\pyskl-main\frame\train.txt"
#train_txt = " "   # File containing train video paths
eval_txt = r"D:\pyskl-main\pyskl-main\frame\eval.txt"  # File containing eval video paths
output_pickle = 'hand_pose_dataset_aug_val_5.pkl'
  # Output pickle file

process_videos_and_save_pickle(video_label_txt, train_txt=None, eval_txt=eval_txt, output_pickle=output_pickle)
