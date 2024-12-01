import cv2
import mediapipe as mp
import numpy as np
import torch
from pyskl.apis import init_recognizer
from pyskl.datasets import GestureDataset
from pyskl.datasets.pipelines import Compose

# Initialize MediaPipe
mp_hands = mp.solutions.hands


def landmark2nparray(landmark):
    """Convert MediaPipe landmarks to NumPy array (x, y)."""
    return np.array([[lm.x, lm.y] for lm in landmark.landmark])


def create_fake_anno(keypoints, clip_len=10):
    """Generate fake annotation for testing."""
    keypoints = np.array(keypoints, dtype=np.float32)
    if len(keypoints) < clip_len:
        pad_len = clip_len - len(keypoints)
        pad = np.zeros((pad_len, *keypoints.shape[1:]), dtype=np.float32)
        keypoints = np.concatenate([pad, keypoints], axis=0)
    keypoints = keypoints[None]  # Add batch dimension

    return dict(
        keypoint=keypoints,
        total_frames=clip_len,
        frame_dir='NA',
        label=0,
        modality='Pose',
        test_mode=True
    )


# Initialize recognizer
config_path = r"D:\pyskl_thesis\demo_config_pose3dconv.py"
checkpoint_path = r"D:\pyskl-main\work_dirs\test_100\posec3d\test_aug8\epoch_11.pth"

recognizer = init_recognizer(config_path, checkpoint_path, device='cpu')
recognizer.eval()

# Get test pipeline
cfg = recognizer.cfg
test_pipeline = Compose(cfg.test_pipeline)

# Video input
cap = cv2.VideoCapture(r"D:\Hand-GCN-main\Hand-Gesture-GCN-main\mtm_augmented_data\Jabil\OneDrive_1_22-11-2024\20241123_230053.mp4")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

keypoints_buffer = []
frame_idx = 0
clip_len = 10  # Temporal length

# MediaPipe Hands
with mp_hands.Hands(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("End of video or failed to read frame.")
            break

        frame_idx += 1

        # Process frame with MediaPipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract hand keypoints
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                keypoints = landmark2nparray(hand_landmarks)
                keypoints_buffer.append(keypoints)
        else:
            keypoints_buffer.append(np.zeros((21, 2)))  # Append zeros if no hands detected

        # Perform prediction every `clip_len` frames
        if len(keypoints_buffer) >= clip_len:
            fake_anno = create_fake_anno(keypoints_buffer[-clip_len:], clip_len=clip_len)
            processed_sample = test_pipeline(fake_anno)
            input_tensor = processed_sample['imgs'][None].to(next(recognizer.parameters()).device)

            with torch.no_grad():
                prediction = recognizer(input_tensor, return_loss=False)[0]
                action = np.argmax(prediction)
                action_name = GestureDataset.label_names[action]

                # Display predictions
                cv2.putText(image, f"Action: {action_name}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(f"Frame {frame_idx}: {action_name}")

        # Display video
        cv2.imshow('PYSKL Gesture Demo [Press ESC to Exit]', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
