import cv2
import mediapipe as mp
import numpy as np
import torch
from pyskl.apis import init_recognizer
from pyskl.datasets import GestureDataset
from pyskl.datasets.pipelines import Compose

# Initialize MediaPipe and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def landmark2nparray(landmark):
    """Convert MediaPipe landmarks to NumPy array (x, y)."""
    return np.array([[lm.x, lm.y] for lm in landmark.landmark])


def create_fake_anno(history, keypoint, clip_len=10):
    """Create a fake annotation dictionary compatible with the pipeline."""
    results = [keypoint]
    for frame in history[::-1]:
        if len(results) >= clip_len:
            break
        results.append(frame)

    # Reverse and convert to NumPy array
    keypoint = np.array(results[::-1], dtype=np.float32)
    if len(keypoint) < clip_len:
        pad_len = clip_len - len(keypoint)
        pad = np.zeros((pad_len, *keypoint.shape[1:]), dtype=np.float32)
        keypoint = np.concatenate([pad, keypoint], axis=0)

    # Add batch and person dimensions
    return dict(
        keypoint=keypoint[None],  # Shape: (1, clip_len, num_keypoints, 2)
        total_frames=clip_len,
        start_index=0,  # Fixed start index for pipeline compatibility
        modality="Pose",
        label=0,  # Default label for inference
    )


# Initialize recognizer
recognizer = init_recognizer(
    r"D:\pyskl-main\pyskl-main\config_STGCN.py",
    r"D:\pyskl-main\work_dirs\stgcn_j_50\epoch_24.pth",
    device="cpu",
)
recognizer.eval()
cfg = recognizer.cfg
test_pipeline = Compose(cfg.test_pipeline)

cap = cv2.VideoCapture(r"D:\Hand-GCN-main\Hand-Gesture-GCN-main\mtm_augmented_data\2.mp4")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

keypoints_buffer = []
frame_idx = 0
predict_per_nframe = 2

with mp_hands.Hands(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("End of video or failed to read frame.")
            break

        frame_idx += 1
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand = landmark2nparray(hand_landmarks)
                keypoints_buffer.append(hand)
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
        else:
            keypoints_buffer.append(np.zeros((21, 2)))  # Append zeros for missing keypoints

        if len(keypoints_buffer) >= 10 and frame_idx % predict_per_nframe == 0:
            try:
                sample = create_fake_anno(keypoints_buffer, keypoints_buffer[-1])
                processed_sample = test_pipeline(sample)
                sample_tensor = processed_sample["keypoint"].to("cpu")
                with torch.no_grad():
                    prediction = recognizer(sample_tensor, return_loss=False)[0]
                    action = np.argmax(prediction)
                    action_name = GestureDataset.label_names[action]

                    # Display prediction on the video frame
                    cv2.putText(
                        image,
                        f"Action: {action_name} ({prediction[action]:.2f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    print(f"Frame {frame_idx}: {action_name} ({prediction[action]:.2f})")
            except Exception as e:
                print(f"Error during prediction at frame {frame_idx}: {e}")

        cv2.imshow("Video", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
