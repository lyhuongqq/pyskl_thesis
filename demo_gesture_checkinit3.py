import cv2
import mediapipe as mp
import numpy as np
import torch
from pyskl.apis import init_recognizer
from pyskl.datasets.pipelines import Compose
from pyskl.datasets import GestureDataset
from pyskl.smp import h2r

# Initialize MediaPipe and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def landmark2nparray(landmark):
    """Convert MediaPipe landmarks to NumPy array (x, y, z)."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmark.landmark])


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
        # Pad with zeros if insufficient frames
        pad_len = clip_len - len(keypoint)
        pad = np.zeros((pad_len, *keypoint.shape[1:]), dtype=np.float32)
        keypoint = np.concatenate([pad, keypoint], axis=0)

    # Add batch and person dimensions
    keypoint = keypoint[None]  # Add batch dimension
    return dict(keypoint=keypoint, total_frames=clip_len, modality="Pose")


def create_fake_anno_empty(clip_len=10):
    """Create a fake annotation dictionary with empty keypoints."""
    return dict(
        keypoint=np.zeros([1, clip_len, 21, 3], dtype=np.float32),  # Shape: (1, 10, 21, 3)
        total_frames=clip_len,
        modality="Pose",
    )


# Initialize the recognizer
recognizer = init_recognizer(
    r"D:\pyskl-main\pyskl-main\config_STGCN.py",
    r"D:\pyskl-main\work_dirs\stgcn_j_50\epoch_24.pth",
    device="cpu",
)
recognizer.eval()
cfg = recognizer.cfg
test_pipeline = Compose(cfg.test_pipeline)

# Debugging: Check the pipeline structure
print("Test pipeline structure:", cfg.test_pipeline)

# Test the recognizer with a dummy annotation
fake_anno = create_fake_anno_empty()
processed_sample = test_pipeline(fake_anno)
print("Processed sample shape:", processed_sample["keypoint"].shape)

# Open the video file
cap = cv2.VideoCapture(r"D:\Hand-GCN-main\Hand-Gesture-GCN-main\mtm_augmented_data\2.mp4")

if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

frame_idx = 0
predict_per_nframe = 2
keypoints_buffer = []
results_buffer = []
plate = "03045E-023E8A-0077B6-0096C7-00B4D8-48CAE4-90E0EF".split("-")
plate = [h2r(x)[::-1] for x in plate]

# Track the frame labels
frame_labels = []

with mp_hands.Hands(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("End of video or failed to read frame.")
            break

        frame_idx += 1

        # Process the frame with MediaPipe
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        keypoints = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand = landmark2nparray(hand_landmarks)
                keypoints.append(hand)

                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                )
        else:
            print(f"No hands detected at frame {frame_idx}.")

        # Append detected keypoints or zero keypoints for this frame
        if keypoints:
            keypoints_buffer.append(keypoints[-1])  # Use the last detected hand
        else:
            keypoints_buffer.append(np.zeros((21, 3)))  # Append zeros for missing data

        # Perform predictions at regular intervals
        if frame_idx % predict_per_nframe == 0 and len(keypoints_buffer) >= 10:
            try:
                sample = create_fake_anno(keypoints_buffer, keypoints_buffer[-1])
                processed_sample = test_pipeline(sample)
                sample_tensor = processed_sample["keypoint"].to("cpu")
                with torch.no_grad():
                    prediction = recognizer(sample_tensor, return_loss=False)[0]
                    action = np.argmax(prediction)
                    action_name = GestureDataset.label_names[action]
                    results_buffer.append(f"Frame {frame_idx}: {action_name} ({prediction[action]:.3f})")
                    frame_labels.append((frame_idx, action_name))
            except Exception as e:
                print(f"Error during prediction at frame {frame_idx}: {e}")

        # Display predictions on the frame
        for i, (action_label, color) in enumerate(zip(results_buffer[::-1][:7], plate)):
            cv2.putText(
                image,
                action_label,
                (10, 24 + i * 24),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                color,
                1,
            )

        # Show the frame
        cv2.imshow("Gesture Recognition Demo [Press ESC to Exit]", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

# Print frame labels for debugging
print("\nFrame Labels:")
for frame, label in frame_labels:
    print(f"Frame {frame}: {label}")
