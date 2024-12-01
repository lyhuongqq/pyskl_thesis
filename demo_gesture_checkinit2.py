import cv2
import mediapipe as mp
import numpy as np
import torch
from pyskl.apis import init_recognizer
from pyskl.datasets.pipelines import Compose
from pyskl.datasets import GestureDataset
from pyskl.smp import h2r

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def landmark2nparray(landmark):
    """Convert MediaPipe hand landmarks to a NumPy array."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmark.landmark])


def kp2box(kpt, margin=0.2):
    """Calculate bounding box from keypoints with margin."""
    min_x, max_x = min(kpt[:, 0]), max(kpt[:, 0])
    min_y, max_y = min(kpt[:, 1]), max(kpt[:, 1])
    c_x, c_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    w, h = max_x - min_x, max_y - min_y
    w2, h2 = w * (1 + margin) / 2, h * (1 + margin) / 2
    return (max(0, c_x - w2), max(0, c_y - h2), min(1, c_x + w2), min(1, c_y + h2))


def create_fake_anno(history, keypoint, clip_len=10):
    """Create a fake annotation dictionary compatible with the pipeline."""
    results = [keypoint]
    for frame in history[::-1]:
        if len(results) >= clip_len:
            break
        results.append(frame)

    keypoint = np.array(results[::-1], dtype=np.float32)  # Reverse to correct order
    keypoint = keypoint[None]  # Add batch dimension

    return dict(keypoint=keypoint, total_frames=keypoint.shape[1], modality="Pose")


def create_fake_anno_empty(clip_len=10):
    """Create a fake annotation dictionary with empty keypoints."""
    return dict(
        keypoint=np.zeros([1, clip_len, 21, 3], dtype=np.float32),  # Adjust to 3D keypoints
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

# Video capture and processing
cap = cv2.VideoCapture(r"D:\Hand-GCN-main\Hand-Gesture-GCN-main\mtm_augmented_data\2.mp4")

with mp_hands.Hands(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5, max_num_hands=1) as hands:
    frame_idx = 0
    predict_per_nframe = 2
    keypoints_buffer = []
    results_buffer = []
    plate = "03045E-023E8A-0077B6-0096C7-00B4D8-48CAE4-90E0EF".split("-")
    plate = [h2r(x)[::-1] for x in plate]

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

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                )

        # Append keypoints to the buffer
        keypoints_buffer.append(keypoints)

        # Perform predictions at regular intervals
        if frame_idx % predict_per_nframe == 0 and len(keypoints) > 0:
            try:
                sample = create_fake_anno(keypoints_buffer, keypoints[-1])
                processed_sample = test_pipeline(sample)
                sample_tensor = processed_sample["keypoint"][None].to("cpu")
                with torch.no_grad():
                    prediction = recognizer(sample_tensor, return_loss=False)[0]
                    action = np.argmax(prediction)
                    action_name = GestureDataset.label_names[action]
                    results_buffer.append(f"{action_name}: {prediction[action]:.3f}")
            except Exception as e:
                print(f"Error during prediction: {e}")

        # Display results
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
