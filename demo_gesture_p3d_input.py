import cv2
import mediapipe as mp
import numpy as np
import torch
import csv

from pyskl.apis import init_recognizer
from pyskl.datasets import GestureDataset
from pyskl.datasets.pipelines import Compose
from pyskl.smp import h2r

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def landmark2nparray(landmark):
    ret = np.array([[lm.x, lm.y] for lm in landmark.landmark])
    assert ret.shape == (21, 2)
    return ret

def kp2box(kpt, margin=0.2):
    min_x, max_x = min(kpt[:, 0]), max(kpt[:, 0])
    min_y, max_y = min(kpt[:, 1]), max(kpt[:, 1])
    c_x, c_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    w, h = max_x - min_x, max_y - min_y
    w2, h2 = w * (1 + margin) / 2, h * (1 + margin) / 2
    min_x = max(0, c_x - w2)
    min_y = max(0, c_y - h2)
    max_x = min(1, c_x + w2)
    max_y = min(1, c_y + h2)
    return (min_x, min_y, max_x - min_x, max_y - min_y)

def flip_box(box):
    return (1 - box[0] - box[2], box[1], box[2], box[3])

def create_fake_anno(history, keypoint, bbox, clip_len=10):
    from mmdet.core import BboxOverlaps2D
    bbox = torch.tensor(bbox)[None]
    iou_calc = BboxOverlaps2D()
    results = [keypoint]

    for frame in history[::-1]:
        anchors = torch.tensor([x[1] for x in frame])
        if anchors.shape[0] == 0:
            break
        ious = iou_calc(bbox, anchors)[0]
        idx = torch.argmax(ious)
        if ious[idx] >= 0.5:
            results.append(frame[idx][0])
            bbox = anchors[idx: idx + 1]
        else:
            break
        if len(results) >= clip_len:
            break

    keypoint = np.array(results[::-1], dtype=np.float32)[None]
    total_frames = keypoint.shape[1]
    return dict(
        keypoint=keypoint,
        total_frames=total_frames,
        frame_dir='NA',
        label=0,
        start_index=0,
        modality='Pose',
        test_mode=True
    )

def create_fake_anno_empty(clip_len=10):
    return dict(
        keypoint=np.zeros([1, clip_len, 21, 2], dtype=np.float32),
        total_frames=clip_len,
        frame_dir='NA',
        label=0,
        start_index=0,
        modality='Pose',
        test_mode=True
    )

# Replace webcam input with video input
video_path = r"D:\Hand-GCN-main\Hand-Gesture-GCN-main\mtm_augmented_data\2.mp4"  # Update with your video file path
cap = cv2.VideoCapture(video_path)
output_csv_path = 'D:\pyskl_thesis\output_predictions.csv'

with open(output_csv_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame', 'Action', 'Confidence'])  # Header for CSV

    with mp_hands.Hands(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5, max_num_hands=1) as hands:
        
        #recognizer = init_recognizer(
        #    r"D:\pyskl_thesis\demo_config_pose3dconv.py", r"D:\pyskl-main\output_results\pose3d\test_100\epoch_11_results.json", 
        #    device='cpu'
        #)
        recognizer = init_recognizer(r"D:\pyskl_thesis\demo_config_pose3dconv.py", r"D:\pyskl-main\work_dirs\test_100\posec3d\test_aug8\epoch_11.pth", device='cpu')
        recognizer.eval()
        cfg = recognizer.cfg
        device = next(recognizer.parameters()).device
        test_pipeline = Compose(cfg.test_pipeline)

        fake_anno = create_fake_anno_empty()
        sample = test_pipeline(fake_anno)['keypoint'][None].to(device)
        prediction = recognizer(sample, return_loss=False)[0]

        keypoints_buffer = []
        results_buffer = []
        frame_idx = 0
        predict_per_nframe = 2

        while cap.isOpened():
            success, image = cap.read()

            if not success:
                print('Video processing complete.')
                break

            frame_idx += 1

            try:
                # Convert the image to RGB as required by MediaPipe
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # Draw the hand annotations on the image if any hands are detected
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                boxes = []
                keypoints = []

                if results.multi_hand_landmarks:
                    
                    print(f"Detected {len(results.multi_hand_landmarks)} hand(s)")
                    for hand_landmarks in results.multi_hand_landmarks:
                        hand = landmark2nparray(hand_landmarks)
                        box = kp2box(hand)
                        boxes.append(box)
                        keypoints.append((hand, box))

                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )

                h, w, _ = image.shape
                for box in boxes:
                    box = flip_box(box)
                    x, y, w, h = [int(v * s) for v, s in zip(box, (w, h, w, h))]
                    cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=w // 120)
                # Before model prediction:
                if len(keypoints) == 0:
                 print(f"No keypoints detected for frame {frame_idx}")
                else:
                 print(f"Detected keypoints for frame {frame_idx}")

# After model prediction:
                 print(f"Frame {frame_idx}: Prediction = {prediction}, Action = {action_name}, Confidence = {confidence:.3f}")
                # Perform predictions if it's time to do so
                if frame_idx % predict_per_nframe == 0:
                    if len(keypoints) == 0:
                        results_buffer.append('No hands detected')
                        csv_writer.writerow([frame_idx, 'No hands detected', 0.0])
                    else:
                        for keypoint, bbox in keypoints:
                            with torch.no_grad():
                                sample = create_fake_anno(keypoints_buffer, keypoint, bbox)
                                sample = test_pipeline(sample)['keypoint'][None].to(device)
                                prediction = recognizer(sample, return_loss=False)[0]
                                action = np.argmax(prediction)
                                action_name = GestureDataset.label_names[action]
                                confidence = prediction[action]
                                results_buffer.append(f'{action_name}: {confidence:.3f}')
                                csv_writer.writerow([frame_idx, action_name, confidence])

                FONTFACE = cv2.FONT_HERSHEY_DUPLEX
                FONTSCALE = 0.6
                THICKNESS = 1
                LINETYPE = 1
                for i, action_label in enumerate(results_buffer[::-1][:7]):
                    cv2.putText(image, action_label, (10, 24 + i * 24), FONTFACE, FONTSCALE, (255, 255, 255), THICKNESS, LINETYPE)

                keypoints_buffer.append(keypoints)

                # Display the image with OpenCV
                cv2.imshow('PYSKL Gesture Demo [Press ESC to Exit]', image)
                if cv2.waitKey(5) & 0xFF == 27:  # Exit if 'ESC' is pressed
                    break
            except Exception as e:
                print(f"Error occurred: {e}")

cap.release()
cv2.destroyAllWindows()
