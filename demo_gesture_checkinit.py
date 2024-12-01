import cv2
import mediapipe as mp
import numpy as np
import torch
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

    # frame contains tuples of (keypoint, bbox)
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
        test_mode=True,
    )


def create_fake_anno_empty(clip_len=10):
    return dict(
        keypoint=np.zeros([1, clip_len, 21, 2], dtype=np.float32),
        total_frames=clip_len,
        frame_dir='NA',
        label=0,
        start_index=0,
        modality='Pose',
        test_mode=True,
    )


# For video input:
cap = cv2.VideoCapture(r"D:\Hand-GCN-main\Hand-Gesture-GCN-main\mtm_augmented_data\2.mp4")
with mp_hands.Hands(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5, max_num_hands=1) as hands:
    recognizer = init_recognizer(
        r"D:\pyskl-main\pyskl-main\config_STGCN.py",
        r"D:\pyskl-main\work_dirs\stgcn_j_50\epoch_24.pth",
        device="cpu",
    )
    recognizer.eval()
    cfg = recognizer.cfg
    device = next(recognizer.parameters()).device
    test_pipeline = Compose(cfg.test_pipeline)

    print("Pipeline steps:", cfg.test_pipeline)

    # Test if recognizer works
    fake_anno = create_fake_anno_empty()
    try:
        sample = test_pipeline(fake_anno)
        print("Sample after pipeline:", sample)
        print("Keypoint shape after pipeline:", sample["keypoint"].shape)

        sample_tensor = sample["keypoint"][None].to(device)
        with torch.no_grad():
            prediction = recognizer(sample_tensor, return_loss=False)
            print("Model prediction shape:", prediction.shape)
            print("Model prediction:", prediction)
    except Exception as e:
        print(f"Error during pipeline or recognizer test: {e}")

    keypoints_buffer = []
    results_buffer = []
    frame_idx = 0
    predict_per_nframe = 2
    plate = "03045E-023E8A-0077B6-0096C7-00B4D8-48CAE4-90E0EF".split("-")
    plate = [h2r(x)[::-1] for x in plate]

    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("End of video or failed to read frame.")
            break

        frame_idx += 1

        try:
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            boxes = []
            keypoints = []

            if results.multi_hand_landmarks:
                print(f"Detected {len(results.multi_hand_landmarks)} hand(s).")
                for hand_landmarks in results.multi_hand_landmarks:
                    hand = landmark2nparray(hand_landmarks)
                    print("Keypoints detected:", hand)
                    box = kp2box(hand)
                    print("Bounding box:", box)
                    boxes.append(box)
                    keypoints.append((hand, box))

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )
            else:
                print("No hands detected in this frame.")

            if frame_idx % predict_per_nframe == 0:
                if len(keypoints) == 0:
                    results_buffer.append("No hands detected")
                else:
                    for keypoint, bbox in keypoints:
                        try:
                            sample = create_fake_anno(keypoints_buffer, keypoint, bbox)
                            sample = test_pipeline(sample)["keypoint"][None].to(device)
                            prediction = recognizer(sample, return_loss=False)[0]
                            action = np.argmax(prediction)
                            action_name = GestureDataset.label_names[action]
                            results_buffer.append(f"{action_name}: {prediction[action]:.3f}")
                        except Exception as e:
                            print(f"Error during prediction: {e}")

            FONTFACE = cv2.FONT_HERSHEY_DUPLEX
            FONTSCALE = 0.6
            THICKNESS = 1
            LINETYPE = 1
            for i, (action_label, color) in enumerate(zip(results_buffer[::-1][:7], plate)):
                cv2.putText(
                    image,
                    action_label,
                    (10, 24 + i * 24),
                    FONTFACE,
                    FONTSCALE,
                    color,
                    THICKNESS,
                    LINETYPE,
                )

            keypoints_buffer.append(keypoints)

            cv2.imshow("PYSKL Gesture Demo [Press ESC to Exit]", image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        except Exception as e:
            print(f"Error occurred during frame processing: {e}")

cap.release()
cv2.destroyAllWindows()
