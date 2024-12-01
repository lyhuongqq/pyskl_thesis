import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
video_path = r"D:\Hand-GCN-main\Hand-Gesture-GCN-main\mtm_augmented_data\2.mp4"

cap = cv2.VideoCapture(video_path)
with mp_hands.Hands(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("End of video or failed to read frame.")
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            print("Hand detected.")
        else:
            print("No hands detected.")
cap.release()
