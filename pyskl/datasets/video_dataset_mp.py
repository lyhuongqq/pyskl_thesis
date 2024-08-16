import cv2
import mediapipe as mp
import numpy as np
from torch.utils.data import Dataset

mp_pose = mp.solutions.pose

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = self._load_video(video_path)
        skeletons = self._extract_skeletons(frames)

        if self.transform:
            skeletons = self.transform(skeletons)

        return skeletons, label

    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def _extract_skeletons(self, frames):
        skeletons = []
        for frame in frames:
            results = self.pose.process(frame)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                skeleton = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
                skeletons.append(skeleton)
        return np.array(skeletons)
