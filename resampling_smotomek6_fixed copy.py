import mmcv
import numpy as np
from imblearn.combine import SMOTETomek
import os
from collections import Counter

def resample_pickle_data(input_file, train_output_file, val_output_file):
    """Resample data from a pickle file and save train and validation data to separate files.

    Args:
        input_file (str): Path to the input pickle file.
        train_output_file (str): Path to save the resampled training pickle file.
        val_output_file (str): Path to save the resampled validation pickle file.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found.")

    # Load data
    data = mmcv.load(input_file)
    annotations = data['annotations']
    split = data.get('split', None)

    if not annotations or 'keypoint' not in annotations[0]:
        raise ValueError("The input pickle file does not contain keypoint data.")

    print(f"Loaded {len(annotations)} annotations from {input_file}.")

    # Define label to action mapping
    label_to_action = {0: "Grasp", 1: "Move", 2: "Position", 3: "Release", 4: "Reach", 5: "Negative"}

    # Separate annotations based on split
    train_annotations = [ann for ann in annotations if ann['frame_dir'] in split['train']]
    val_annotations = [ann for ann in annotations if ann['frame_dir'] in split['val']]

    def print_class_distribution(labels, label_to_action, message):
        class_counts = Counter(labels)
        print(message)
        for label, count in sorted(class_counts.items()):
            print(f"  {label_to_action[label]} ({label}): {count}")

    def resample_annotations(annotations, synthetic=False):
        print(f"Processing {len(annotations)} annotations.")
        max_frames = max(ann['keypoint'].shape[1] for ann in annotations)
        max_keypoints = annotations[0]['keypoint'].shape[2:]  # Keypoint dimensions

        features = []
        labels = []
        meta_info = []  # Store metadata like frame_dir and img_shape

        for ann in annotations:
            keypoints = np.array(ann['keypoint'], dtype=np.float32)
            padded_keypoints = np.zeros((1, max_frames, *max_keypoints), dtype=np.float32)
            frames_to_copy = min(keypoints.shape[1], max_frames)
            padded_keypoints[:, :frames_to_copy, :, :] = keypoints[:, :frames_to_copy, :, :]

            features.append(padded_keypoints.flatten())
            labels.append(int(ann['label']))  # Ensure label is int
            meta_info.append({
                'frame_dir': ann['frame_dir'],
                'img_shape': ann['img_shape'],
                'original_shape': ann['original_shape']
            })

        features = np.array(features, dtype=np.float32)  # Ensure features are float32
        labels = np.array(labels, dtype=np.int32)  # Ensure labels are int32

        print_class_distribution(labels, label_to_action, "Class distribution before resampling:")

        # Apply SMOTE-Tomek globally across all classes
        smote_tomek = SMOTETomek(random_state=42)
        try:
            resampled_features, resampled_labels = smote_tomek.fit_resample(features, labels)
        except ValueError as e:
            print(f"Error during SMOTE-Tomek resampling: {e}")
            return annotations  # Return original if resampling fails

        print_class_distribution(resampled_labels, label_to_action, "Class distribution after resampling:")

        # Rebuild annotations from resampled data
        resampled_annotations = []
        for i, (feature, label) in enumerate(zip(resampled_features, resampled_labels)):
            reshaped_keypoints = feature.reshape((1, max_frames, *max_keypoints))
            total_frames = reshaped_keypoints.shape[1]
            meta = meta_info[i % len(meta_info)]

            # Keep the original name for original data and update synthetic data
            frame_dir = meta['frame_dir'] if i < len(annotations) else f"{meta['frame_dir']}_synthetic_{i}"

            resampled_annotations.append({
                'frame_dir': str(frame_dir),
                'total_frames': total_frames,
                'img_shape': meta['img_shape'],
                'original_shape': meta['original_shape'],
                'label': int(label),
                'keypoint': reshaped_keypoints.astype(np.float32),
                'keypoint_score': np.zeros_like(reshaped_keypoints[..., 0], dtype=np.float32) if label == 5 else np.ones_like(reshaped_keypoints[..., 0], dtype=np.float32)
            })

        return resampled_annotations

    # Resample train and validation annotations separately
    resampled_train_annotations = resample_annotations(train_annotations, synthetic=True)
    resampled_val_annotations = resample_annotations(val_annotations, synthetic=True)

    # Debugging: Print sample data to verify correctness
    print("Sample resampled train annotation:", resampled_train_annotations[0])
    print("Sample resampled val annotation:", resampled_val_annotations[0])

    # Save train and validation pickle files
    train_data = {'annotations': resampled_train_annotations, 'split': {'train': split['train']}}
    val_data = {'annotations': resampled_val_annotations, 'split': {'val': split['val']}}

    mmcv.dump(train_data, train_output_file)
    mmcv.dump(val_data, val_output_file)

    print(f"Resampled train data saved to {train_output_file}.")
    print(f"Resampled validation data saved to {val_output_file}.")

# Example usage
resample_pickle_data(
    input_file=r"/root/pyskl_thesis/hand_pose_dataset_aug_6.pkl",
    train_output_file=r"/root/pyskl_thesis/smotetomek_aug_8_train5.pkl",
    val_output_file=r"/root/pyskl_thesis/smotetomek_aug_8_val5.pkl"
)
