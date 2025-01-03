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

    def resample_annotations(annotations):
        print(f"Processing {len(annotations)} annotations.")
        max_frames = max(ann['keypoint'].shape[1] for ann in annotations)
        max_keypoints = annotations[0]['keypoint'].shape[2:]  # Keypoint dimensions

        features = []
        labels = []
        meta_info = []  # Store metadata like frame_dir and img_shape

        for ann in annotations:
            keypoints = ann['keypoint'][0]  # Remove person dimension for flattening
            keypoints_flattened = keypoints.reshape(-1, np.prod(max_keypoints))  # Flatten spatial dimensions
            features.extend(keypoints_flattened)
            labels.extend([ann['label']] * keypoints_flattened.shape[0])  # Replicate labels for each frame
            meta_info.append({
                'frame_dir': ann['frame_dir'],
                'img_shape': ann['img_shape'],
                'original_shape': ann['original_shape'],
                'keypoint_shape': keypoints.shape,
                'label': ann['label']  # Ensure label is captured here
            })

        features = np.array(features, dtype=np.float32)  # Ensure features are float32
        labels = np.array(labels, dtype=np.int32)  # Ensure labels are int32

        print_class_distribution(labels, label_to_action, "Class distribution before resampling:")

        # Apply SMOTETomek to keypoints
        smote_tomek = SMOTETomek(random_state=42)
        try:
            resampled_features, resampled_labels = smote_tomek.fit_resample(features, labels)
        except ValueError as e:
            print(f"Error during SMOTETomek resampling: {e}")
            return annotations  # Return original if resampling fails

        print_class_distribution(resampled_labels, label_to_action, "Class distribution after resampling:")

        # Rebuild annotations from resampled data
        resampled_annotations = []
        start_idx = 0

        for meta in meta_info:
            frame_count = meta['keypoint_shape'][0]
            reshaped_keypoints = resampled_features[start_idx:start_idx + frame_count].reshape(
                (1, frame_count, *max_keypoints)
            )

            # Ensure class 5 (Negative) has all zeros for keypoints and keypoint scores
            if meta['label'] == 5:
                reshaped_keypoints.fill(0)
                keypoint_score = np.zeros((1, frame_count, max_keypoints[0]), dtype=np.float32)
            else:
                keypoint_score = np.ones((1, frame_count, max_keypoints[0]), dtype=np.float32)

            resampled_annotations.append({
                'frame_dir': meta['frame_dir'],
                'total_frames': frame_count,
                'img_shape': meta['img_shape'],
                'original_shape': meta['original_shape'],
                'label': meta['label'],  # Use the label stored in meta_info
                'keypoint': reshaped_keypoints.astype(np.float32),
                'keypoint_score': keypoint_score
            })
            start_idx += frame_count

        return resampled_annotations

    # Resample train and validation annotations separately
    resampled_train_annotations = resample_annotations(train_annotations)
    resampled_val_annotations = resample_annotations(val_annotations)

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
    input_file=r"/root/pyskl_thesis/hand_pose_dataset_combined_2Dec_modified.pkl",
    train_output_file=r"/root/pyskl_thesis/test/smotetomek_aug_8_train.pkl",
    val_output_file=r"/root/pyskl_thesis/test/smotetomek_aug_8_val.pkl"
)
