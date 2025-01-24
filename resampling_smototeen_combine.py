import mmcv
import numpy as np
from imblearn.combine import SMOTETomek
import os
from collections import Counter

def resample_pickle_data(input_file, train_output_file, val_output_file):
    """Resample keypoints and split into train/validation datasets."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found.")

    data = mmcv.load(input_file)
    annotations = data['annotations']

    if not annotations or 'keypoint' not in annotations[0]:
        raise ValueError("The input pickle file does not contain keypoint data.")

    print(f"Loaded {len(annotations)} annotations from {input_file}.")

    # Calculate total frames and target frames per class
    total_frames = sum(ann['total_frames'] for ann in annotations)
    train_target_frames_per_class = int((total_frames * 0.8) / 6)  # Assuming 6 classes
    val_target_frames_per_class = int((total_frames * 0.2) / 6)

    print(f"Target frames per class for training: {train_target_frames_per_class}")
    print(f"Target frames per class for validation: {val_target_frames_per_class}")

    def resample_by_label(annotations, target_frames_per_class):
        resampled_annotations = []
        combined_keypoints = []
        combined_labels = []

        # Combine all keypoints and labels
        for ann in annotations:
            keypoints = ann['keypoint'].reshape(-1, 2)  # Flatten keypoints
            combined_keypoints.extend(keypoints)
            combined_labels.extend([ann['label']] * len(keypoints))

        combined_keypoints = np.array(combined_keypoints, dtype=np.float32)
        combined_labels = np.array(combined_labels, dtype=np.int32)

        # Print original class distribution
        print("Original class distribution:", Counter(combined_labels))

        try:
            print("Applying SMOTETomek.")
            smotetomek = SMOTETomek()
            X_resampled, y_resampled = smotetomek.fit_resample(combined_keypoints, combined_labels)
            print(f"Resampling completed. Original size: {len(combined_keypoints)}, Resampled size: {len(X_resampled)}")
        except ValueError as e:
            print(f"SMOTETomek failed: {e}. Using original data.")
            X_resampled, y_resampled = combined_keypoints, combined_labels

        # Print resampled class distribution
        print("Resampled class distribution:", Counter(y_resampled))

        # Redistribute resampled keypoints back into annotations
        label_to_keypoints = {label: [] for label in set(y_resampled)}
        for point, label in zip(X_resampled, y_resampled):
            label_to_keypoints[label].append(point)

        for ann in annotations:
            label = ann['label']
            if label in label_to_keypoints:
                keypoints_needed = ann['total_frames'] * ann['keypoint'].shape[2]
                resampled_keypoints = np.array(label_to_keypoints[label][:keypoints_needed])

                if len(resampled_keypoints) < keypoints_needed:
                    print(f"Not enough resampled points for annotation {ann['frame_dir']}. Filling with original data.")
                    original_keypoints = ann['keypoint'].reshape(-1, 2)
                    if len(original_keypoints) > 0:
                        resampled_keypoints = np.vstack([
                            resampled_keypoints,
                            original_keypoints[:keypoints_needed - len(resampled_keypoints)]
                        ]) if len(resampled_keypoints) > 0 else original_keypoints[:keypoints_needed]

                resampled_keypoints = resampled_keypoints[:keypoints_needed]
                label_to_keypoints[label] = label_to_keypoints[label][keypoints_needed:]

                num_frames = len(resampled_keypoints) // ann['keypoint'].shape[2]
                resampled_keypoints = resampled_keypoints.reshape(
                    (1, num_frames, ann['keypoint'].shape[2], 2))

                ann['keypoint'] = resampled_keypoints
                ann['total_frames'] = num_frames  # Update the total_frames based on resampled data
                resampled_annotations.append(ann)

        return resampled_annotations

    # Split annotations into training and validation sets
    train_annotations = annotations[:int(0.8 * len(annotations))]
    val_annotations = annotations[int(0.8 * len(annotations)) :]

    # Resample keypoints for training and validation
    resampled_train_annotations = resample_by_label(train_annotations, train_target_frames_per_class)
    resampled_val_annotations = resample_by_label(val_annotations, val_target_frames_per_class)

    # Assign split keys
    split_data = {"train": [], "val": []}
    for ann in resampled_train_annotations:
        split_data["train"].append(ann['frame_dir'])
    for ann in resampled_val_annotations:
        split_data["val"].append(ann['frame_dir'])

    # Save resampled annotations
    train_data = {'split': split_data, 'annotations': resampled_train_annotations}
    val_data = {'split': split_data, 'annotations': resampled_val_annotations}

    mmcv.dump(train_data, train_output_file)
    mmcv.dump(val_data, val_output_file)

    print(f"Resampled train data saved to {train_output_file}.")
    print(f"Resampled validation data saved to {val_output_file}.")


# Combine Labels Temporarily for SMOTETomek

# Example usage
#resample_pickle_data(
#    input_file=r"/root/pyskl_thesis/hand_pose_dataset_combined_2Dec_modified.pkl",
#    train_output_file=r"/root/pyskl_thesis/test/smoteenn_2dec_train.pkl",
#    val_output_file=r"/root/pyskl_thesis/test/smoteenn_2dec_val.pkl")

#Combine Labels Temporarily for SMOTEENN
# Example usage
resample_pickle_data(
    input_file=r"D:\pyskl-main\hand_pose_dataset_combined_2Dec_modified.pkl",
    train_output_file=r"D:\pyskl-main\test_pkl\smotenn_train_2dec_Ja_combine.pkl",
    val_output_file=r"D:\pyskl-main\test_pkl\smotenn_val_2dec_Ja_combine.pkl")

#resample_pickle_data(
#    input_file=r"/root/pyskl_thesis/hand_pose_dataset_aug_6.pkl",
#    train_output_file=r"D:\pyskl-main\test_pkl\smotenn_train_24Jan_paper_combine.pkl",
#    val_output_file=r"D:\pyskl-main\test_pkl\smotenn_val_24Jan_paper_combine.pkl")
