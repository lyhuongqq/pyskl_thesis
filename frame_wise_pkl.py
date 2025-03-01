import pickle
import numpy as np
import os

# Load the PKL file
with open("D:\pyskl-main\hybrid_aug_8_val.pkl", "rb") as f:
    data = pickle.load(f)

# Verify structure
if not isinstance(data, dict) or "annotations" not in data or "split" not in data:
    raise ValueError("Unexpected PKL structure! Expected dictionary with 'annotations' and 'split'.")

frame_wise_annotations = []
train_annotations = []
val_annotations = []

# Iterate through each video sample
for sample in data["annotations"]:
    frame_dir = sample["frame_dir"]  # Video directory
    total_frames = sample["total_frames"]
    img_shape = sample["img_shape"]
    original_shape = sample["original_shape"]
    label = sample["label"]
    keypoints = sample["keypoint"]  # (1, T, 21, 2)
    keypoint_score = sample["keypoint_score"]  # (1, T, 21)

    # Extract each frame as a separate sample
    for frame_idx in range(total_frames):
        frame_sample = {
            "frame_dir": os.path.join(frame_dir, f"frame_{frame_idx:04d}"),  # Unique frame name
            "total_frames": 1,  # Each sample is one frame
            "img_shape": img_shape,
            "original_shape": original_shape,
            "label": label,
            "keypoint": keypoints[:, frame_idx:frame_idx+1, :, :],  # Extract only this frame
            "keypoint_score": keypoint_score[:, frame_idx:frame_idx+1, :]  # Extract score
        }
        frame_wise_annotations.append(frame_sample)

        # Add to correct split
#        if frame_dir in data["split"]["train"]:
#            train_annotations.append(frame_sample)
        #elif frame_dir in data["split"]["val"]:
        if frame_dir in data["split"]["val"]:
            val_annotations.append(frame_sample)

# Ensure proper format with train/val splits
frame_wise_data = {
    "annotations": frame_wise_annotations,
    "split": {
        "train": [ann["frame_dir"] for ann in train_annotations],
        "val": [ann["frame_dir"] for ann in val_annotations],
    },
}

# Save the new frame-wise PKL
new_pkl_path = "D:\pyskl-main\hybrid_aug_8_val_frame_wise.pkl"
with open(new_pkl_path, "wb") as f:
    pickle.dump(frame_wise_data, f)

print(f"✅ Successfully created frame-wise dataset with {len(frame_wise_annotations)} frames!")
print(f"📂 Saved at: {new_pkl_path}")
