import pickle
import csv

# Step 1: Load the pickle file
with open(r"D:\pyskl-main\smoteenn_2Dec_val_Ja.pkl", 'rb') as f:
    data = pickle.load(f)

# Step 2: Process the data
# Assuming 'data' is a dictionary with 'annotations' containing the required information
annotations = data['annotations']

# Step 3: Write to a CSV file
with open('smoteenn_2Dec_val_Ja.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write header
    writer.writerow(['frame_dir', 'total_frames', 'img_shape','label','keypoints', 'keypoint_score']) #, 'img_shape',

    # Write data rows
    for annotation in annotations:
        frame_dir = annotation['frame_dir']
        total_frames = annotation['total_frames']
        img_shape = annotation['img_shape']
        label = annotation['label']
        keypoint = annotation['keypoint']
        keypoint_score = annotation['keypoint_score']

        # Assuming keypoints and keypoint_scores are lists of values, you might need to process them further
        writer.writerow([frame_dir, total_frames, img_shape, label, keypoint,keypoint_score])#, ,, keypoint_scores])

print("CSV file created successfully.")
