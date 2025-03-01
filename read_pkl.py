
import pickle
import numpy as np
# Load the PKL file
with open("/root/pyskl_thesis/aug_6_frame_wise_dataset.pkl", "rb") as f:
    data = pickle.load(f)

print("PKL Data Type:", type(data))
# If PKL is a dictionary, print its keys
if isinstance(data, dict):
    print("PKL Keys:", data.keys())
    # If it contains 'annotations', check the first sample
    if "annotations" in data:
        print("First Annotation Sample:", type(data["annotations"][0]))
        print("Annotation Keys:", data["annotations"][0].keys())
exit()
