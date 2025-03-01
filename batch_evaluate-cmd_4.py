import os
import subprocess
import pandas as pd
import re

# Paths and configurations
#config_path = r"D:\pyskl-main\pyskl-main\config_slow_24oct.py"
#config_path = r"D:\pyskl-main\pyskl-main\work_dirs\slow_100_tomek_Ja_combine\config_slow_24oct.py"
config_path = r"D:\pyskl-main\pyskl-main\config_c3d_smotek.py"
checkpoint_folder = r"D:\pyskl-main\pyskl-main\work_dirs\c3d_paper_testframe"#r"D:\pyskl-main\work_dirs\c3d_ep100_Jabil"
#checkpoint_folder = r"D:\pyskl-main\pyskl-main\work_dirs\c3d_hybrid_Ja"
output_folder = r"D:\pyskl-main\pyskl-main\work_dirs\c3d_paper_testframe/output_results"
output_csv_path = os.path.join(output_folder, "c3d_ep10_Ja_val_paper_len10_clip1_loss_results.csv")

# Create output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Initialize list to collect results for each checkpoint
results = []

# Define a regex pattern to match key-value pairs for metrics
metric_pattern = re.compile(r"^(\w+(_\w+)*):\s*([0-9]*\.?[0-9]+)$")

# Iterate over checkpoint files in the folder
for checkpoint_file in sorted(os.listdir(checkpoint_folder)):
    if checkpoint_file.endswith(".pth"):
        # Define the checkpoint path
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_file)
        output_json_path = os.path.join(output_folder, f"{checkpoint_file.replace('.pth', '')}_results.json")
        
        # Construct the command to run the evaluation
        command = [
            "python",
            r"D:\pyskl-main\pyskl-main\test-cpu.py",
            config_path,
            "--checkpoint",
            checkpoint_path,
            "--out",  # Out flag
            output_json_path,  # The path to the output JSON
            "--eval",
            "top_k_accuracy",
            "mean_class_accuracy",
            "mean_average_precision",
            "recall"
        
        ]

        print(f"Running: {' '.join(command)}")
        try:
            # Run the subprocess and capture any printed output
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            output = result.stdout  # Captured output from evaluate function

            # Process the output to extract only key-value metric pairs
            eval_data = {}
            for line in output.splitlines():
                line = line.strip()
                match = metric_pattern.match(line)
                if match:
                    key = match.group(1).strip()
                    value = match.group(3).strip()
                    # Try to convert value to float if possible
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep value as a string if it cannot be converted
                    eval_data[key] = value
            
            # Add the epoch/checkpoint name to the data
            eval_data["epoch"] = checkpoint_file.replace('.pth', '')
            results.append(eval_data)

        except subprocess.CalledProcessError as e:
            print(f"Error during testing for {checkpoint_file}: {e}")
            print(e.output)

# Save results to CSV
if results:
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")
else:
    print("No results to save.")
