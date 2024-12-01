import torch

def load_and_inspect_checkpoint(checkpoint_path):
    """
    Load a PyTorch checkpoint and inspect its keys and structure.
    """
    try:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        print("Checkpoint successfully loaded!")
        print("Available keys in the checkpoint:", list(checkpoint.keys()))

        # Inspect model state_dict
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print("\nKeys in state_dict:")
            print(list(state_dict.keys()))
            print(f"\nNumber of parameters in checkpoint: {len(state_dict)}")

        # Inspect additional info (e.g., epoch, optimizer)
        if "meta" in checkpoint:
            print("\nMeta information in checkpoint:")
            print(checkpoint["meta"])
        if "optimizer" in checkpoint:
            print("\nOptimizer state_dict keys:")
            print(list(checkpoint["optimizer"].keys()))

    except Exception as e:
        print(f"Error loading checkpoint: {e}")


# Path to the checkpoint file
checkpoint_path = r"D:\pyskl-main\pyskl-main\demo\hagrid.pth"

# Inspect the checkpoint
load_and_inspect_checkpoint(checkpoint_path)
