import torch
print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"Using device: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "No GPU found, using CPU.")
