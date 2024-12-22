import torch
print("CUDA Available:", torch.cuda.is_available())  # Should print True if CUDA is available
print("CUDA Version:", torch.version.cuda)           # Should print '10.1'
