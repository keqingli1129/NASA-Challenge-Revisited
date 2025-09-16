import torch

def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))
        print("CUDA version (PyTorch):", torch.version.cuda)
        print("CUDA device name:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    main()