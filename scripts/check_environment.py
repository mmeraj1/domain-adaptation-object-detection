"""Print the installed Python and PyTorch environment details."""

import platform


def main() -> None:
    print(f"Python version: {platform.python_version()}")

    try:
        import torch
        import torchvision
    except ImportError as error:
        print(f"PyTorch/torchvision import error: {error}")
        raise SystemExit(1) from error

    print(f"PyTorch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version reported by PyTorch: {torch.version.cuda or 'None'}")

    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU name: None")


if __name__ == "__main__":
    main()