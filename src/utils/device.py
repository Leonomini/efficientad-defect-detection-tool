import torch

def resolve_device(device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")

    elif device == "cpu":
        return torch.device("cpu")

    else:
        raise ValueError(
            f"Invalid device '{device}'. "
            "Expected one of: auto | cuda | cpu"
    )
