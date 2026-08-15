from src.models.pdn import build_pdn_small, build_pdn_medium
from src.models.ae import build_autoencoder

def build_models(out_channels: int, model_size: str):

    if model_size == "small":
        teacher = build_pdn_small(out_channels)
        student = build_pdn_small(2 * out_channels)
    elif model_size == "medium":
        teacher = build_pdn_medium(out_channels)
        student = build_pdn_medium(2 * out_channels)
    else:
        raise ValueError(f"Unknown model size: {model_size}")

    autoencoder = build_autoencoder(out_channels)
    return teacher, student, autoencoder
