import logging

logger = logging.getLogger(__name__)

def resolve_device(requested: str) -> str:
    requested = (requested or "cpu").strip().lower()
    if requested == "cpu":
        return "cpu"
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("Device %s requested but CUDA is unavailable; falling back to CPU", requested)
        return "cpu"
    return requested
