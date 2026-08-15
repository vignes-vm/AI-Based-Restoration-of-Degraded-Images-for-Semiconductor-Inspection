"""Shared device-selection helper: honors a requested device but falls back
gracefully (cuda -> mps -> cpu) since this codebase runs on CUDA GPUs in
production but is also developed/tested on Apple Silicon (MPS) and CPU-only
machines.
"""
import torch


def resolve_device(requested="cuda"):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            print("[device] CUDA not available, falling back to MPS.")
            return torch.device("mps")
        print("[device] CUDA/MPS not available, falling back to CPU.")
        return torch.device("cpu")
    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[device] MPS not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)
