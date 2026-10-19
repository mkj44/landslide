"""Random seed management for reproducible runs across CPU, GPU, and MPS."""

import os
import random
from typing import Optional
import numpy as np
import torch


def seed_everything(seed: int = 42) -> int:
    """Sets deterministic seeds across python standard library, numpy, and PyTorch.

    Args:
        seed: Integer seed value.

    Returns:
        The applied seed integer.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Apple Silicon MPS deterministic operations if supported in torch version
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            # Some operations on MPS or certain PyTorch versions may warn or raise on strict deterministic mode
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass

    return seed
