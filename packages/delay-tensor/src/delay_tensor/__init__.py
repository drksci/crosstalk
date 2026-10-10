"""Delay Tensor simulated experiments."""

__version__ = "0.2.0"

from delay_tensor.config import DelayTensorConfig, load_config
from delay_tensor.experiments import run_classify, run_xor
from delay_tensor.reservoir import DelayTensor

__all__ = [
    "DelayTensor",
    "DelayTensorConfig",
    "__version__",
    "load_config",
    "run_classify",
    "run_xor",
]
