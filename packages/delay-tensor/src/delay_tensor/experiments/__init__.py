"""Experiment entrypoints."""

from delay_tensor.experiments.base import Summary, run_replicates
from delay_tensor.experiments.classify import run_classify
from delay_tensor.experiments.xor import run_xor

__all__ = ["Summary", "run_classify", "run_replicates", "run_xor"]
