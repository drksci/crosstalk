"""Command line interface: ``delay-tensor <experiment> --config ...``."""

from __future__ import annotations

import argparse
import shlex
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from delay_tensor.config import ConfigError, DelayTensorConfig, load_config
from delay_tensor.experiments import Summary, run_classify, run_mnist, run_xor
from delay_tensor.experiments.base import DEFAULT_SEEDS, DEFAULT_TEST_FRACTION
from delay_tensor.experiments.mnist import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_MLP_MAX_ITER,
    DEFAULT_RESERVOIR_BATCH,
)
from delay_tensor.report import RunPaths, write_run

DEFAULT_SEEDS_TEXT = ",".join(str(seed) for seed in DEFAULT_SEEDS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="delay-tensor",
        description="Run the Delay Tensor simulated experiments.",
    )
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    classify = subparsers.add_parser("classify", help="six-class synthetic waveform classification")
    _add_common_arguments(classify)
    classify.add_argument(
        "--samples-per-class", type=int, default=64, help="waveforms generated per class"
    )
    classify.add_argument("--noise-std", type=float, default=0.03, help="input noise std")

    xor = subparsers.add_parser("xor", help="temporal XOR: two pulses, label = their XOR")
    _add_common_arguments(xor)
    xor.add_argument("--samples", type=int, default=256, help="total waveforms generated")
    xor.add_argument("--noise-std", type=float, default=0.02, help="input noise std")

    # MNIST gets its own flag set rather than _add_common_arguments: its shape
    # (784 slots) and its split (the canonical 60000/10) are fixed by the
    # dataset, so --time-samples and --test-fraction would be flags that take
    # a value and do nothing.
    mnist = subparsers.add_parser(
        "mnist", help="full MNIST: raw pixels vs tensor features vs a small MLP"
    )
    mnist.add_argument("--config", required=True, help="path to a tensor YAML config")
    mnist.add_argument(
        "--seeds",
        default=DEFAULT_SEEDS_TEXT,
        help=f"comma-separated replicate seeds (default: {DEFAULT_SEEDS_TEXT})",
    )
    mnist.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="SGD epochs, one budget shared by both linear arms",
    )
    mnist.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="readout batch size, shared by both linear arms",
    )
    mnist.add_argument(
        "--reservoir-batch",
        type=int,
        default=DEFAULT_RESERVOIR_BATCH,
        help="waveforms pushed through the tensor at a time",
    )
    mnist.add_argument(
        "--mlp-max-iter",
        type=int,
        default=DEFAULT_MLP_MAX_ITER,
        help="MLP training epochs (early stopping may stop sooner)",
    )
    mnist.add_argument("--out", default="results", help="directory that receives the run folder")
    mnist.add_argument("--quiet", action="store_true", help="print only the run directory")

    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="path to a tensor YAML config")
    parser.add_argument(
        "--seeds",
        default=DEFAULT_SEEDS_TEXT,
        help=f"comma-separated replicate seeds (default: {DEFAULT_SEEDS_TEXT})",
    )
    parser.add_argument("--time-samples", type=int, default=64, help="samples per waveform")
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help="held-out fraction, taken per class",
    )
    parser.add_argument("--out", default="results", help="directory that receives the run folder")
    parser.add_argument("--quiet", action="store_true", help="print only the run directory")


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(args_list)

    try:
        seeds = _parse_seeds(args.seeds)
        config = load_config(args.config)
        summary = _dispatch(args, config, seeds)
    except (ConfigError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    command = " ".join(["delay-tensor", *args_list])
    paths = write_run(_run_dir(args.out, summary), summary, command=command)
    _report(summary, paths, quiet=args.quiet)
    return 0


def _dispatch(args: argparse.Namespace, config: DelayTensorConfig, seeds: list[int]) -> Summary:
    if args.experiment == "mnist":
        # Checked first: the mnist parser has no time_samples/test_fraction
        # attributes, so reading them in a fall-through branch would raise
        # AttributeError instead of running.
        return run_mnist(
            config,
            seeds=seeds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            reservoir_batch=args.reservoir_batch,
            mlp_max_iter=args.mlp_max_iter,
        )
    if args.experiment == "classify":
        return run_classify(
            config,
            seeds=seeds,
            samples_per_class=args.samples_per_class,
            time_samples=args.time_samples,
            noise_std=args.noise_std,
            test_fraction=args.test_fraction,
        )
    return run_xor(
        config,
        seeds=seeds,
        samples=args.samples,
        time_samples=args.time_samples,
        noise_std=args.noise_std,
        test_fraction=args.test_fraction,
    )


def _parse_seeds(text: str) -> list[int]:
    try:
        seeds = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"--seeds must be comma-separated integers, got {text!r}") from exc
    if not seeds:
        raise ValueError("--seeds must name at least one seed")
    return seeds


def _run_dir(out_root: str, summary: Summary) -> Path:
    stamp = _timestamp()
    return Path(out_root) / f"{stamp}-{summary.config.name}-{summary.experiment}"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _report(summary: Summary, paths: RunPaths, *, quiet: bool) -> None:
    if quiet:
        print(paths.directory)
        return
    delta = summary.delta
    print(f"experiment   : {summary.experiment} ({summary.raw.n} seeds, {summary.config.name})")
    print(f"chance level : {summary.chance_level:.3f}")
    print(f"raw baseline : {summary.raw.format()}")
    print(f"reservoir    : {summary.reservoir.format()}")
    if summary.mlp is not None:
        print(f"mlp (ref)    : {summary.mlp.format()}")
    print(f"delta        : {delta.mean:+.3f} ± {delta.std:.3f}")
    print(f"features     : {summary.feature_count} virtual nodes")
    print(f"report       : {paths.report}")
    print(f"observed     : {shlex.quote(str(paths.observations))}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
