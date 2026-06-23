"""
CLI entry point — train and evaluate selected models.

Usage:
    uv run python -m src.models.run_all --models linear_regression svr
    uv run python -m src.models.run_all --models linear_regression svr --benchmark
    uv run python -m src.models.run_all --benchmark   # benchmark only (no training)
"""

import argparse
import sys

# Import all model modules so @register_model decorators fire
from . import linear_regression  # noqa: F401
from . import svr as svr_module  # noqa: F401
from . import random_forest  # noqa: F401
from . import xgboost_model  # noqa: F401

from .config import MODEL_REGISTRY
from .benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="EV Price Prediction — Model Training & Evaluation"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        default=[],
        help="Models to train and evaluate.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark comparison after training.",
    )
    parser.add_argument(
        "--use-weights",
        action="store_true",
        help="Enable sample weights during training (off by default).",
    )

    args = parser.parse_args()

    if not args.models and not args.benchmark:
        parser.print_help()
        print("\nAvailable models:", list(MODEL_REGISTRY.keys()))
        sys.exit(0)

    # Train requested models
    trained = []
    for name in args.models:
        cls = MODEL_REGISTRY[name]
        model = cls()
        model.run(use_sample_weights=args.use_weights)
        trained.append(name)

    # Benchmark
    if args.benchmark:
        print("\n" + "=" * 60)
        print("  Running benchmark comparison")
        print("=" * 60)
        run_benchmark()

    if trained:
        print(f"\nDone. Trained models: {trained}")
        print("Results saved to: reports/model_benchmark/")


if __name__ == "__main__":
    main()
