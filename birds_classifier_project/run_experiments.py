import argparse
import itertools
from pathlib import Path

import pandas as pd

from src.constants import CLASS_PAIRS, FEATURES
from src.pipeline import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all feature/class combinations.")
    parser.add_argument("--csv", required=True, help="Path to birds.csv")
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=["perceptron", "adaline"],
        help="Algorithm to use.",
    )
    parser.add_argument("--eta", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument(
        "--mse-threshold",
        type=float,
        default=0.05,
        help="MSE threshold for Adaline.",
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        help="Include bias in the model.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV filename. If omitted, a default name is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = []
    for feature_pair in itertools.combinations(FEATURES, 2):
        for class_pair in CLASS_PAIRS:
            result = run_experiment(
                csv_path=args.csv,
                selected_features=list(feature_pair),
                class_pair=class_pair,
                algorithm=args.algorithm,
                eta=args.eta,
                epochs=args.epochs,
                mse_threshold=args.mse_threshold,
                use_bias=args.bias,
                seed=args.seed,
            )

            rows.append(
                {
                    "algorithm": args.algorithm,
                    "feature_1": feature_pair[0],
                    "feature_2": feature_pair[1],
                    "class_1": class_pair[0],
                    "class_2": class_pair[1],
                    "accuracy": round(result.accuracy, 4),
                    "epochs_completed": result.epochs_completed,
                    "decision_equation": result.decision_equation,
                    "confusion_matrix": (
                        f"[[{result.confusion_matrix[0][0]}, {result.confusion_matrix[0][1]}], "
                        f"[{result.confusion_matrix[1][0]}, {result.confusion_matrix[1][1]}]]"
                    ),
                }
            )

    df = pd.DataFrame(rows).sort_values(
        by=["accuracy", "algorithm", "class_1", "class_2"],
        ascending=[False, True, True, True],
    )

    output_name = (
        args.output
        if args.output
        else f"experiment_summary_{args.algorithm}.csv"
    )
    output_path = Path(output_name)
    df.to_csv(output_path, index=False)

    print("\nTop 10 results:\n")
    print(df.head(10).to_string(index=False))
    print(f"\nSaved summary to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
