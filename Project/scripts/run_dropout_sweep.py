"""Run a small Hybrid CBM side-channel dropout sweep."""

import argparse
import sys
from pathlib import Path

import pandas as pd


project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.concepts import CONCEPT_NAMES
from src.data import create_dataloaders
from src.evaluate import evaluate_joint_cbm
from src.models import HybridCBM
from src.plots import plot_dropout_metric
from src.train import train_joint_cbm
from src.utils import ensure_dir, get_device, set_seed


DROPOUT_VALUES = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
EARLY_STOPPING_PATIENCE = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Hybrid CBM models with different side-channel dropout values.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum number of training epochs for each dropout value.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for train, validation, and test.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--lambda-concept",
        type=float,
        default=1.0,
        help="Weight for the concept loss.",
    )
    parser.add_argument(
        "--dropouts",
        nargs="+",
        type=float,
        default=DROPOUT_VALUES,
        help="Side-channel dropout values to try.",
    )
    return parser.parse_args()


def train_one_dropout(side_dropout, train_loader, val_loader, test_loader, args, device):
    set_seed(args.seed)

    model = HybridCBM(side_dropout=side_dropout)

    history = train_joint_cbm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        lambda_concept=args.lambda_concept,
        device=device,
        patience=EARLY_STOPPING_PATIENCE,
    )

    test_metrics = evaluate_joint_cbm(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    row = {
        "side_dropout": side_dropout,
        "accuracy": test_metrics["accuracy"],
        "auroc_ovr": test_metrics["auroc_ovr"],
        "macro_concept_accuracy": test_metrics["macro_concept_accuracy"],
        "macro_concept_f1": test_metrics["macro_concept_f1"],
        "epochs_trained": history["epochs_trained"],
        "best_val_loss": history["best_val_loss"],
    }

    return row


def make_run_name(args):
    lr_text = str(args.lr).replace(".", "_")
    dropout_text = "_".join(str(value).replace(".", "_") for value in args.dropouts)

    return (
        f"hybrid_dropout_sweep_seed{args.seed}"
        f"_epochs{args.epochs}"
        f"_bs{args.batch_size}"
        f"_lr{lr_text}"
        f"_dropouts{dropout_text}"
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    data_dir = project_root / "data"
    metrics_dir = project_root / "outputs" / "metrics"
    plots_dir = project_root / "outputs" / "plots"

    ensure_dir(metrics_dir)
    ensure_dir(plots_dir)

    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        batch_size=args.batch_size,
        data_dir=str(data_dir),
        seed=args.seed,
    )

    rows = []
    print(f"Device: {device}")
    print(f"Concepts: {', '.join(CONCEPT_NAMES)}")
    print(f"Classes: {', '.join(class_names)}")

    for side_dropout in args.dropouts:
        print(f"\nTraining Hybrid CBM with side_dropout={side_dropout}")
        row = train_one_dropout(
            side_dropout=side_dropout,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            args=args,
            device=device,
        )
        rows.append(row)
        print(
            "accuracy={:.4f}, auroc={:.4f}, macro_concept_f1={:.4f}".format(
                row["accuracy"],
                row["auroc_ovr"],
                row["macro_concept_f1"],
            )
        )

    run_name = make_run_name(args)
    summary_path = metrics_dir / f"{run_name}.csv"

    summary = pd.DataFrame(rows)
    summary.to_csv(summary_path, index=False)

    plot_dropout_metric(
        rows=rows,
        metric_name="accuracy",
        ylabel="Accuracy",
        output_path=plots_dir / f"{run_name}_accuracy.png",
    )
    plot_dropout_metric(
        rows=rows,
        metric_name="auroc_ovr",
        ylabel="AUROC (OvR)",
        output_path=plots_dir / f"{run_name}_auroc.png",
    )
    plot_dropout_metric(
        rows=rows,
        metric_name="macro_concept_f1",
        ylabel="Macro concept F1",
        output_path=plots_dir / f"{run_name}_macro_concept_f1.png",
    )

    print(f"\nSummary saved to: {summary_path}")
    print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
