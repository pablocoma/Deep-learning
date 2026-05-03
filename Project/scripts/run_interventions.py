"""Run concept intervention analysis on a trained CBM checkpoint."""

import argparse
import sys
from pathlib import Path

import pandas as pd


project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.concepts import CONCEPT_NAMES
from src.data import create_dataloaders
from src.interventions import analyze_concept_interventions, load_cbm_checkpoint
from src.plots import plot_concept_influence_ranking
from src.utils import ensure_dir, get_device, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure concept influence by flipping predicted binary concepts.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a trained Joint, Hybrid, or Independent CBM checkpoint.",
    )
    parser.add_argument(
        "--model",
        choices=["cbm_joint", "hybrid", "cbm_independent"],
        default=None,
        help="Optional model type override if the checkpoint does not store it.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for the test dataloader.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for dataloader setup.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used to binarize predicted concept probabilities.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional prefix for output files. Defaults to the checkpoint name.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    data_dir = project_root / "data"
    metrics_dir = project_root / "outputs" / "metrics"
    plots_dir = project_root / "outputs" / "plots"
    ensure_dir(metrics_dir)
    ensure_dir(plots_dir)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        if checkpoint_path.exists():
            checkpoint_path = checkpoint_path.resolve()
        else:
            checkpoint_path = project_root / checkpoint_path

    model, checkpoint, model_name = load_cbm_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        model_name=args.model,
    )

    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        batch_size=args.batch_size,
        data_dir=str(data_dir),
        seed=args.seed,
    )

    rows = analyze_concept_interventions(
        model=model,
        data_loader=test_loader,
        device=device,
        concept_names=CONCEPT_NAMES,
        threshold=args.threshold,
    )

    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = f"{checkpoint_path.stem}_interventions"

    csv_path = metrics_dir / f"{output_prefix}.csv"
    json_path = metrics_dir / f"{output_prefix}.json"
    plot_path = plots_dir / f"{output_prefix}.png"

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    notes = []
    if model_name == "hybrid":
        notes.append(
            "Hybrid CBM interventions modify only the concept path while keeping "
            "the side-channel logits fixed. These interventions are less clean "
            "because the side channel still contributes to the final prediction."
        )

    save_json(
        {
            "checkpoint_path": str(checkpoint_path),
            "model": model_name,
            "checkpoint_args": checkpoint.get("args", {}),
            "seed": args.seed,
            "batch_size": args.batch_size,
            "threshold": args.threshold,
            "class_names": class_names,
            "concept_names": CONCEPT_NAMES,
            "ranking_metric": "avg_abs_change_original_class_prob",
            "notes": notes,
            "results": rows,
        },
        json_path,
    )

    plot_concept_influence_ranking(
        rows=rows,
        output_path=plot_path,
        metric_name="avg_abs_change_original_class_prob",
        ylabel="Avg abs change in originally predicted class probability",
    )

    print(f"Device: {device}")
    print(f"Loaded model: {model_name}")
    if notes:
        print(notes[0])
    print(f"Metrics CSV saved to: {csv_path}")
    print(f"Metrics JSON saved to: {json_path}")
    print(f"Plot saved to: {plot_path}")
    print("\nConcept influence ranking:")
    for row in rows:
        print(
            "{rank}. {name}: original_prob_delta={delta:.4f}, "
            "max_prob_delta={max_delta:.4f}, label_changes={label_changes:.2f}%".format(
                rank=row["influence_rank"],
                name=row["concept_name"],
                delta=row["avg_abs_change_original_class_prob"],
                max_delta=row["avg_abs_change_max_class_prob"],
                label_changes=row["label_change_percent"],
            )
        )


if __name__ == "__main__":
    main()
