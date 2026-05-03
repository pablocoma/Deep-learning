"""Run a small optional hyperparameter search for Joint or Hybrid CBMs."""

import argparse
import itertools
import random
import sys
from pathlib import Path

import pandas as pd
from torch import nn


project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.concepts import CONCEPT_NAMES
from src.data import create_dataloaders
from src.evaluate import evaluate_joint_cbm
from src.models import ConceptPredictor, HybridCBM, JointCBM, NUM_CLASSES, NUM_CONCEPTS
from src.train import train_joint_cbm
from src.utils import ensure_dir, get_device, save_json, set_seed


LEARNING_RATES = [3e-4, 1e-3, 3e-3]
LAMBDA_CONCEPT_VALUES = [0.3, 1.0, 3.0]
DROPOUT_VALUES = [0.0, 0.3, 0.5]
HEAD_HIDDEN_DIMS = [16, 32, 64]


class TunableConceptLabelHead(nn.Module):
    """Local HPO-only concept-to-label head with a configurable hidden size."""

    def __init__(
        self,
        num_concepts=NUM_CONCEPTS,
        num_classes=NUM_CLASSES,
        hidden_dim=32,
    ):
        super().__init__()

        self.label_head = nn.Sequential(
            nn.Linear(num_concepts, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, concepts):
        return self.label_head(concepts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Small optional HPO for Joint or Hybrid Concept Bottleneck Models.",
    )
    parser.add_argument(
        "--model",
        choices=["cbm_joint", "hybrid"],
        default="hybrid",
        help="Model family to tune.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=6,
        help="Number of sampled configurations to evaluate.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Maximum epochs per trial.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for train, validation, and test loaders.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data split and trial sampling.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Early stopping patience per trial.",
    )
    parser.add_argument(
        "--objective",
        choices=["accuracy", "auroc_ovr", "macro_concept_f1"],
        default="accuracy",
        help="Validation metric used to choose the best trial.",
    )
    return parser.parse_args()


def make_model(model_name, params):
    if model_name == "cbm_joint":
        model = JointCBM()
        model.concept_predictor = ConceptPredictor(dropout=params["dropout"])
        model.label_head = TunableConceptLabelHead(hidden_dim=params["head_hidden_dim"])
        return model

    model = HybridCBM(side_dropout=params["dropout"])
    model.concept_label_head = TunableConceptLabelHead(
        hidden_dim=params["head_hidden_dim"]
    )
    return model


def make_candidate_space():
    keys = [
        "learning_rate",
        "lambda_concept",
        "dropout",
        "head_hidden_dim",
    ]
    values = [
        LEARNING_RATES,
        LAMBDA_CONCEPT_VALUES,
        DROPOUT_VALUES,
        HEAD_HIDDEN_DIMS,
    ]

    candidates = []
    for combination in itertools.product(*values):
        candidates.append(dict(zip(keys, combination)))

    return candidates


def sample_candidates(num_trials, seed):
    candidates = make_candidate_space()
    rng = random.Random(seed)
    rng.shuffle(candidates)

    if num_trials > len(candidates):
        num_trials = len(candidates)

    return candidates[:num_trials]


def train_one_trial(
    trial_number,
    params,
    train_loader,
    val_loader,
    args,
    device,
):
    set_seed(args.seed)
    model = make_model(args.model, params)

    history = train_joint_cbm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=params["learning_rate"],
        lambda_concept=params["lambda_concept"],
        device=device,
        patience=args.patience,
    )
    val_metrics = evaluate_joint_cbm(
        model=model,
        data_loader=val_loader,
        device=device,
    )

    row = {
        "trial": trial_number,
        "model": args.model,
        "objective": args.objective,
        "objective_value": val_metrics[args.objective],
        "learning_rate": params["learning_rate"],
        "lambda_concept": params["lambda_concept"],
        "dropout": params["dropout"],
        "head_hidden_dim": params["head_hidden_dim"],
        "epochs_trained": history["epochs_trained"],
        "best_val_loss": history["best_val_loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_auroc_ovr": val_metrics["auroc_ovr"],
        "val_macro_concept_accuracy": val_metrics["macro_concept_accuracy"],
        "val_macro_concept_f1": val_metrics["macro_concept_f1"],
    }

    return row


def make_run_name(args):
    return (
        f"hpo_{args.model}"
        f"_seed{args.seed}"
        f"_trials{args.trials}"
        f"_epochs{args.epochs}"
        f"_bs{args.batch_size}"
        f"_objective{args.objective}"
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    data_dir = project_root / "data"
    metrics_dir = project_root / "outputs" / "metrics"
    ensure_dir(metrics_dir)

    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        batch_size=args.batch_size,
        data_dir=str(data_dir),
        seed=args.seed,
    )
    del test_loader

    candidates = sample_candidates(args.trials, args.seed)
    run_name = make_run_name(args)
    trials_path = metrics_dir / f"{run_name}_trials.csv"
    best_path = metrics_dir / f"{run_name}_best.json"

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Trials: {len(candidates)}")
    print(f"Objective: validation {args.objective}")
    print(f"Concepts: {', '.join(CONCEPT_NAMES)}")
    print(f"Classes: {', '.join(class_names)}")

    rows = []
    for trial_number, params in enumerate(candidates, start=1):
        print(f"\nTrial {trial_number}/{len(candidates)}: {params}")
        row = train_one_trial(
            trial_number=trial_number,
            params=params,
            train_loader=train_loader,
            val_loader=val_loader,
            args=args,
            device=device,
        )
        rows.append(row)
        print(
            "val_accuracy={:.4f}, val_auroc={:.4f}, val_macro_concept_f1={:.4f}".format(
                row["val_accuracy"],
                row["val_auroc_ovr"],
                row["val_macro_concept_f1"],
            )
        )

    results = pd.DataFrame(rows)
    results = results.sort_values("objective_value", ascending=False)
    results.to_csv(trials_path, index=False)

    best_row = results.iloc[0].to_dict()
    best_params = {
        "learning_rate": float(best_row["learning_rate"]),
        "lambda_concept": float(best_row["lambda_concept"]),
        "dropout": float(best_row["dropout"]),
        "head_hidden_dim": int(best_row["head_hidden_dim"]),
    }
    best_data = {
        "run_name": run_name,
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "objective": args.objective,
        "best_trial": int(best_row["trial"]),
        "best_objective_value": float(best_row["objective_value"]),
        "best_params": best_params,
        "search_space": {
            "learning_rate": LEARNING_RATES,
            "lambda_concept": LAMBDA_CONCEPT_VALUES,
            "dropout": DROPOUT_VALUES,
            "head_hidden_dim": HEAD_HIDDEN_DIMS,
        },
        "trial_results_path": str(trials_path),
    }
    save_json(best_data, best_path)

    print(f"\nTrial results saved to: {trials_path}")
    print(f"Best parameters saved to: {best_path}")
    print(f"Best trial: {best_data['best_trial']}")
    print(f"Best validation {args.objective}: {best_data['best_objective_value']:.4f}")
    print(f"Best params: {best_params}")


if __name__ == "__main__":
    main()
