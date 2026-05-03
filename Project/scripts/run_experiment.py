"""Simple training runner for project experiments."""

import argparse
import sys
from pathlib import Path

import torch


project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.concepts import CONCEPT_NAMES
from src.data import create_dataloaders
from src.evaluate import (
    evaluate_classifier,
    evaluate_concept_predictor,
    evaluate_joint_cbm,
)
from src.models import (
    BaselineClassifier,
    ConceptLabelHead,
    ConceptPredictor,
    HybridCBM,
    JointCBM,
)
from src.train import (
    train_classifier,
    train_concept_label_head,
    train_concept_predictor,
    train_joint_cbm,
)
from src.utils import ensure_dir, get_device, save_json, set_seed


EARLY_STOPPING_PATIENCE = 3


class IndependentCBM(torch.nn.Module):
    """Independent CBM pipeline: x -> predicted concepts -> y."""

    def __init__(self, concept_predictor, label_head):
        super().__init__()
        self.concept_predictor = concept_predictor
        self.label_head = label_head

    def forward(self, x):
        concept_logits = self.concept_predictor(x)
        predicted_concepts = torch.sigmoid(concept_logits)
        class_logits = self.label_head(predicted_concepts)

        return {
            "concept_logits": concept_logits,
            "class_logits": class_logits,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one simple Fashion-MNIST experiment.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["baseline", "concept", "cbm_independent", "cbm_joint", "hybrid"],
        help="Which model to train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum number of training epochs.",
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
        help="Weight for the concept loss when training Joint or Hybrid CBM.",
    )
    parser.add_argument(
        "--side-dropout",
        type=float,
        default=0.3,
        help="Dropout probability for the Hybrid CBM side channel.",
    )
    return parser.parse_args()


def make_run_name(args):
    lr_text = str(args.lr).replace(".", "_")
    run_name = (
        f"{args.model}_seed{args.seed}"
        f"_epochs{args.epochs}"
        f"_bs{args.batch_size}"
        f"_lr{lr_text}"
    )
    if args.model in ["cbm_joint", "hybrid"]:
        lambda_text = str(args.lambda_concept).replace(".", "_")
        run_name = f"{run_name}_lambda{lambda_text}"
    if args.model == "hybrid":
        side_dropout_text = str(args.side_dropout).replace(".", "_")
        run_name = f"{run_name}_side{side_dropout_text}"

    return run_name


def save_checkpoint(model, checkpoint_path, args, history):
    torch.save(
        {
            "model_name": args.model,
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "history": history,
        },
        checkpoint_path,
    )


def save_independent_cbm_checkpoint(
    concept_predictor,
    label_head,
    checkpoint_path,
    args,
    history,
):
    torch.save(
        {
            "model_name": args.model,
            "concept_predictor_state_dict": concept_predictor.state_dict(),
            "label_head_state_dict": label_head.state_dict(),
            "args": vars(args),
            "history": history,
        },
        checkpoint_path,
    )


def train_baseline_model(train_loader, val_loader, test_loader, args, device):
    model = BaselineClassifier()

    history = train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        patience=EARLY_STOPPING_PATIENCE,
    )

    test_metrics = evaluate_classifier(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    return model, history, test_metrics


def train_concept_model(train_loader, val_loader, test_loader, args, device):
    model = ConceptPredictor()

    history = train_concept_predictor(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        patience=EARLY_STOPPING_PATIENCE,
    )

    test_metrics = evaluate_concept_predictor(
        model=model,
        data_loader=test_loader,
        device=device,
    )
    test_metrics["concept_names"] = CONCEPT_NAMES

    return model, history, test_metrics


def freeze_model(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False


def train_independent_cbm(train_loader, val_loader, test_loader, args, device):
    concept_predictor = ConceptPredictor()

    concept_history = train_concept_predictor(
        model=concept_predictor,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        patience=EARLY_STOPPING_PATIENCE,
    )

    freeze_model(concept_predictor)

    label_head = ConceptLabelHead()
    label_head_history = train_concept_label_head(
        concept_predictor=concept_predictor,
        label_head=label_head,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        patience=EARLY_STOPPING_PATIENCE,
    )

    pipeline_model = IndependentCBM(
        concept_predictor=concept_predictor,
        label_head=label_head,
    )
    test_metrics = evaluate_joint_cbm(
        model=pipeline_model,
        data_loader=test_loader,
        device=device,
    )
    test_metrics["concept_names"] = CONCEPT_NAMES

    history = {
        "concept_predictor": concept_history,
        "label_head": label_head_history,
    }

    return concept_predictor, label_head, history, test_metrics


def train_joint_cbm_model(train_loader, val_loader, test_loader, args, device):
    model = JointCBM()

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
    test_metrics["concept_names"] = CONCEPT_NAMES

    return model, history, test_metrics


def train_hybrid_cbm_model(train_loader, val_loader, test_loader, args, device):
    model = HybridCBM(side_dropout=args.side_dropout)

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
    test_metrics["concept_names"] = CONCEPT_NAMES

    return model, history, test_metrics


def main():
    args = parse_args()

    set_seed(args.seed)
    device = get_device()

    checkpoint_dir = project_root / "outputs" / "checkpoints"
    metrics_dir = project_root / "outputs" / "metrics"
    data_dir = project_root / "data"

    ensure_dir(checkpoint_dir)
    ensure_dir(metrics_dir)
    ensure_dir(data_dir)

    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        batch_size=args.batch_size,
        data_dir=str(data_dir),
        seed=args.seed,
    )

    run_name = make_run_name(args)
    checkpoint_path = checkpoint_dir / f"{run_name}.pt"
    metrics_path = metrics_dir / f"{run_name}.json"

    if args.model == "baseline":
        model, history, test_metrics = train_baseline_model(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            args=args,
            device=device,
        )
        save_checkpoint(model, checkpoint_path, args, history)
    elif args.model == "concept":
        model, history, test_metrics = train_concept_model(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            args=args,
            device=device,
        )
        save_checkpoint(model, checkpoint_path, args, history)
    elif args.model == "cbm_independent":
        concept_predictor, label_head, history, test_metrics = train_independent_cbm(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            args=args,
            device=device,
        )
        save_independent_cbm_checkpoint(
            concept_predictor=concept_predictor,
            label_head=label_head,
            checkpoint_path=checkpoint_path,
            args=args,
            history=history,
        )
    elif args.model == "cbm_joint":
        model, history, test_metrics = train_joint_cbm_model(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            args=args,
            device=device,
        )
        save_checkpoint(model, checkpoint_path, args, history)
    elif args.model == "hybrid":
        model, history, test_metrics = train_hybrid_cbm_model(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            args=args,
            device=device,
        )
        save_checkpoint(model, checkpoint_path, args, history)

    metrics_data = {
        "run_name": run_name,
        "model": args.model,
        "device": str(device),
        "args": vars(args),
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "class_names": class_names,
        "history": history,
        "test_metrics": test_metrics,
        "checkpoint_path": str(checkpoint_path),
    }
    save_json(metrics_data, metrics_path)

    print(f"Finished run: {run_name}")
    print(f"Device: {device}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Metrics saved to: {metrics_path}")

    if args.model == "baseline":
        print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test AUROC (OvR): {test_metrics['auroc_ovr']:.4f}")
    elif args.model == "concept":
        print(f"Macro concept accuracy: {test_metrics['macro_concept_accuracy']:.4f}")
        print(f"Macro concept F1: {test_metrics['macro_concept_f1']:.4f}")
    else:
        print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test AUROC (OvR): {test_metrics['auroc_ovr']:.4f}")
        print(f"Macro concept accuracy: {test_metrics['macro_concept_accuracy']:.4f}")
        print(f"Macro concept F1: {test_metrics['macro_concept_f1']:.4f}")


if __name__ == "__main__":
    main()
