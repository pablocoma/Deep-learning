"""Evaluation helpers for classification and concept models."""

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

from src.utils import get_device


def classification_accuracy(y_true, y_pred):
    """Compute normal classification accuracy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def multiclass_ovr_auroc(y_true, class_probabilities):
    """Compute multiclass one-vs-rest AUROC."""
    y_true = np.asarray(y_true)
    class_probabilities = np.asarray(class_probabilities)

    try:
        return float(
            roc_auc_score(
                y_true,
                class_probabilities,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        return float("nan")


def concept_accuracy_per_concept(concept_true, concept_pred):
    """Compute accuracy separately for each concept."""
    concept_true = np.asarray(concept_true)
    concept_pred = np.asarray(concept_pred)
    return (concept_true == concept_pred).mean(axis=0).tolist()


def concept_f1_per_concept(concept_true, concept_pred):
    """Compute F1 separately for each concept."""
    concept_true = np.asarray(concept_true)
    concept_pred = np.asarray(concept_pred)

    scores = []
    for concept_index in range(concept_true.shape[1]):
        score = f1_score(
            concept_true[:, concept_index],
            concept_pred[:, concept_index],
            zero_division=0,
        )
        scores.append(float(score))

    return scores


def macro_concept_accuracy(concept_true, concept_pred):
    """Average the per-concept accuracies."""
    scores = concept_accuracy_per_concept(concept_true, concept_pred)
    return float(np.mean(scores))


def macro_concept_f1(concept_true, concept_pred):
    """Average the per-concept F1 scores."""
    scores = concept_f1_per_concept(concept_true, concept_pred)
    return float(np.mean(scores))


def _class_logits_from_output(output):
    if isinstance(output, dict):
        return output["class_logits"]
    return output


def _concept_logits_from_output(output):
    if isinstance(output, dict):
        return output["concept_logits"]
    return output


def _classification_metrics_from_logits(y_true, class_logits):
    class_probabilities = torch.softmax(class_logits, dim=1).cpu().numpy()
    y_pred = class_probabilities.argmax(axis=1)

    return {
        "accuracy": classification_accuracy(y_true, y_pred),
        "auroc_ovr": multiclass_ovr_auroc(y_true, class_probabilities),
    }


def _concept_metrics_from_logits(concept_true, concept_logits, threshold=0.5):
    concept_probabilities = torch.sigmoid(concept_logits).cpu().numpy()
    concept_pred = (concept_probabilities >= threshold).astype(int)

    per_concept_accuracy = concept_accuracy_per_concept(concept_true, concept_pred)
    per_concept_f1 = concept_f1_per_concept(concept_true, concept_pred)

    return {
        "concept_accuracy_per_concept": per_concept_accuracy,
        "concept_f1_per_concept": per_concept_f1,
        "macro_concept_accuracy": float(np.mean(per_concept_accuracy)),
        "macro_concept_f1": float(np.mean(per_concept_f1)),
    }


def evaluate_classifier(model, data_loader, device=None):
    """Evaluate a model that predicts class logits."""
    if device is None:
        device = get_device()

    model.to(device)
    model.eval()

    all_y = []
    all_class_logits = []

    with torch.no_grad():
        for x, y, concepts in data_loader:
            x = x.to(device)
            output = model(x)
            class_logits = _class_logits_from_output(output)

            all_y.append(y.cpu())
            all_class_logits.append(class_logits.cpu())

    y_true = torch.cat(all_y).numpy()
    class_logits = torch.cat(all_class_logits)

    return _classification_metrics_from_logits(y_true, class_logits)


def evaluate_concept_predictor(model, data_loader, device=None, threshold=0.5):
    """Evaluate a model that predicts concept logits."""
    if device is None:
        device = get_device()

    model.to(device)
    model.eval()

    all_concepts = []
    all_concept_logits = []

    with torch.no_grad():
        for x, y, concepts in data_loader:
            x = x.to(device)
            output = model(x)
            concept_logits = _concept_logits_from_output(output)

            all_concepts.append(concepts.cpu())
            all_concept_logits.append(concept_logits.cpu())

    concept_true = torch.cat(all_concepts).numpy()
    concept_logits = torch.cat(all_concept_logits)

    return _concept_metrics_from_logits(concept_true, concept_logits, threshold)


def evaluate_joint_cbm(model, data_loader, device=None, threshold=0.5):
    """Evaluate a joint CBM with both class and concept outputs."""
    if device is None:
        device = get_device()

    model.to(device)
    model.eval()

    all_y = []
    all_concepts = []
    all_class_logits = []
    all_concept_logits = []

    with torch.no_grad():
        for x, y, concepts in data_loader:
            x = x.to(device)
            output = model(x)

            all_y.append(y.cpu())
            all_concepts.append(concepts.cpu())
            all_class_logits.append(output["class_logits"].cpu())
            all_concept_logits.append(output["concept_logits"].cpu())

    y_true = torch.cat(all_y).numpy()
    concept_true = torch.cat(all_concepts).numpy()
    class_logits = torch.cat(all_class_logits)
    concept_logits = torch.cat(all_concept_logits)

    metrics = {}
    metrics.update(_classification_metrics_from_logits(y_true, class_logits))
    metrics.update(
        _concept_metrics_from_logits(concept_true, concept_logits, threshold)
    )

    return metrics
