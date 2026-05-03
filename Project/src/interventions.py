"""Concept intervention analysis for CBM models."""

from pathlib import Path

import torch

from src.concepts import CONCEPT_NAMES
from src.models import ConceptLabelHead, ConceptPredictor, HybridCBM, JointCBM
from src.utils import get_device


class IndependentCBM(torch.nn.Module):
    """Independent CBM wrapper used when loading two-stage checkpoints."""

    def __init__(self, concept_predictor, label_head):
        super().__init__()
        self.concept_predictor = concept_predictor
        self.label_head = label_head

    def forward(self, x):
        concept_logits = self.concept_predictor(x)
        concepts_for_label = torch.sigmoid(concept_logits)
        class_logits = self.label_head(concepts_for_label)

        return {
            "concept_logits": concept_logits,
            "class_logits": class_logits,
        }


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_cbm_checkpoint(checkpoint_path, device=None, model_name=None):
    """Load a trained Joint, Hybrid, or Independent CBM checkpoint."""
    if device is None:
        device = get_device()

    checkpoint_path = Path(checkpoint_path)
    checkpoint = _torch_load(checkpoint_path, device)
    checkpoint_model_name = checkpoint.get("model_name")

    if model_name is None:
        model_name = checkpoint_model_name

    if model_name == "cbm_joint":
        model = JointCBM()
        model.load_state_dict(checkpoint["model_state_dict"])
    elif model_name == "hybrid":
        args = checkpoint.get("args", {})
        model = HybridCBM(side_dropout=args.get("side_dropout", 0.3))
        model.load_state_dict(checkpoint["model_state_dict"])
    elif model_name == "cbm_independent":
        concept_predictor = ConceptPredictor()
        label_head = ConceptLabelHead()
        concept_predictor.load_state_dict(checkpoint["concept_predictor_state_dict"])
        label_head.load_state_dict(checkpoint["label_head_state_dict"])
        model = IndependentCBM(concept_predictor, label_head)
    else:
        raise ValueError(
            "Interventions require a CBM checkpoint. "
            "Supported model names are: cbm_joint, hybrid, cbm_independent."
        )

    model.to(device)
    model.eval()

    return model, checkpoint, model_name


def _class_logits_from_intervened_concepts(model, concepts, side_channel_logits=None):
    if isinstance(model, HybridCBM):
        concept_path_logits = model.concept_label_head(concepts)
        return concept_path_logits + side_channel_logits

    return model.label_head(concepts)


def analyze_concept_interventions(
    model,
    data_loader,
    device=None,
    concept_names=None,
    threshold=0.5,
):
    """Flip each predicted binary concept and measure prediction changes."""
    if device is None:
        device = get_device()
    if concept_names is None:
        concept_names = CONCEPT_NAMES

    model.to(device)
    model.eval()

    num_concepts = len(concept_names)
    change_original_class = torch.zeros(num_concepts, device=device)
    change_max_class = torch.zeros(num_concepts, device=device)
    label_changes = torch.zeros(num_concepts, device=device)
    total_examples = 0

    with torch.no_grad():
        for x, y, concepts in data_loader:
            x = x.to(device)
            output = model(x)

            concept_probabilities = torch.sigmoid(output["concept_logits"])
            binary_concepts = (concept_probabilities >= threshold).float()

            side_channel_logits = None
            if isinstance(model, HybridCBM):
                side_channel_logits = output["side_channel_logits"]

            base_logits = _class_logits_from_intervened_concepts(
                model=model,
                concepts=binary_concepts,
                side_channel_logits=side_channel_logits,
            )
            base_probabilities = torch.softmax(base_logits, dim=1)
            base_labels = base_probabilities.argmax(dim=1)
            base_original_probabilities = base_probabilities.gather(
                1,
                base_labels.unsqueeze(1),
            ).squeeze(1)
            base_max_probabilities = base_probabilities.max(dim=1).values

            batch_size = x.size(0)
            total_examples += batch_size

            for concept_index in range(num_concepts):
                intervened_concepts = binary_concepts.clone()
                intervened_concepts[:, concept_index] = (
                    1.0 - intervened_concepts[:, concept_index]
                )

                intervened_logits = _class_logits_from_intervened_concepts(
                    model=model,
                    concepts=intervened_concepts,
                    side_channel_logits=side_channel_logits,
                )
                intervened_probabilities = torch.softmax(intervened_logits, dim=1)
                intervened_labels = intervened_probabilities.argmax(dim=1)
                intervened_original_probabilities = intervened_probabilities.gather(
                    1,
                    base_labels.unsqueeze(1),
                ).squeeze(1)
                intervened_max_probabilities = intervened_probabilities.max(dim=1).values

                change_original_class[concept_index] += torch.abs(
                    intervened_original_probabilities - base_original_probabilities
                ).sum()
                change_max_class[concept_index] += torch.abs(
                    intervened_max_probabilities - base_max_probabilities
                ).sum()
                label_changes[concept_index] += (
                    intervened_labels != base_labels
                ).float().sum()

    rows = []
    for concept_index, concept_name in enumerate(concept_names):
        rows.append(
            {
                "concept_index": concept_index,
                "concept_name": concept_name,
                "avg_abs_change_original_class_prob": float(
                    change_original_class[concept_index].item() / total_examples
                ),
                "avg_abs_change_max_class_prob": float(
                    change_max_class[concept_index].item() / total_examples
                ),
                "label_change_percent": float(
                    100.0 * label_changes[concept_index].item() / total_examples
                ),
            }
        )

    rows = sorted(
        rows,
        key=lambda row: row["avg_abs_change_original_class_prob"],
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["influence_rank"] = rank

    return rows
