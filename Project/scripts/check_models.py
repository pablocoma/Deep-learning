"""Simple script to check model forward passes."""

import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import torch

from src.models import (
    BaselineClassifier,
    ConceptLabelHead,
    ConceptPredictor,
    HybridCBM,
    JointCBM,
)


def print_shape(name, tensor):
    print(f"  {name}: {tuple(tensor.shape)}")


def main():
    batch_size = 4

    x = torch.randn(batch_size, 1, 28, 28)
    # x: (batch, 1, 28, 28)

    c = torch.randn(batch_size, 8)
    # c: (batch, 8)

    baseline = BaselineClassifier()
    concept_predictor = ConceptPredictor()
    concept_label_head = ConceptLabelHead()
    joint_cbm = JointCBM()
    hybrid_cbm = HybridCBM(side_dropout=0.3)

    print("Dummy input shapes:")
    print_shape("x", x)
    print_shape("c", c)

    print("\nBaselineClassifier output:")
    y_logits = baseline(x)
    print_shape("class_logits", y_logits)

    print("\nConceptPredictor output:")
    concept_logits = concept_predictor(x)
    print_shape("concept_logits", concept_logits)

    print("\nConceptLabelHead output:")
    concept_label_logits = concept_label_head(c)
    print_shape("class_logits", concept_label_logits)

    print("\nJointCBM output:")
    joint_output = joint_cbm(x)
    print_shape("concept_logits", joint_output["concept_logits"])
    print_shape("class_logits", joint_output["class_logits"])

    print("\nHybridCBM output:")
    hybrid_output = hybrid_cbm(x)
    print_shape("concept_logits", hybrid_output["concept_logits"])
    print_shape("concept_class_logits", hybrid_output["concept_class_logits"])
    print_shape("side_class_logits", hybrid_output["side_class_logits"])
    print_shape("class_logits", hybrid_output["class_logits"])


if __name__ == "__main__":
    main()
