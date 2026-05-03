"""Concept utilities for Fashion-MNIST."""

import torch


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

CONCEPT_NAMES = [
    "is_footwear",
    "is_closed_footwear",
    "is_footwear_or_bag",
    "has_sleeves",
    "has_collar",
    "is_long_garment",
    "is_outerwear_layer",
    "is_legwear_or_footwear",
]

CLASS_TO_CONCEPTS = {
    0: [0, 0, 0, 1, 0, 0, 0, 0],  # T-shirt/top
    1: [0, 0, 0, 0, 0, 0, 0, 1],  # Trouser
    2: [0, 0, 0, 1, 0, 0, 1, 0],  # Pullover
    3: [0, 0, 0, 1, 0, 1, 0, 0],  # Dress
    4: [0, 0, 0, 1, 1, 1, 1, 0],  # Coat
    5: [1, 0, 1, 0, 0, 0, 0, 1],  # Sandal
    6: [0, 0, 0, 1, 1, 0, 0, 0],  # Shirt
    7: [1, 1, 1, 0, 0, 0, 0, 1],  # Sneaker
    8: [0, 0, 1, 0, 0, 0, 0, 0],  # Bag
    9: [1, 1, 1, 0, 0, 0, 0, 1],  # Ankle boot
}


def label_to_concept_vector(label):
    """Convert one class label into an 8-value concept tensor."""
    concept_values = CLASS_TO_CONCEPTS[int(label)]
    return torch.tensor(concept_values, dtype=torch.float32)


def labels_to_concepts(labels):
    """Convert many class labels into a tensor of concept vectors."""
    concept_vectors = []

    for label in labels:
        concept_vectors.append(label_to_concept_vector(label))

    return torch.stack(concept_vectors)
