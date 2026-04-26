"""Simple PyTorch models for Fashion-MNIST CBM experiments."""

import torch
from torch import nn


NUM_CLASSES = 10
NUM_CONCEPTS = 8


class SmallCNNBackbone(nn.Module):
    """Small CNN that turns a 1x28x28 image into a feature vector."""

    def __init__(self, use_batchnorm=True):
        super().__init__()

        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm2d(32)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()

        self.output_size = 32 * 7 * 7

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        # x: (batch, 16, 14, 14)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        # x: (batch, 32, 7, 7)

        x = self.flatten(x)
        # x: (batch, 32 * 7 * 7)
        return x


class BaselineClassifier(nn.Module):
    """Baseline model: image x -> class logits y."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.backbone = SmallCNNBackbone()
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.output_size, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        features = self.backbone(x)
        # features: (batch, 32 * 7 * 7)
        class_logits = self.classifier(features)
        # class_logits: (batch, 10)
        return class_logits


class ConceptPredictor(nn.Module):
    """Concept model: image x -> concept logits c."""

    def __init__(self, num_concepts=NUM_CONCEPTS):
        super().__init__()

        self.backbone = SmallCNNBackbone()
        self.concept_head = nn.Sequential(
            nn.Linear(self.backbone.output_size, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_concepts),
        )

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        features = self.backbone(x)
        # features: (batch, 32 * 7 * 7)
        concept_logits = self.concept_head(features)
        # concept_logits: (batch, 8)
        return concept_logits


class ConceptLabelHead(nn.Module):
    """Label head: concept vector c -> class logits y."""

    def __init__(self, num_concepts=NUM_CONCEPTS, num_classes=NUM_CLASSES):
        super().__init__()

        self.label_head = nn.Sequential(
            nn.Linear(num_concepts, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, c):
        # c: (batch, 8)
        class_logits = self.label_head(c)
        # class_logits: (batch, 10)
        return class_logits


class JointCBM(nn.Module):
    """Joint CBM: image x -> concept logits c -> class logits y."""

    def __init__(self, num_concepts=NUM_CONCEPTS, num_classes=NUM_CLASSES):
        super().__init__()

        self.concept_predictor = ConceptPredictor(num_concepts=num_concepts)
        self.label_head = ConceptLabelHead(
            num_concepts=num_concepts,
            num_classes=num_classes,
        )

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        concept_logits = self.concept_predictor(x)
        # concept_logits: (batch, 8)

        concepts_for_label = torch.sigmoid(concept_logits)
        # concepts_for_label: (batch, 8), values between 0 and 1

        class_logits = self.label_head(concepts_for_label)
        # class_logits: (batch, 10)

        return {
            "concept_logits": concept_logits,
            "class_logits": class_logits,
        }


class HybridCBM(nn.Module):
    """Hybrid CBM: final logits = concept path f(c) + side channel s(x)."""

    def __init__(
        self,
        num_concepts=NUM_CONCEPTS,
        num_classes=NUM_CLASSES,
        side_dropout=0.3,
    ):
        super().__init__()

        self.concept_predictor = ConceptPredictor(num_concepts=num_concepts)
        self.concept_label_head = ConceptLabelHead(
            num_concepts=num_concepts,
            num_classes=num_classes,
        )

        self.side_backbone = SmallCNNBackbone()
        self.side_dropout = nn.Dropout(p=side_dropout)
        self.side_head = nn.Linear(self.side_backbone.output_size, num_classes)

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        concept_logits = self.concept_predictor(x)
        # concept_logits: (batch, 8)

        concepts_for_label = torch.sigmoid(concept_logits)
        # concepts_for_label: (batch, 8)

        concept_class_logits = self.concept_label_head(concepts_for_label)
        # concept_class_logits: (batch, 10)

        side_features = self.side_backbone(x)
        # side_features: (batch, 32 * 7 * 7)

        side_features = self.side_dropout(side_features)
        side_class_logits = self.side_head(side_features)
        # side_class_logits: (batch, 10)

        class_logits = concept_class_logits + side_class_logits
        # class_logits: (batch, 10)

        return {
            "concept_logits": concept_logits,
            "concept_class_logits": concept_class_logits,
            "side_class_logits": side_class_logits,
            "class_logits": class_logits,
        }
