"""Simple PyTorch training loops for the project models."""

import copy

import torch
from torch import nn

from src.utils import get_device


def _class_logits_from_output(output):
    if isinstance(output, dict):
        return output["class_logits"]
    return output


def _concept_logits_from_output(output):
    if isinstance(output, dict):
        return output["concept_logits"]
    return output


def _remember_best_model(model):
    return copy.deepcopy(model.state_dict())


def _restore_best_model(model, best_state):
    if best_state is not None:
        model.load_state_dict(best_state)


def train_classifier(
    model,
    train_loader,
    val_loader=None,
    epochs=10,
    learning_rate=0.001,
    optimizer=None,
    criterion=None,
    device=None,
    patience=None,
):
    """Train a standard image classifier: x -> y."""
    if device is None:
        device = get_device()

    model.to(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for x, y, concepts in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x)
            class_logits = _class_logits_from_output(output)
            loss = criterion(class_logits, y)
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        train_loss = total_loss / total_examples
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            model.eval()
            val_total_loss = 0.0
            val_total_examples = 0

            with torch.no_grad():
                for x, y, concepts in val_loader:
                    x = x.to(device)
                    y = y.to(device)

                    output = model(x)
                    class_logits = _class_logits_from_output(output)
                    loss = criterion(class_logits, y)

                    batch_size = x.size(0)
                    val_total_loss += loss.item() * batch_size
                    val_total_examples += batch_size

            val_loss = val_total_loss / val_total_examples
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = _remember_best_model(model)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if patience is not None and epochs_without_improvement >= patience:
                break

    _restore_best_model(model, best_state)
    if best_state is None:
        history["best_val_loss"] = None
    else:
        history["best_val_loss"] = best_val_loss
    history["epochs_trained"] = len(history["train_loss"])

    return history


def train_concept_predictor(
    model,
    train_loader,
    val_loader=None,
    epochs=10,
    learning_rate=0.001,
    optimizer=None,
    criterion=None,
    device=None,
    patience=None,
):
    """Train a concept predictor: x -> c."""
    if device is None:
        device = get_device()

    model.to(device)

    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for x, y, concepts in train_loader:
            x = x.to(device)
            concepts = concepts.to(device).float()

            optimizer.zero_grad()
            output = model(x)
            concept_logits = _concept_logits_from_output(output)
            loss = criterion(concept_logits, concepts)
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        train_loss = total_loss / total_examples
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            model.eval()
            val_total_loss = 0.0
            val_total_examples = 0

            with torch.no_grad():
                for x, y, concepts in val_loader:
                    x = x.to(device)
                    concepts = concepts.to(device).float()

                    output = model(x)
                    concept_logits = _concept_logits_from_output(output)
                    loss = criterion(concept_logits, concepts)

                    batch_size = x.size(0)
                    val_total_loss += loss.item() * batch_size
                    val_total_examples += batch_size

            val_loss = val_total_loss / val_total_examples
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = _remember_best_model(model)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if patience is not None and epochs_without_improvement >= patience:
                break

    _restore_best_model(model, best_state)
    if best_state is None:
        history["best_val_loss"] = None
    else:
        history["best_val_loss"] = best_val_loss
    history["epochs_trained"] = len(history["train_loss"])

    return history


def train_concept_label_head(
    concept_predictor,
    label_head,
    train_loader,
    val_loader=None,
    epochs=10,
    learning_rate=0.001,
    optimizer=None,
    criterion=None,
    device=None,
    patience=None,
):
    """Train a label head from predicted concepts: x -> c -> y."""
    if device is None:
        device = get_device()

    concept_predictor.to(device)
    concept_predictor.eval()

    for parameter in concept_predictor.parameters():
        parameter.requires_grad = False

    label_head.to(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if optimizer is None:
        optimizer = torch.optim.Adam(label_head.parameters(), lr=learning_rate)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        label_head.train()
        total_loss = 0.0
        total_examples = 0

        for x, y, concepts in train_loader:
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                concept_output = concept_predictor(x)
                concept_logits = _concept_logits_from_output(concept_output)
                predicted_concepts = torch.sigmoid(concept_logits)

            optimizer.zero_grad()
            class_logits = label_head(predicted_concepts)
            loss = criterion(class_logits, y)
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        train_loss = total_loss / total_examples
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            label_head.eval()
            val_total_loss = 0.0
            val_total_examples = 0

            with torch.no_grad():
                for x, y, concepts in val_loader:
                    x = x.to(device)
                    y = y.to(device)

                    concept_output = concept_predictor(x)
                    concept_logits = _concept_logits_from_output(concept_output)
                    predicted_concepts = torch.sigmoid(concept_logits)

                    class_logits = label_head(predicted_concepts)
                    loss = criterion(class_logits, y)

                    batch_size = x.size(0)
                    val_total_loss += loss.item() * batch_size
                    val_total_examples += batch_size

            val_loss = val_total_loss / val_total_examples
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = _remember_best_model(label_head)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if patience is not None and epochs_without_improvement >= patience:
                break

    _restore_best_model(label_head, best_state)
    if best_state is None:
        history["best_val_loss"] = None
    else:
        history["best_val_loss"] = best_val_loss
    history["epochs_trained"] = len(history["train_loss"])

    return history


def train_joint_cbm(
    model,
    train_loader,
    val_loader=None,
    epochs=10,
    learning_rate=0.001,
    optimizer=None,
    classification_criterion=None,
    concept_criterion=None,
    lambda_concept=1.0,
    device=None,
    patience=None,
):
    """Train a joint CBM with classification loss and concept loss."""
    if device is None:
        device = get_device()

    model.to(device)

    if classification_criterion is None:
        classification_criterion = nn.CrossEntropyLoss()

    if concept_criterion is None:
        concept_criterion = nn.BCEWithLogitsLoss()

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "train_classification_loss": [],
        "train_concept_loss": [],
        "val_loss": [],
        "val_classification_loss": [],
        "val_concept_loss": [],
    }
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_classification_loss = 0.0
        total_concept_loss = 0.0
        total_examples = 0

        for x, y, concepts in train_loader:
            x = x.to(device)
            y = y.to(device)
            concepts = concepts.to(device).float()

            optimizer.zero_grad()
            output = model(x)
            class_logits = output["class_logits"]
            concept_logits = output["concept_logits"]

            classification_loss = classification_criterion(class_logits, y)
            concept_loss = concept_criterion(concept_logits, concepts)
            loss = classification_loss + lambda_concept * concept_loss

            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_classification_loss += classification_loss.item() * batch_size
            total_concept_loss += concept_loss.item() * batch_size
            total_examples += batch_size

        history["train_loss"].append(total_loss / total_examples)
        history["train_classification_loss"].append(
            total_classification_loss / total_examples
        )
        history["train_concept_loss"].append(total_concept_loss / total_examples)

        if val_loader is not None:
            model.eval()
            val_total_loss = 0.0
            val_total_classification_loss = 0.0
            val_total_concept_loss = 0.0
            val_total_examples = 0

            with torch.no_grad():
                for x, y, concepts in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    concepts = concepts.to(device).float()

                    output = model(x)
                    class_logits = output["class_logits"]
                    concept_logits = output["concept_logits"]

                    classification_loss = classification_criterion(class_logits, y)
                    concept_loss = concept_criterion(concept_logits, concepts)
                    loss = classification_loss + lambda_concept * concept_loss

                    batch_size = x.size(0)
                    val_total_loss += loss.item() * batch_size
                    val_total_classification_loss += (
                        classification_loss.item() * batch_size
                    )
                    val_total_concept_loss += concept_loss.item() * batch_size
                    val_total_examples += batch_size

            val_loss = val_total_loss / val_total_examples
            history["val_loss"].append(val_loss)
            history["val_classification_loss"].append(
                val_total_classification_loss / val_total_examples
            )
            history["val_concept_loss"].append(
                val_total_concept_loss / val_total_examples
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = _remember_best_model(model)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if patience is not None and epochs_without_improvement >= patience:
                break

    _restore_best_model(model, best_state)
    if best_state is None:
        history["best_val_loss"] = None
    else:
        history["best_val_loss"] = best_val_loss
    history["epochs_trained"] = len(history["train_loss"])

    return history
