"""Simple script to check the Fashion-MNIST data."""

import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.concepts import CONCEPT_NAMES
from src.data import create_dataloaders


def main():
    train_loader, val_loader, test_loader, class_names = create_dataloaders(batch_size=4)

    print("Class names:")
    for idx, class_name in enumerate(class_names):
        print(f"  {idx}: {class_name}")

    print("\nConcept names:")
    for idx, concept_name in enumerate(CONCEPT_NAMES):
        print(f"  {idx}: {concept_name}")

    print("\nDataset sizes:")
    print(f"  train: {len(train_loader.dataset)}")
    print(f"  val:   {len(val_loader.dataset)}")
    print(f"  test:  {len(test_loader.dataset)}")

    x_batch, y_batch, c_batch = next(iter(train_loader))

    print("\nBatch shapes:")
    print(f"  x: {tuple(x_batch.shape)}")
    print(f"  y: {tuple(y_batch.shape)}")
    print(f"  c: {tuple(c_batch.shape)}")

    print("\nExamples from the first training batch:")
    for i in range(len(y_batch)):
        label = y_batch[i].item()
        class_name = class_names[label]
        concepts = c_batch[i].tolist()
        print(f"  example {i}")
        print(f"    label: {label}")
        print(f"    class: {class_name}")
        print(f"    concepts: {concepts}")


if __name__ == "__main__":
    main()
