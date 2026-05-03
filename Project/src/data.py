"""Dataset and dataloader utilities for Fashion-MNIST."""

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from src.concepts import CLASS_NAMES, label_to_concept_vector


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


class FashionMNISTWithConcepts(Dataset):
    """Return image, class label, and concept vector."""

    def __init__(self, root="./data", train=True, download=True):
        self.dataset = datasets.FashionMNIST(
            root=root,
            train=train,
            transform=transform,
            download=download,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        c = label_to_concept_vector(y)
        return x, y, c


def create_dataloaders(
    batch_size=64,
    data_dir="./data",
    val_split=0.2,
    seed=42,
    num_workers=0,
    download=True,
):
    """Create train, validation, and test dataloaders."""
    full_train_dataset = FashionMNISTWithConcepts(
        root=data_dir,
        train=True,
        download=download,
    )
    test_dataset = FashionMNISTWithConcepts(
        root=data_dir,
        train=False,
        download=download,
    )

    train_size = int((1.0 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, CLASS_NAMES
