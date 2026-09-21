import numpy as np
import torch
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from config import Hyperparameters
from prepare_dataset import prepare, get_class_to_idx

class RoadIssuesDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path, label = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def download_dataset():
    dataset_root = prepare(force=False)
    print("Dataset root:", dataset_root)
    return dataset_root


def get_dataset(dataset_root):
    # Load a dataset from a specified root directory.
    # Args: Dataset_root (str): The root directory of the dataset (dataset/raw).
    # Returns: A list of (image_path, label) tuples.
    class_to_idx = get_class_to_idx()
    image_paths = []
    for class_name, label in class_to_idx.items():
        class_dir = Path(dataset_root) / class_name
        for image_path in class_dir.glob("*.jpg"):
            image_paths.append((image_path, label))

    return image_paths


def split_dataset(image_paths):
    labels = [label for _, label in image_paths]
    train_paths, temp_paths = train_test_split(
        image_paths,
        test_size=Hyperparameters.validation_ratio + Hyperparameters.test_ratio,
        random_state=42,
        stratify=labels
    )

    temp_labels = [label for _, label in temp_paths]
    val_paths, test_paths = train_test_split(
        temp_paths,
        test_size=Hyperparameters.test_ratio / (Hyperparameters.validation_ratio + Hyperparameters.test_ratio),
        random_state=42,
        stratify=temp_labels
    )
    return train_paths, val_paths, test_paths


def load_dataset(dataset_root):
    image_paths = get_dataset(dataset_root)
    train_paths, val_paths, test_paths = split_dataset(image_paths)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    # dataset = RoadIssuesDataset(image_paths, transform=transform)
    train_dataset = RoadIssuesDataset(train_paths, transform=transform)
    val_dataset = RoadIssuesDataset(val_paths, transform=transform)
    test_dataset = RoadIssuesDataset(test_paths, transform=transform)

    print(f"Train / Validation / Test Dataset size: {len(train_dataset)} / {len(val_dataset)} / {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset


def count_labels(dataset: RoadIssuesDataset, num_classes: int) -> torch.Tensor:
    # Return the number of samples for each category in the dataset.
    labels = torch.tensor([label for _, label in dataset.image_paths])
    return torch.bincount(labels, minlength=num_classes)


def compute_class_weights(dataset: RoadIssuesDataset, num_classes: int) -> torch.Tensor:
    counts = count_labels(dataset, num_classes).float()
    weights = 1.0 / counts.clamp(min=1)
    weights = weights / weights.mean()
    return weights


def create_dataloaders(train_dataset, val_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=Hyperparameters.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=Hyperparameters.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=Hyperparameters.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
