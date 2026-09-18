import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader

from config import Hyperparameters, CNN_Parameters, Output_paths

class RoadIssuesCNN(nn.Module):
    def __init__(self):
        super(RoadIssuesCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=CNN_Parameters.input_channels,
                               out_channels=CNN_Parameters.conv1_out_channels,
                               kernel_size=CNN_Parameters.conv_kernel_size,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=CNN_Parameters.conv1_out_channels,
                               out_channels=CNN_Parameters.conv2_out_channels,
                               kernel_size=CNN_Parameters.conv_kernel_size,
                               padding=1)
        self.conv3 = nn.Conv2d(in_channels=CNN_Parameters.conv2_out_channels,
                               out_channels=CNN_Parameters.conv3_out_channels,
                               kernel_size=CNN_Parameters.conv_kernel_size,
                               padding=1)

        self.dropout1 = nn.Dropout(p=CNN_Parameters.dropout1_rate)
        self.dropout2 = nn.Dropout(p=CNN_Parameters.dropout2_rate)
        self.dropout3 = nn.Dropout(p=CNN_Parameters.dropout3_rate)

        self.maxpool1 = nn.MaxPool2d(kernel_size=CNN_Parameters.pool_kernel_size, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=CNN_Parameters.pool_kernel_size, stride=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=CNN_Parameters.pool_kernel_size, stride=2)

        self.bn1 = nn.BatchNorm2d(num_features=CNN_Parameters.bn1_in_channels)
        self.bn2 = nn.BatchNorm2d(num_features=CNN_Parameters.bn2_in_channels)
        self.bn3 = nn.BatchNorm2d(num_features=CNN_Parameters.bn3_in_channels)

        self.fc1 = nn.Linear(in_features=CNN_Parameters.fc1_in, out_features=CNN_Parameters.fc2_in)
        self.fc2 = nn.Linear(in_features=CNN_Parameters.fc2_in, out_features=CNN_Parameters.fc3_in)
        self.fc3 = nn.Linear(in_features=CNN_Parameters.fc3_in, out_features=CNN_Parameters.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


def set_model(class_weights: torch.Tensor | None = None
              ) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer, torch.device]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RoadIssuesCNN().to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print("Using weighted CrossEntropyLoss, class weights:",
              [f"{w:.2f}" for w in class_weights.tolist()])
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=Hyperparameters.learning_rate)
    return model, loss_function, optimizer, device


def train_epoch(
        device: torch.device,
        loss_function: nn.Module,
        model:nn.Module,
        optimizer:torch.optim.Optimizer,
        train_data:DataLoader):
    model.train()
    correct_train, total_train, total_train_loss = 0, 0, 0
    for input_data, labels in train_data:
        inputs, labels = input_data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        # Comparison results
        _, predicted = outputs.max(1)
        correct_train += predicted.eq(labels).sum().item()
        total_train += labels.size(0)
    epoch_train_accuracy = 100 * correct_train / total_train
    epoch_train_loss = total_train_loss / len(train_data)

    return epoch_train_accuracy, epoch_train_loss

def validate_epoch(
        device: torch.device,
        loss_function: nn.Module,
        model:nn.Module,
        val_loader:DataLoader):

    model.eval()
    total_eval_loss, correct_eval, total_eval = 0, 0, 0
    num_classes = CNN_Parameters.num_classes
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = loss_function(logits, labels)

            # Accuracy calculation
            total_eval_loss += loss.item()
            _, predicted = logits.max(1)
            correct_eval += predicted.eq(labels).sum().item()
            total_eval += labels.size(0)
            cm += compute_confusion_matrix(predicted.cpu(), labels.cpu(), num_classes)

    eval_accuracy = 100 * correct_eval / total_eval
    eval_loss = total_eval_loss / len(val_loader)

    return eval_accuracy, eval_loss, cm


def compute_confusion_matrix(
        predictions: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int = CNN_Parameters.num_classes) -> torch.Tensor:
    # Returns a [num_classes, num_classes] confusion matrix.
    # Rows represent the ground-truth classes, while columns represent the predicted classes.
    # The diagonal contains the number of correct predictions.
    # Off-diagonal entries represent misclassifications:
    # ground-truth class i was predicted as class j.
    #
    # Implementation:
    # Encode each (ground-truth, prediction) pair as a single integer using the formula: label * num_classes + prediction.
    # torch.bincount() is then used to count the occurrences of each pair.
    indices = labels * num_classes + predictions
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def per_class_accuracy(cm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Caculate every class recall(accuracy) and dataset num by confusion matrix
    correct_per_class = cm.diag().float()
    total_per_class = cm.sum(dim=1).float()
    accuracy = torch.where(total_per_class > 0,
                           100 * correct_per_class / total_per_class.clamp(min=1),
                           torch.zeros_like(total_per_class))
    return accuracy, total_per_class.long()


def precision_recall_f1(cm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Class precision / recall / F1 (0~1)
    tp = cm.diag().float()
    predicted_per_class = cm.sum(dim=0).float()   # column sums
    actual_per_class = cm.sum(dim=1).float()      # row sums
    precision = tp / predicted_per_class.clamp(min=1)
    recall = tp / actual_per_class.clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    return precision, recall, f1


def recall_and_f1(cm: torch.Tensor) -> tuple[float, float]:
    #balanced accuracy = all class recall average；macro F1 = all class F1 average
    _, recall, f1 = precision_recall_f1(cm)
    return 100 * recall.mean().item(), 100 * f1.mean().item()


def load_best_model(model: nn.Module, device: torch.device,
                    path: str = Output_paths.best_model_path) -> nn.Module:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model


def model_training(device: torch.device,
        loss_function: nn.Module,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader) -> dict:

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [], "val_balanced_acc": [],
        "best_epoch": 0, "metric": "",
    }
    best_epoch, best_score = 0, 0
    metric = Hyperparameters.model_selection_metric   # "accuracy" or "balanced_accuracy"

    Path(Output_paths.result_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, Hyperparameters.num_epochs + 1):
        train_acc, train_loss = train_epoch(device, loss_function, model, optimizer, train_loader)
        val_acc, val_loss, val_cm = validate_epoch(device, loss_function, model, val_loader)
        val_balanced = recall_and_f1(val_cm)[0]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_balanced_acc"].append(val_balanced)

        # Best model select
        score = val_balanced if metric == "balanced_accuracy" else val_acc
        is_best = score > best_score
        if is_best:
            best_score, best_epoch = score, epoch
            torch.save(model.state_dict(), Output_paths.best_model_path)

        print(f"Epoch {epoch:>3}/{Hyperparameters.num_epochs} | "
              f"train loss {train_loss:.4f} / accuracy {train_acc:6.2f}% | "
              f"val loss {val_loss:.4f} / accuracy {val_acc:6.2f}% / balance-accuracy {val_balanced:6.2f}%"
              + (" <- best saved" if is_best else ""))

    print(f"Training finished.")
    print(f"Best epoch:{best_epoch} (val {metric} {best_score:.2f}%)")
    history["best_epoch"] = best_epoch
    history["metric"] = metric
    return history
