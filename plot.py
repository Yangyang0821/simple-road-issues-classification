import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataclasses import asdict
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pathlib import Path

from config import Config, Output_paths, PROJECT_DIR
from cnn_model import per_class_accuracy, precision_recall_f1, recall_and_f1


def save_and_show(fig: Figure, filename: str, show: bool = True):
    out_dir = Path(Output_paths.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / filename
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def to_numpy(x) -> np.ndarray:
    return x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


# Loss / Accuracy / Balanced accuracy
def plot_training_curves(history: dict,
                         test_loss: float,
                         test_accuracy: float,
                         test_balanced_acc: float | None = None,
                         show: bool = True):
    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = history.get("best_epoch")
    metric = history.get("metric", "accuracy")
    # Test metrics are computed with the best checkpoint (weights saved at best_epoch),
    # so the test point is placed at best_epoch, not at the last epoch.
    test_epoch = best_epoch if best_epoch else len(history["train_loss"])
    fig, (ax_loss, ax_acc, ax_bal) = plt.subplots(1, 3, figsize=(16, 6))

    # Loss
    ax_loss.plot(epochs, history["train_loss"], color="tab:blue", linewidth=2, label="Train")
    ax_loss.plot(epochs, history["val_loss"], color="tab:red", linewidth=2, label="Validation")
    ax_loss.set_title("Loss", fontsize=15)
    ax_loss.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax_loss.scatter(test_epoch, test_loss, color="tab:orange", s=75, zorder=5,
                    label=f"Test, best model ({test_loss:.4f})")

    # Accuracy
    ax_acc.plot(epochs, history["train_acc"], color="tab:blue", linewidth=2, label="Train")
    ax_acc.plot(epochs, history["val_acc"], color="tab:red", linewidth=2, label="Validation")
    ax_acc.set_title("Accuracy", fontsize=15)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=12)
    ax_acc.set_ylim(0, 100)
    ax_acc.scatter(test_epoch, test_accuracy, color="tab:orange", s=75, zorder=5,
                   label=f"Test, best model ({test_accuracy:.2f}%)")

    # Balanced accuracy
    ax_bal.plot(epochs, history["val_acc"], color="tab:red", linewidth=2,
                alpha=0.5, label="Val accuracy")
    ax_bal.plot(epochs, history["val_balanced_acc"], color="tab:purple", linewidth=2,
                label="Val balanced accuracy")
    ax_bal.set_title("Validation: Accuracy vs Balanced Accuracy", fontsize=15)
    ax_bal.set_ylabel("(%)", fontsize=12)
    ax_bal.set_ylim(0, 100)
    if test_balanced_acc is not None:
        ax_bal.scatter(test_epoch, test_balanced_acc, color="tab:orange", s=75, zorder=5,
                       label=f"Test balanced acc, best model ({test_balanced_acc:.2f}%)")

    # Show best epoch
    if best_epoch:
        for ax in (ax_loss, ax_acc, ax_bal):
            ax.axvline(best_epoch, color="green", linestyle=":", linewidth=1.5,
                       label=f"Best epoch {best_epoch} ({metric})")

    for ax in (ax_loss, ax_acc, ax_bal):
        ax.set_xlabel("Epoch", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    fig.suptitle("Training & Validation Curves", fontsize=18)
    fig.tight_layout()
    save_and_show(fig, "training_curves.png", show)


def draw_per_class_accuracy(ax: Axes, cm: torch.Tensor, class_names: list[str], title: str):
    acc, tot = per_class_accuracy(cm)
    acc, tot = to_numpy(acc), to_numpy(tot)
    overall = 100 * cm.diag().sum().item() / max(cm.sum().item(), 1)

    bars = ax.bar(class_names, acc, color="#CCCCCC", edgecolor="black")
    for bar, a, t in zip(bars, acc, tot):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{a:.1f}%\n(n={int(t)})", ha="center", va="bottom", fontsize=9)

    ax.axhline(overall, color="gray", linestyle="--", linewidth=1.5,
               label=f"Overall accuracy: {overall:.2f}%")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 120)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")


def draw_metrics_table(ax: Axes, cm: torch.Tensor, class_names: list[str], title: str):
    header, rows = report_rows(cm, class_names)
    ax.axis("off")
    ax.set_title(title, fontsize=14)
    table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(header))))
    table.scale(1.5, 2)
    for (row, col), cell in table.get_celld().items():
        if row == 0:                                   # Header
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#CCCCCC")
        elif row > len(class_names):                   # Overall accuracy / Balanced accuracy / Macro F1
            cell.set_facecolor("#CCCCCC")
            cell.set_text_props(fontweight="bold")


def draw_confusion_matrix(ax: Axes, cm: torch.Tensor, class_names: list[str], title: str):
    cm_np = to_numpy(cm)
    row_sums = cm_np.sum(axis=1, keepdims=True)
    cm_pct = cm_np / np.maximum(row_sums, 1) * 100

    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="Row-normalized (%)")

    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(title, fontsize=14)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm_pct[i, j]:.1f}%\n({cm_np[i, j]})",
                    ha="center", va="center", fontsize=8,
                    fontweight="bold" if i == j else "normal",
                    color="white" if cm_pct[i, j] > 50 else "black")


# Confusion matrix / Per class accuracy
def plot_evaluation_result(test_cm: torch.Tensor,
                           val_cm: torch.Tensor,
                           class_names: list[str],
                           show: bool = True):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    draw_per_class_accuracy(axes[0, 0], test_cm, class_names, "Test: Per-class Accuracy")
    draw_confusion_matrix(axes[0, 1], val_cm, class_names, "Validation: Confusion Matrix")
    draw_metrics_table(axes[1, 0], test_cm, class_names, "Test: Per-class Metrics")
    draw_confusion_matrix(axes[1, 1], test_cm, class_names, "Test: Confusion Matrix")

    fig.suptitle("Evaluation Result (best model)", fontsize=18)
    fig.tight_layout()
    save_and_show(fig, "evaluation_result.png", show)


def report_rows(cm: torch.Tensor, class_names: list[str]) -> tuple[list[str], list[list]]:
    # Confusion Matrix -> (header, rows)
    acc, totals = per_class_accuracy(cm)
    precision, recall, f1 = precision_recall_f1(cm)
    header = ["Class", " Samples ", " Correct", " Accuracy(%)", " Precision", " Recall", " F1"]
    rows = []
    for i, name in enumerate(class_names):
        rows.append([name, int(totals[i]), int(cm[i, i]), f"{acc[i]:.2f}",
                     f"{precision[i]:.3f}", f"{recall[i]:.3f}", f"{f1[i]:.3f}"])

    overall = 100 * cm.diag().sum().item() / max(cm.sum().item(), 1)
    balanced, macro_f1 = recall_and_f1(cm)
    rows.append(["Overall accuracy", int(cm.sum()), int(cm.diag().sum()), f"{overall:.2f}", "", "", ""])
    rows.append(["Balanced accuracy", "", "", f"{balanced:.2f}", "", f"{balanced / 100:.3f}", ""])
    rows.append(["Macro F1", "", "", "", "", "", f"{macro_f1 / 100:.3f}"])
    return header, rows


def save_report_csv(cm: torch.Tensor, class_names: list[str], filename: str = "result_test.csv"):
    header, rows = report_rows(cm, class_names)
    out_path = Path(Output_paths.result_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Result saved: {out_path}")


def _relativize_paths(obj):
    # Replace absolute paths under PROJECT_DIR with project-relative ones, so the
    # saved config.json is portable.
    if isinstance(obj, dict):
        return {k: _relativize_paths(v) for k, v in obj.items()}
    if isinstance(obj, str):
        try:
            return Path(obj).relative_to(PROJECT_DIR).as_posix()
        except ValueError:
            return obj
    return obj


def save_config(config: Config, filename: str = "config.json"):
    out_path = Path(Output_paths.result_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_relativize_paths(asdict(config)), f, indent=4)
