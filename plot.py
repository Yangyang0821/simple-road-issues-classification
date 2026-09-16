"""所有「呈現」都在這裡：兩張圖 + 一份 CSV。

1. training_curves.png    - 訓練過程的折線圖（loss / accuracy / balanced accuracy）
2. evaluation_report.png  - best model 的評估結果（per-class accuracy、confusion matrix、指標表格）
3. report_test.csv        - 表格的數字版，方便貼到 README
"""
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

from config import Output_paths
from cnn_model import per_class_accuracy, precision_recall_f1, recall_and_f1


def _save_and_show(fig: plt.Figure, filename: str, show: bool = True):
    out_dir = Path(Output_paths.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / filename
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def _to_numpy(x) -> np.ndarray:
    return x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


# ---------------------------------------------------------------------------
# 圖 1：訓練曲線（1 x 3 子圖）
# ---------------------------------------------------------------------------
def plot_training_curves(history: dict, show: bool = True):
    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = history.get("best_epoch")
    metric = history.get("metric", "accuracy")

    fig, (ax_loss, ax_acc, ax_bal) = plt.subplots(1, 3, figsize=(20, 6))

    # --- Loss ---
    ax_loss.plot(epochs, history["train_loss"], marker="o", color="tab:blue", linewidth=2, label="Train")
    ax_loss.plot(epochs, history["val_loss"], marker="o", color="tab:red", linewidth=2, label="Validation")
    ax_loss.set_title("Loss", fontsize=15)
    ax_loss.set_ylabel("Cross-Entropy Loss", fontsize=12)

    # --- Accuracy ---
    ax_acc.plot(epochs, history["train_acc"], marker="o", color="tab:blue", linewidth=2, label="Train")
    ax_acc.plot(epochs, history["val_acc"], marker="o", color="tab:red", linewidth=2, label="Validation")
    ax_acc.set_title("Accuracy", fontsize=15)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=12)
    ax_acc.set_ylim(0, 100)

    # --- Balanced accuracy (validation only; train 沒有算) ---
    ax_bal.plot(epochs, history["val_acc"], marker="o", color="tab:red", linewidth=2,
                alpha=0.4, label="Val accuracy")
    ax_bal.plot(epochs, history["val_balanced_acc"], marker="s", color="tab:purple", linewidth=2,
                label="Val balanced accuracy")
    ax_bal.set_title("Validation: Accuracy vs Balanced Accuracy", fontsize=15)
    ax_bal.set_ylabel("(%)", fontsize=12)
    ax_bal.set_ylim(0, 100)

    # 標出 best model 的 epoch（用 config 裡選的 metric）
    if best_epoch:
        for ax in (ax_loss, ax_acc, ax_bal):
            ax.axvline(best_epoch, color="green", linestyle=":", linewidth=1.5,
                       label=f"Best epoch {best_epoch} ({metric})")

    for ax in (ax_loss, ax_acc, ax_bal):
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_xticks(list(epochs))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    fig.suptitle("Training & Validation Curves", fontsize=18)
    fig.tight_layout()
    _save_and_show(fig, "training_curves.png", show)


# ---------------------------------------------------------------------------
# 圖 2：評估報告（2 x 2 子圖）
#   左上 per-class accuracy 長條圖  右上 test confusion matrix
#   左下 指標表格                    右下 validation confusion matrix
# ---------------------------------------------------------------------------
def _draw_confusion_matrix(ax: plt.Axes, cm: torch.Tensor, class_names: list[str], title: str):
    cm_np = _to_numpy(cm)
    row_sums = cm_np.sum(axis=1, keepdims=True)
    cm_pct = cm_np / np.maximum(row_sums, 1) * 100

    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized (%)")

    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
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


def _draw_per_class_accuracy(ax: plt.Axes, cm: torch.Tensor, class_names: list[str], title: str):
    acc, tot = per_class_accuracy(cm)
    acc, tot = _to_numpy(acc), _to_numpy(tot)
    overall = 100 * cm.diag().sum().item() / max(cm.sum().item(), 1)

    colors = ["tab:green" if a >= 80 else "tab:orange" if a >= 60 else "tab:red" for a in acc]
    bars = ax.bar(class_names, acc, color=colors, edgecolor="black")
    for bar, a, t in zip(bars, acc, tot):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{a:.1f}%\n(n={int(t)})", ha="center", va="bottom", fontsize=9)

    ax.axhline(overall, color="gray", linestyle="--", linewidth=1.5,
               label=f"Overall accuracy: {overall:.2f}%")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")


def _report_rows(cm: torch.Tensor, class_names: list[str]) -> tuple[list[str], list[list]]:
    """把表格資料整理成 (header, rows)，畫表格與存 CSV 共用。"""
    acc, totals = per_class_accuracy(cm)
    precision, recall, f1 = precision_recall_f1(cm)
    header = ["Class", "Samples", "Correct", "Accuracy(%)", "Precision", "Recall", "F1"]
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


def _draw_metrics_table(ax: plt.Axes, cm: torch.Tensor, class_names: list[str], title: str):
    header, rows = _report_rows(cm, class_names)
    ax.axis("off")
    ax.set_title(title, fontsize=14)
    table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:                                   # 表頭
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#dbe5f1")
        elif row > len(class_names):                   # 最後三列總指標
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(fontweight="bold")


def plot_evaluation_report(test_cm: torch.Tensor,
                           val_cm: torch.Tensor,
                           class_names: list[str],
                           show: bool = True):
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    _draw_per_class_accuracy(axes[0, 0], test_cm, class_names, "Test: Per-class Accuracy")
    _draw_confusion_matrix(axes[0, 1], test_cm, class_names, "Test: Confusion Matrix")
    _draw_metrics_table(axes[1, 0], test_cm, class_names, "Test: Per-class Metrics")
    _draw_confusion_matrix(axes[1, 1], val_cm, class_names, "Validation: Confusion Matrix")

    fig.suptitle("Evaluation Report (best model)", fontsize=18)
    fig.tight_layout()
    _save_and_show(fig, "evaluation_report.png", show)


def save_report_csv(cm: torch.Tensor, class_names: list[str], filename: str = "report_test.csv"):
    header, rows = _report_rows(cm, class_names)
    out_path = Path(Output_paths.result_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Report saved: {out_path}")
