"""Train BayesianCNN and evaluate with MC Dropout on Fashion-MNIST."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from metrics import (
    accuracy,
    confusion_matrix_counts,
    expected_calibration_error,
    negative_log_likelihood,
)
from model import BayesianCNN

# Fashion-MNIST class names
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

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
CKPT_DIR = ROOT / "checkpoints"


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_loaders(
    batch_size: int = 128,
    train_subset: int | None = None,
    num_workers: int = 2,
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    train_set = datasets.FashionMNIST(
        DATA_DIR, train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        DATA_DIR, train=False, download=True, transform=transform
    )
    if train_subset is not None and train_subset < len(train_set):
        idx = np.random.RandomState(42).permutation(len(train_set))[:train_subset]
        train_set = Subset(train_set, idx.tolist())

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_deterministic(model, loader, criterion, device):
    """Standard eval: dropout OFF (eval mode)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_mc_dropout(model, loader, device, t: int = 30):
    """MC Dropout: keep dropout ON, average T stochastic forward passes."""
    model.eval()
    model.enable_mc_dropout()

    all_probs, all_entropy, all_labels = [], [], []
    for x, y in tqdm(loader, desc=f"MC Dropout T={t}"):
        x = x.to(device)
        mean_probs, entropy = model.mc_predict(x, t=t)
        all_probs.append(mean_probs.cpu().numpy())
        all_entropy.append(entropy.cpu().numpy())
        all_labels.append(y.numpy())

    probs = np.concatenate(all_probs, axis=0)
    entropy = np.concatenate(all_entropy, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds = probs.argmax(axis=1)
    return probs, preds, labels, entropy


def plot_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix (MC Dropout)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reliability_diagram(
    y_true: np.ndarray, probs: np.ndarray, path: Path, n_bins: int = 15
) -> None:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float64)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
        if np.any(mask):
            bin_accs.append(accuracies[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            bin_counts.append(0)

    fig, ax = plt.subplots(figsize=(6, 5))
    width = 1.0 / n_bins
    centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    ax.bar(
        centers,
        bin_accs,
        width=width * 0.9,
        edgecolor="black",
        alpha=0.7,
        label="Accuracy",
    )
    ax.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram (MC Dropout)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_entropy_vs_correct(
    entropy: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, path: Path
) -> None:
    correct = y_true == y_pred
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(
        entropy[correct],
        bins=40,
        alpha=0.7,
        label=f"Correct (n={correct.sum()})",
        color="steelblue",
        density=True,
    )
    ax.hist(
        entropy[~correct],
        bins=40,
        alpha=0.7,
        label=f"Incorrect (n={(~correct).sum()})",
        color="salmon",
        density=True,
    )
    ax.set_xlabel("Predictive entropy")
    ax.set_ylabel("Density")
    ax.set_title("Predictive Entropy: Correct vs Incorrect")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val (det.)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val (det.)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    set_seed(42)
    for d in (DATA_DIR, FIG_DIR, RES_DIR, CKPT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Hyperparameters (CPU-friendly)
    config = {
        "dataset": "Fashion-MNIST",
        "epochs": 8,
        "batch_size": 128,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout_p": 0.25,
        "mc_samples_t": 30,
        "seed": 42,
        "optimizer": "Adam",
        "algorithm": "Monte Carlo Dropout (Gal & Ghahramani, ICML 2016)",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_loaders(batch_size=config["batch_size"])
    model = BayesianCNN(num_classes=10, dropout_p=config["dropout_p"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    t0 = time.time()
    for epoch in range(1, config["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        va_loss, va_acc = evaluate_deterministic(model, test_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        print(
            f"Epoch {epoch:02d}/{config['epochs']}  "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.4f}  "
            f"val_loss={va_loss:.4f} acc={va_acc:.4f}"
        )

    train_seconds = time.time() - t0
    ckpt_path = CKPT_DIR / "bayesian_cnn_mc_dropout.pt"
    torch.save({"model_state": model.state_dict(), "config": config}, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    # Deterministic baseline (dropout off)
    det_loss, det_acc = evaluate_deterministic(model, test_loader, criterion, device)

    # MC Dropout evaluation
    probs, preds, labels, entropy = evaluate_mc_dropout(
        model, test_loader, device, t=config["mc_samples_t"]
    )
    mc_acc = accuracy(labels, preds)
    nll = negative_log_likelihood(labels, probs)
    ece = expected_calibration_error(labels, probs, n_bins=15)
    cm = confusion_matrix_counts(labels, preds, num_classes=10)
    report = classification_report(
        labels, preds, target_names=CLASS_NAMES, digits=4
    )

    results = {
        **config,
        "device": str(device),
        "train_seconds": round(train_seconds, 2),
        "deterministic_test_accuracy": round(det_acc, 6),
        "deterministic_test_loss": round(det_loss, 6),
        "mc_dropout_test_accuracy": round(mc_acc, 6),
        "mc_dropout_nll": round(nll, 6),
        "mc_dropout_ece": round(ece, 6),
        "mean_predictive_entropy": round(float(entropy.mean()), 6),
        "mean_entropy_correct": round(
            float(entropy[labels == preds].mean()), 6
        ),
        "mean_entropy_incorrect": round(
            float(entropy[labels != preds].mean()), 6
        ),
        "num_test_samples": int(len(labels)),
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
    }

    # Save artifacts
    (RES_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    (RES_DIR / "classification_report.txt").write_text(report)
    np.save(RES_DIR / "confusion_matrix.npy", cm)
    np.savetxt(RES_DIR / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    (RES_DIR / "history.json").write_text(json.dumps(history, indent=2))

    plot_training_curves(history, FIG_DIR / "training_curves.png")
    plot_confusion_matrix(cm, FIG_DIR / "confusion_matrix.png")
    plot_reliability_diagram(labels, probs, FIG_DIR / "reliability_diagram.png")
    plot_entropy_vs_correct(entropy, labels, preds, FIG_DIR / "entropy_correct_vs_incorrect.png")

    print("\n===== RESULTS =====")
    print(json.dumps(results, indent=2))
    print("\nClassification report:\n", report)
    print(f"Figures saved under {FIG_DIR}")
    print(f"Metrics saved under {RES_DIR}")


if __name__ == "__main__":
    main()
