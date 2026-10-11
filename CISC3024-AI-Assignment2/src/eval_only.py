"""Load trained checkpoint and finish MC Dropout evaluation + plots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

from metrics import (
    accuracy,
    confusion_matrix_counts,
    expected_calibration_error,
    negative_log_likelihood,
)
from model import BayesianCNN
from train_eval import (
    CLASS_NAMES,
    CKPT_DIR,
    FIG_DIR,
    RES_DIR,
    evaluate_deterministic,
    evaluate_mc_dropout,
    get_loaders,
    plot_confusion_matrix,
    plot_entropy_vs_correct,
    plot_reliability_diagram,
    plot_training_curves,
    set_seed,
)

# Partial history from the interrupted run (printed to stdout)
HISTORY = {
    "train_loss": [0.5398, 0.3572, 0.3098, 0.2849, 0.2654, 0.2523, 0.2371, 0.2273],
    "train_acc": [0.8047, 0.8724, 0.8882, 0.8955, 0.9031, 0.9070, 0.9134, 0.9157],
    "val_loss": [0.3561, 0.3338, 0.2787, 0.2673, 0.2519, 0.2466, 0.2451, 0.2399],
    "val_acc": [0.8708, 0.8749, 0.8958, 0.9016, 0.9069, 0.9112, 0.9108, 0.9111],
}


def main():
    set_seed(42)
    for d in (FIG_DIR, RES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    ckpt_path = CKPT_DIR / "bayesian_cnn_mc_dropout.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loaded: {ckpt_path}")

    _, test_loader = get_loaders(batch_size=config["batch_size"])
    model = BayesianCNN(num_classes=10, dropout_p=config["dropout_p"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    criterion = nn.CrossEntropyLoss()

    det_loss, det_acc = evaluate_deterministic(model, test_loader, criterion, device)
    print(f"Deterministic: loss={det_loss:.4f} acc={det_acc:.4f}")

    probs, preds, labels, entropy = evaluate_mc_dropout(
        model, test_loader, device, t=config["mc_samples_t"]
    )
    mc_acc = accuracy(labels, preds)
    nll = negative_log_likelihood(labels, probs)
    ece = expected_calibration_error(labels, probs, n_bins=15)
    cm = confusion_matrix_counts(labels, preds, num_classes=10)
    report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)

    results = {
        **config,
        "device": str(device),
        "train_seconds": None,  # training already finished in prior run
        "deterministic_test_accuracy": round(det_acc, 6),
        "deterministic_test_loss": round(det_loss, 6),
        "mc_dropout_test_accuracy": round(mc_acc, 6),
        "mc_dropout_nll": round(nll, 6),
        "mc_dropout_ece": round(ece, 6),
        "mean_predictive_entropy": round(float(entropy.mean()), 6),
        "mean_entropy_correct": round(float(entropy[labels == preds].mean()), 6),
        "mean_entropy_incorrect": round(float(entropy[labels != preds].mean()), 6),
        "num_test_samples": int(len(labels)),
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "source_code_url": "https://github.com/LeiManLam/CISC3024-AI-Assignment2",
    }

    (RES_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    (RES_DIR / "classification_report.txt").write_text(report)
    np.save(RES_DIR / "confusion_matrix.npy", cm)
    np.savetxt(RES_DIR / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    (RES_DIR / "history.json").write_text(json.dumps(HISTORY, indent=2))

    plot_training_curves(HISTORY, FIG_DIR / "training_curves.png")
    plot_confusion_matrix(cm, FIG_DIR / "confusion_matrix.png")
    plot_reliability_diagram(labels, probs, FIG_DIR / "reliability_diagram.png")
    plot_entropy_vs_correct(
        entropy, labels, preds, FIG_DIR / "entropy_correct_vs_incorrect.png"
    )

    print("\n===== RESULTS =====")
    print(json.dumps(results, indent=2))
    print("\nClassification report:\n", report)


if __name__ == "__main__":
    main()
