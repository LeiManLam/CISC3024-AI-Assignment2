"""
Bayesian CNN via Monte Carlo Dropout (Gal & Ghahramani, ICML 2016).

Dropout training is interpreted as approximate variational inference
in a deep Gaussian process. Keeping dropout active at test time and
averaging T stochastic forward passes yields a Monte Carlo estimate
of the predictive posterior:

    q(y*|x*) ≈ (1/T) Σ_t p(y*|x*, ω̂_t),  ω̂_t ~ q(ω)

Reference:
  Yarin Gal, Zoubin Ghahramani.
  "Dropout as a Bayesian Approximation: Representing Model Uncertainty
   in Deep Learning." ICML 2016. arXiv:1506.02142
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianCNN(nn.Module):
    """Small CNN with dropout after every major block (MC Dropout ready)."""

    def __init__(self, num_classes: int = 10, dropout_p: float = 0.25):
        super().__init__()
        self.dropout_p = dropout_p

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(p=dropout_p)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout_fc = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout_conv(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout_conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return self.fc2(x)

    def enable_mc_dropout(self) -> None:
        """Force all Dropout layers into train mode (stochastic at inference)."""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

    @torch.no_grad()
    def mc_predict(
        self,
        x: torch.Tensor,
        t: int = 30,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout predictive mean and per-sample predictive entropy.

        Returns
        -------
        mean_probs : (N, C) averaged softmax probabilities
        entropy    : (N,) predictive entropy H[mean_probs]
        """
        was_training = self.training
        self.eval()
        self.enable_mc_dropout()

        probs_sum = None
        for _ in range(t):
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            probs_sum = probs if probs_sum is None else probs_sum + probs

        mean_probs = probs_sum / t
        # Predictive entropy as epistemic+aleatoric uncertainty summary
        entropy = -(mean_probs * (mean_probs + 1e-12).log()).sum(dim=-1)

        if was_training:
            self.train()
        else:
            self.eval()
        return mean_probs, entropy
