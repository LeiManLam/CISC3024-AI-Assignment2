# CISC3024 Pattern Recognition — AI Assignment #2

**Monte Carlo Dropout as a Bayesian Approximation for Image Classification**

- Course: CISC3024 Pattern Recognition, University of Macau  
- Date: October 2026  
- **Public repository:** https://github.com/LeiManLam/CISC3024-AI-Assignment2

## Algorithm

**Monte Carlo Dropout (MC Dropout)** — Gal & Ghahramani, ICML 2016  

Paper: [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142) (arXiv:1506.02142)

Dropout training is cast as approximate variational inference in a deep Gaussian process. At test time, dropout remains active; averaging \(T\) stochastic forward passes approximates the Bayesian predictive posterior and yields uncertainty estimates (predictive entropy, calibrated probabilities).

Related recent work considered: MC-DropConnect (Mobiny et al., Scientific Reports 2021, arXiv:1906.04569).

## Dataset

Fashion-MNIST (Zalando): 60,000 training + 10,000 test grayscale 28×28 images, 10 clothing classes.

## Results (this run)

| Metric | Value |
|--------|-------|
| Deterministic test accuracy | 91.11% |
| MC Dropout test accuracy (T=30) | 91.07% |
| Mean NLL | 0.2450 |
| ECE (15 bins) | 0.0199 |
| Mean entropy (correct / incorrect) | 0.239 / 0.891 |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run experiment

```bash
cd src
python train_eval.py
```

Optional (evaluate an existing checkpoint only):

```bash
python eval_only.py
```

Outputs:
- `checkpoints/bayesian_cnn_mc_dropout.pt`
- `results/metrics.json`
- `figures/` (training curves, confusion matrix, reliability diagram, entropy histograms)
- `report/AI_Assignment2_CISC3024.pdf` and `.docx`

## Project layout

```
├── README.md
├── requirements.txt
├── src/
│   ├── model.py           # BayesianCNN + MC Dropout
│   ├── metrics.py         # Accuracy, NLL, ECE
│   ├── train_eval.py      # Full train + evaluate
│   ├── eval_only.py       # Evaluate from checkpoint
│   └── generate_report.py # Rebuild PDF/DOCX
├── data/                  # Fashion-MNIST (auto-downloaded; gitignored)
├── checkpoints/
├── results/
├── figures/
└── report/
```

## Citation

```bibtex
@inproceedings{gal2016dropout,
  title={Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning},
  author={Gal, Yarin and Ghahramani, Zoubin},
  booktitle={ICML},
  pages={1050--1059},
  year={2016}
}
```

## Note

Per course rules, all code and the report were produced by an AI agent; the student did not write the source code.
