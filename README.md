# Week 6 — Gaussian Naive Bayes Classifier from Scratch (NumPy)

## Learning goal

Build intuition for **generative classification** by implementing **Gaussian Naive Bayes** yourself with NumPy: class priors, per-class mean/variance, log-likelihood under a diagonal Gaussian, predict via argmax of the log-posterior, optional variance smoothing for stability, a simple train/test split on synthetic multi-class Gaussian blobs, accuracy + confusion counts, and a 2D decision-region plot.

## What you'll build

- Synthetic 2D three-class Gaussian blobs (no downloads, no scikit-learn for the core)
- A `GaussianNaiveBayes` class that learns priors + per-class feature means/variances
- Log-likelihood under independent (diagonal) Gaussians and log-posterior scoring
- Optional **variance smoothing** (Laplace-style floor) for numerical stability
- Test-set accuracy, simple confusion counts, and a 2D decision-region visualization

## How to run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Script (VS Code / terminal):**

```bash
python naive_bayes_scratch.py
```

This generates 2D multi-class blobs, splits train/test, fits Gaussian NB, prints priors / means / variances, test accuracy + confusion counts, and saves `outputs/decision_regions.png`.

**Notebook (interactive):**

```bash
jupyter notebook notebooks/naive_bayes_scratch.ipynb
```

Both share the same core ideas. The notebook adds markdown intuition and inline plots; the `.py` script is a clean, runnable walkthrough.

## Requirements

- Python 3.9+
- `numpy`, `matplotlib`, `jupyter` (see `requirements.txt`)
- No scikit-learn for the Naive Bayes core — the ML logic and toy data are from scratch

## What you'll learn

- How **class priors** and **class-conditional densities** combine into a posterior via Bayes' rule
- Why the **naive** assumption (features independent given the class) yields a product of 1D Gaussians
- How **log-space** scoring avoids underflow and turns products into sums
- Why a small **variance floor / smoothing** keeps `log` and division numerically safe
- How Gaussian NB draws smooth, elliptical decision regions on a 2D multi-class toy set
