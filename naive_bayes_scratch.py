"""
Week 6 — Gaussian Naive Bayes Classifier from Scratch (NumPy)

Educational script: class priors, per-class mean/variance, log-likelihood
under a diagonal Gaussian, predict via argmax of log-posterior, optional
variance smoothing for stability, train/test on synthetic multi-class
Gaussian blobs, accuracy + confusion counts, and a 2D decision-region plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# 1. Synthetic multi-class Gaussian blobs (no scikit-learn)
# ---------------------------------------------------------------------------

def make_blobs(
    n_per_class: int = 80,
    centers: list[tuple[float, float]] | None = None,
    cluster_std: float = 0.75,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a simple 2D multi-class classification dataset.

    Returns
    -------
    X : (n, 2) float features
    y : (n,) int labels in {0, 1, ..., n_classes-1}
    """
    if centers is None:
        centers = [(-1.8, -1.2), (1.7, -0.6), (0.1, 1.8)]

    rng = np.random.default_rng(seed)
    cov = np.eye(2) * (cluster_std**2)
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for label, center in enumerate(centers):
        mean = np.asarray(center, dtype=float)
        Xk = rng.multivariate_normal(mean, cov, size=n_per_class)
        Xs.append(Xk)
        ys.append(np.full(n_per_class, label, dtype=int))

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple random train/test split (no stratification)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_size)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ---------------------------------------------------------------------------
# 2. Gaussian Naive Bayes
# ---------------------------------------------------------------------------

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes with a diagonal covariance (independent features).

    Model
    -----
    For each class c:
      prior:   P(y=c) = n_c / n
      likelihood (feature j independent given c):
        P(x_j | y=c) = N(x_j; mu_{c,j}, sigma^2_{c,j})

    Prediction uses log-posterior:
      log P(y=c|x) ∝ log P(y=c) + sum_j log N(x_j; mu_{c,j}, sigma^2_{c,j})
    """

    def __init__(self, var_smoothing: float = 1e-9):
        """
        Parameters
        ----------
        var_smoothing : float
            Added to every per-class feature variance for numerical stability
            (Laplace-style floor on variance; prevents log(0) / divide-by-zero).
        """
        self.var_smoothing = float(var_smoothing)
        self.classes_: np.ndarray | None = None
        self.priors_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.vars_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianNaiveBayes:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        n = len(y)

        self.priors_ = np.zeros(n_classes, dtype=float)
        self.means_ = np.zeros((n_classes, n_features), dtype=float)
        self.vars_ = np.zeros((n_classes, n_features), dtype=float)

        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.priors_[i] = len(Xc) / n
            self.means_[i] = Xc.mean(axis=0)
            # population-style variance (ddof=0); add smoothing floor
            self.vars_[i] = Xc.var(axis=0) + self.var_smoothing

        return self

    def _log_gaussian(self, X: np.ndarray) -> np.ndarray:
        """
        Log-likelihood under diagonal Gaussian for every class.

        Parameters
        ----------
        X : (n_samples, n_features)

        Returns
        -------
        log_lik : (n_samples, n_classes)
        """
        assert self.means_ is not None and self.vars_ is not None
        # Broadcast: X[:, None, :] -> (n, 1, d); means (C, d) -> (1, C, d)
        # log N(x; mu, s2) = -0.5 * (log(2πs2) + (x-mu)^2 / s2)
        diff = X[:, None, :] - self.means_[None, :, :]
        log_lik = -0.5 * (
            np.log(2.0 * np.pi * self.vars_[None, :, :])
            + (diff**2) / self.vars_[None, :, :]
        )
        return log_lik.sum(axis=2)  # sum over features (naive independence)

    def log_posterior(self, X: np.ndarray) -> np.ndarray:
        """
        Unnormalized log-posterior: log prior + log likelihood.

        Returns
        -------
        log_post : (n_samples, n_classes)
        """
        assert self.priors_ is not None
        X = np.asarray(X, dtype=float)
        return np.log(self.priors_[None, :]) + self._log_gaussian(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels via argmax of log-posterior."""
        assert self.classes_ is not None
        log_post = self.log_posterior(X)
        return self.classes_[np.argmax(log_post, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Softmax over log-posterior for class probabilities."""
        log_post = self.log_posterior(X)
        # numerically stable softmax
        log_post = log_post - log_post.max(axis=1, keepdims=True)
        probs = np.exp(log_post)
        return probs / probs.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Simple confusion matrix: rows = true, cols = predicted.

    Returns
    -------
    cm : (n_classes, n_classes) int
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t, p] += 1
    return cm


# ---------------------------------------------------------------------------
# 4. Decision-region plot
# ---------------------------------------------------------------------------

def plot_decision_regions(
    model: GaussianNaiveBayes,
    X: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    title: str = "Gaussian Naive Bayes — decision regions",
) -> None:
    """Scatter training points over a 2D grid of predicted class colors."""
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(xx, yy, zz, alpha=0.35, cmap="coolwarm", levels=np.arange(-0.5, 3.5, 1.0))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=40)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    legend = ax.legend(*scatter.legend_elements(), title="class", loc="best")
    ax.add_artist(legend)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved decision-region plot → {out_path}")


# ---------------------------------------------------------------------------
# 5. Main walkthrough
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Week 6 — Gaussian Naive Bayes from Scratch (NumPy)")
    print("=" * 60)

    X, y = make_blobs(n_per_class=80, cluster_std=0.75, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, seed=0)
    n_classes = int(y.max()) + 1

    print(f"\nData: {len(X)} samples, {X.shape[1]} features, {n_classes} classes")
    print(f"Train: {len(y_train)}  |  Test: {len(y_test)}")

    model = GaussianNaiveBayes(var_smoothing=1e-9)
    model.fit(X_train, y_train)

    print("\nLearned class priors:")
    for c, p in zip(model.classes_, model.priors_):
        print(f"  P(y={c}) = {p:.4f}")

    print("\nPer-class means (rows=class, cols=feature):")
    print(np.round(model.means_, 3))
    print("\nPer-class variances (+ smoothing):")
    print(np.round(model.vars_, 3))

    y_pred = model.predict(X_test)
    acc = accuracy(y_test, y_pred)
    cm = confusion_counts(y_test, y_pred, n_classes)

    print(f"\nTest accuracy: {acc:.4f}  ({(y_test == y_pred).sum()}/{len(y_test)})")
    print("Confusion counts (rows=true, cols=pred):")
    print(cm)

    out = Path(__file__).resolve().parent / "outputs" / "decision_regions.png"
    plot_decision_regions(model, X_train, y_train, out)

    print("\nDone. Open the notebook for step-by-step intuition.")


if __name__ == "__main__":
    main()
