"""03 — a text classifier written from scratch in numpy.

No scikit-learn. Two reasons, one honest and one better than the honest one:
  - honest: scipy is a 36 MB wheel and this machine's connection could not finish it.
  - better: `model.fit()` teaches you nothing. TF-IDF and logistic regression are
    each about thirty lines, and writing them means you can explain every number
    the evaluation prints.

Contains:
  TfidfVectorizer  - fit/transform, sublinear tf, smoothed idf, L2 normalisation
  LogisticRegression - full-batch gradient descent with L2, per-epoch history
  train_test_split, metrics, save/load

The per-epoch history is what makes the training curve real rather than decorative.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TOKEN = re.compile(r"[a-z0-9£$'_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, plus two hand-made features.

    `__num__` and `__url__` exist because 'call 09061701461' and 'call me' are very
    different messages, but every digit string is otherwise a token seen once and
    therefore pruned by min_df. Collapsing them into one symbol gives the model a
    feature it can actually learn from. This is feature engineering -- the part a
    library cannot do for you because it depends on the data.
    """
    lowered = text.lower()
    lowered = re.sub(r"https?://\S+|www\.\S+", " __url__ ", lowered)
    lowered = re.sub(r"\b\d[\d\s-]{4,}\b", " __num__ ", lowered)
    return TOKEN.findall(lowered)


class TfidfVectorizer:
    def __init__(self, min_df: int = 2, max_features: int = 4000):
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray = np.zeros(0, dtype=np.float32)

    def fit(self, documents: list[str]) -> "TfidfVectorizer":
        document_frequency: dict[str, int] = {}
        for document in documents:
            for token in set(tokenize(document)):
                document_frequency[token] = document_frequency.get(token, 0) + 1

        # Drop tokens seen in fewer than min_df documents: they cannot generalise,
        # they only let the model memorise individual training examples.
        kept = [(token, count) for token, count in document_frequency.items()
                if count >= self.min_df]
        kept.sort(key=lambda pair: (-pair[1], pair[0]))
        kept = kept[:self.max_features]

        self.vocabulary_ = {token: index for index, (token, _) in enumerate(kept)}
        total = len(documents)
        # Smoothed idf: log((1+n)/(1+df)) + 1. The +1s stop a division by zero and
        # stop a token present in every document from getting weight exactly 0.
        self.idf_ = np.array(
            [np.log((1 + total) / (1 + count)) + 1.0 for _, count in kept],
            dtype=np.float32,
        )
        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        matrix = np.zeros((len(documents), len(self.vocabulary_)), dtype=np.float32)
        for row, document in enumerate(documents):
            counts: dict[int, int] = {}
            for token in tokenize(document):
                column = self.vocabulary_.get(token)
                if column is not None:
                    counts[column] = counts.get(column, 0) + 1
            for column, count in counts.items():
                # Sublinear tf: the 20th 'free' in a message is not 20x the signal
                # of the first.
                matrix[row, column] = (1.0 + np.log(count)) * self.idf_[column]
        # L2 normalise so a long message and a short one are compared on direction,
        # not length. Without this the model largely learns "long = spam".
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        return self.fit(documents).transform(documents)


@dataclass
class LogisticRegression:
    learning_rate: float = 3.0
    epochs: int = 300
    l2: float = 1e-4
    balance_classes: bool = True
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    bias: float = 0.0
    history: list[dict[str, float]] = field(default_factory=list)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Split by sign so exp() never overflows on large positive or negative z.
        out = np.empty_like(z)
        positive = z >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        out[~positive] = exp_z / (1.0 + exp_z)
        return out

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "LogisticRegression":
        samples, features = X.shape
        self.weights = np.zeros(features, dtype=np.float32)
        self.bias = 0.0
        self.history = []

        # Class weighting. Only 13% of this dataset is spam, so an unweighted model
        # minimises loss fastest by leaning towards 'ham' -- which shows up as high
        # accuracy and poor RECALL (it misses real spam). Weighting each spam example
        # by n_ham/n_spam makes a missed spam cost as much as a false alarm.
        weight = np.ones(samples, dtype=np.float32)
        if self.balance_classes:
            positives = float(y.sum())
            negatives = float(samples - positives)
            if positives > 0:
                weight = np.where(y == 1, negatives / positives, 1.0).astype(np.float32)
        weight_total = float(weight.sum())

        for epoch in range(1, self.epochs + 1):
            predicted = self._sigmoid(X @ self.weights + self.bias)
            error = (predicted - y) * weight
            gradient_w = (X.T @ error) / weight_total + self.l2 * self.weights
            gradient_b = float(error.sum() / weight_total)
            self.weights -= self.learning_rate * gradient_w
            self.bias -= self.learning_rate * gradient_b

            if epoch % 10 == 0 or epoch == 1:
                clipped = np.clip(predicted, 1e-7, 1 - 1e-7)
                loss = float(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)).mean())
                entry = {
                    "epoch": epoch,
                    "loss": round(loss, 5),
                    "train_accuracy": round(float(((predicted >= 0.5) == y).mean()), 5),
                }
                if X_val is not None and y_val is not None:
                    val = self._sigmoid(X_val @ self.weights + self.bias)
                    entry["val_accuracy"] = round(float(((val >= 0.5) == y_val).mean()), 5)
                self.history.append(entry)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int8)


def train_test_split(
    documents: list[str], labels: np.ndarray, test_size: float = 0.2, seed: int = 42
):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(documents))
    cut = int(len(documents) * (1 - test_size))
    train_idx, test_idx = order[:cut], order[cut:]
    return (
        [documents[i] for i in train_idx], [documents[i] for i in test_idx],
        labels[train_idx], labels[test_idx],
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    """Precision/recall/F1 by hand, so the numbers are not a black box.

    Positive class = spam. Precision is 'when it says spam, how often is it right';
    recall is 'of all the spam, how much did it catch'. They trade off, which is
    why the threshold is adjustable at prediction time.
    """
    true_positive = int(((y_pred == 1) & (y_true == 1)).sum())
    false_positive = int(((y_pred == 1) & (y_true == 0)).sum())
    false_negative = int(((y_pred == 0) & (y_true == 1)).sum())
    true_negative = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": round((true_positive + true_negative) / len(y_true), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positive": true_positive, "false_positive": false_positive,
        "false_negative": false_negative, "true_negative": true_negative,
    }


# --- persistence: npz for the arrays, json for everything readable ---------

def save(folder: Path, vectorizer: TfidfVectorizer, model: LogisticRegression,
         report: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        folder / "spam_model.npz",
        weights=model.weights, idf=vectorizer.idf_,
        bias=np.array([model.bias], dtype=np.float32),
    )
    (folder / "spam_vocab.json").write_text(
        json.dumps(vectorizer.vocabulary_), encoding="utf-8"
    )
    (folder / "spam_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def load(folder: Path) -> tuple[TfidfVectorizer, LogisticRegression, dict]:
    arrays = np.load(folder / "spam_model.npz")
    vectorizer = TfidfVectorizer()
    vectorizer.vocabulary_ = json.loads(
        (folder / "spam_vocab.json").read_text(encoding="utf-8")
    )
    vectorizer.idf_ = arrays["idf"]
    model = LogisticRegression()
    model.weights = arrays["weights"]
    model.bias = float(arrays["bias"][0])
    report = json.loads((folder / "spam_report.json").read_text(encoding="utf-8"))
    return vectorizer, model, report


def top_features(vectorizer: TfidfVectorizer, model: LogisticRegression, n: int = 12):
    """The most spammy and most legitimate tokens the model learned.

    This is the check that catches a model that scored well for the wrong reason.
    """
    inverse = {index: token for token, index in vectorizer.vocabulary_.items()}
    order = np.argsort(model.weights)
    lowest = [(inverse[int(i)], round(float(model.weights[i]), 3)) for i in order[:n]]
    highest = [(inverse[int(i)], round(float(model.weights[i]), 3)) for i in order[-n:][::-1]]
    return {"spam_indicators": highest, "ham_indicators": lowest}
