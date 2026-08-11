"""Train the project-03 classifier on the real UCI SMS Spam Collection.

    python train_model.py

Dataset: 5,574 real SMS messages, hand-labelled ham/spam.
Source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
Downloaded to data/SMSSpamCollection -- this script will fetch it if missing.

Nothing about the numbers below is decorative: the test set is held out before a
single weight is updated, and the metrics printed are computed on messages the
model has never seen.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile

import numpy as np

from core import mlmodel
from core.config import DATA_DIR, MODEL_DIR

DATASET = DATA_DIR / "SMSSpamCollection"
URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def ensure_dataset() -> None:
    if DATASET.exists():
        return
    print(f"Downloading dataset from {URL} …")
    raw = urllib.request.urlopen(URL, timeout=120).read()
    zipfile.ZipFile(io.BytesIO(raw)).extractall(DATA_DIR)
    print(f"Saved to {DATASET}")


def load_dataset() -> tuple[list[str], np.ndarray]:
    documents: list[str] = []
    labels: list[int] = []
    for line in DATASET.read_text(encoding="utf-8", errors="replace").splitlines():
        if "\t" not in line:
            continue
        label, _, text = line.partition("\t")
        if label not in ("ham", "spam"):
            continue
        documents.append(text)
        labels.append(1 if label == "spam" else 0)
    return documents, np.array(labels, dtype=np.float32)


def main() -> None:
    ensure_dataset()
    documents, labels = load_dataset()
    print(f"Loaded {len(documents)} messages — "
          f"{int(labels.sum())} spam, {int((labels == 0).sum())} ham "
          f"({labels.mean() * 100:.1f}% spam)")

    # Split FIRST, then fit the vectoriser on the training half only. Fitting the
    # vocabulary on all the data before splitting leaks test information into
    # training and inflates every score that follows.
    train_docs, test_docs, y_train, y_test = mlmodel.train_test_split(
        documents, labels, test_size=0.2, seed=42
    )
    print(f"Train {len(train_docs)} · Test {len(test_docs)}")

    vectorizer = mlmodel.TfidfVectorizer(min_df=2, max_features=4000)
    X_train = vectorizer.fit_transform(train_docs)
    X_test = vectorizer.transform(test_docs)
    print(f"Vocabulary {len(vectorizer.vocabulary_)} features · "
          f"matrix {X_train.shape} ({X_train.nbytes / 1e6:.1f} MB dense)")

    model = mlmodel.LogisticRegression(
        learning_rate=6.0, epochs=2000, l2=1e-5, balance_classes=True
    )
    print("Training …")
    model.fit(X_train, y_train, X_test, y_test)
    for entry in model.history[::25]:
        print(f"  epoch {entry['epoch']:>4}  loss {entry['loss']:.4f}  "
              f"train {entry['train_accuracy']:.4f}  val {entry.get('val_accuracy', 0):.4f}")

    predictions = model.predict(X_test)
    scores = mlmodel.metrics(y_test.astype(np.int8), predictions)
    features = mlmodel.top_features(vectorizer, model)

    # A majority-class baseline. Any model has to beat this to have earned anything;
    # 87% "accuracy" sounds fine until you learn that guessing 'ham' every time
    # scores exactly that.
    baseline = float((y_test == 0).mean())

    print("\n--- held-out test set ---")
    for key in ("accuracy", "precision", "recall", "f1"):
        print(f"  {key:<10} {scores[key] * 100:.2f}%")
    print(f"  {'baseline':<10} {baseline * 100:.2f}%  (always predict 'ham')")
    print(f"  confusion: TP {scores['true_positive']} · FP {scores['false_positive']} "
          f"· FN {scores['false_negative']} · TN {scores['true_negative']}")
    print("\n  learned spam words:", ", ".join(t for t, _ in features["spam_indicators"][:8]))
    print("  learned ham words: ", ", ".join(t for t, _ in features["ham_indicators"][:8]))

    report = {
        "dataset": "UCI SMS Spam Collection",
        "dataset_url": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
        "total_messages": len(documents),
        "train_size": len(train_docs),
        "test_size": len(test_docs),
        "spam_share": round(float(labels.mean()), 4),
        "features": len(vectorizer.vocabulary_),
        "epochs": model.epochs,
        "learning_rate": model.learning_rate,
        "l2": model.l2,
        "balanced_classes": model.balance_classes,
        "metrics": scores,
        "majority_baseline_accuracy": round(baseline, 4),
        "history": model.history,
        **features,
    }
    mlmodel.save(MODEL_DIR, vectorizer, model, report)
    print(f"\nSaved model + report to {MODEL_DIR}")


if __name__ == "__main__":
    main()
