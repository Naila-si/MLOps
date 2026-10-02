from pathlib import Path
import json
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "tickets.csv"
MODEL_PATH = ROOT / "artifacts" / "ticket_priority.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"


def train():
    df = pd.read_csv(DATA_PATH)

    missing = {"text", "urgent"} - set(df.columns)

    if missing:
        raise ValueError(f"Kolom wajib hilang: {sorted(missing)}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["urgent"],
        test_size=0.25,
        random_state=42,
        stratify=df["urgent"],
    )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    metrics = {
        "test_rows": int(len(y_test)),
        "precision": round(
            float(
                precision_score(
                    y_test,
                    pred,
                    zero_division=0,
                )
            ),
            4,
        ),
        "recall": round(
            float(
                recall_score(
                    y_test,
                    pred,
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1": round(
            float(
                f1_score(
                    y_test,
                    pred,
                    zero_division=0,
                )
            ),
            4,
        ),
        "classification_report": classification_report(
            y_test,
            pred,
            zero_division=0,
        ),
    }

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, MODEL_PATH)

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2))
    print(f"Model tersimpan di: {MODEL_PATH}")


def predict(text):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Jalankan train lebih dahulu"
        )

    model = joblib.load(MODEL_PATH)

    label = int(
        model.predict([text])[0]
    )

    probability = float(
        model.predict_proba([text])[0][label]
    )

    result = {
        "text": text,
        "urgent": label,
        "confidence": round(probability, 4),
    }

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    command = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "train"
    )

    if command == "train":
        train()

    elif command == "predict" and len(sys.argv) > 2:
        predict(" ".join(sys.argv[2:]))

    else:
        raise SystemExit(
            'Gunakan: python src/pipeline.py train '
            'atau predict "teks tiket"'
        )
