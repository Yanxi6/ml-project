from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

from model import build_pipeline, split_xy, add_features

DATA_PATH = Path("data/raw/titanic_train.csv")
MODEL_PATH = Path("models/titanic_logreg.joblib")


def main():
    df = pd.read_csv(DATA_PATH)
    df = add_features(df)

    X, y = split_xy(df)

    # Cross-validation (more reliable estimate)
    pipe_cv = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe_cv, X, y, cv=cv, scoring="accuracy")
    print(f"5-fold CV accuracy: mean={scores.mean():.4f} std={scores.std():.4f}")

    # Keep a simple validation split for a readable report
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_val)
    acc = accuracy_score(y_val, preds)

    print(f"Validation accuracy: {acc:.4f}")
    print(classification_report(y_val, preds))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
