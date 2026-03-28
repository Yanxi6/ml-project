from pathlib import Path

import joblib
import pandas as pd

from model import add_features

MODEL_PATH = Path("models/titanic_logreg.joblib")


def main():
    pipe = joblib.load(MODEL_PATH)

    sample = pd.DataFrame([{
        "PassengerId": 9999,
        "Pclass": 3,
        "Name": "Doe, Mr. John",
        "Sex": "male",
        "Age": 22,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "A/5 21171",
        "Fare": 7.25,
        "Cabin": None,
        "Embarked": "S",
    }])

    sample = add_features(sample)

    pred = pipe.predict(sample)[0]
    proba = pipe.predict_proba(sample)[0].max()
    print(f"Predicted Survived={pred} (confidence={proba:.3f})")


if __name__ == "__main__":
    main()
