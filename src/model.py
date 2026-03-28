import re
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

TARGET_COL = "Survived"

# Base columns in Titanic dataset:
# PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

NUM_COLS = ["Age", "Fare", "Pclass", "FamilySize"]
CAT_COLS = ["Sex", "Embarked", "Title", "IsAlone", "CabinKnown"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Family features
    df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int).astype(str)  # as categorical "0"/"1"

    # Cabin known
    df["CabinKnown"] = df["Cabin"].notna().astype(int).astype(str)

    # Title from Name
    def extract_title(name: str) -> str:
        if not isinstance(name, str):
            return "Unknown"
        m = re.search(r",\s*([^\.]+)\.", name)
        if not m:
            return "Unknown"
        title = m.group(1).strip()
        # Group rare titles
        rare = {"Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"}
        if title in rare:
            return "Rare"
        if title in {"Mlle", "Ms"}:
            return "Miss"
        if title == "Mme":
            return "Mrs"
        return title

    df["Title"] = df["Name"].apply(extract_title)

    return df


def build_pipeline() -> Pipeline:
    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric, NUM_COLS),
            ("cat", categorical, CAT_COLS),
        ]
    )

    model = LogisticRegression(max_iter=2000)

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])
    return pipe


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y
