import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}. Download train.csv from Kaggle.")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}. Download test.csv from Kaggle.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def build_pipeline(train_df: pd.DataFrame) -> Pipeline:
    feature_cols = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
    ]

    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = LogisticRegression(max_iter=1000)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.feature_cols = feature_cols
    return pipeline


def train_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    pipeline = build_pipeline(train_df)

    x_train = train_df[pipeline.feature_cols]
    y_train = train_df["Survived"]
    x_test = test_df[pipeline.feature_cols]

    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    submission = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Survived": predictions}
    )
    return submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Titanic baseline model.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing train.csv and test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to write submission.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df, test_df = load_data(args.data_dir)
    submission = train_and_predict(train_df, test_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    main()
