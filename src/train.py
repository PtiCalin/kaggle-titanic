import argparse
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def resolve_data_dir(data_dir: Path) -> Path:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if train_path.exists() and test_path.exists():
        return data_dir

    nested_dir = data_dir / "titanic"
    nested_train = nested_dir / "train.csv"
    nested_test = nested_dir / "test.csv"
    if nested_train.exists() and nested_test.exists():
        return nested_dir

    missing = []
    if not train_path.exists() and not nested_train.exists():
        missing.append(str(train_path))
    if not test_path.exists() and not nested_test.exists():
        missing.append(str(test_path))
    raise FileNotFoundError(
        "Missing input files: "
        + ", ".join(missing)
        + ". Download them from Kaggle and place them in data/ or data/titanic/."
    )


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolved_dir = resolve_data_dir(data_dir)
    train_df = pd.read_csv(resolved_dir / "train.csv")
    test_df = pd.read_csv(resolved_dir / "test.csv")
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


def train_and_predict(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, Pipeline]:
    pipeline = build_pipeline(train_df)

    x_train = train_df[pipeline.feature_cols]
    y_train = train_df["Survived"]
    x_test = test_df[pipeline.feature_cols]

    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    submission = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Survived": predictions}
    )
    return submission, pipeline


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
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models") / "titanic_pipeline.joblib",
        help="Path to write the trained model artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df, test_df = load_data(args.data_dir)
    submission, pipeline = train_and_predict(train_df, test_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(pipeline, args.model_path)
    print(f"Saved model artifact to {args.model_path}")


if __name__ == "__main__":
    main()
