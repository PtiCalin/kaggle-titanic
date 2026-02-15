import argparse
from pathlib import Path

import pandas as pd
from joblib import load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Titanic predictions.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models") / "titanic_pipeline.joblib",
        help="Path to the trained model artifact.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data") / "test.csv",
        help="CSV file to score.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("output") / "submission.csv",
        help="Output CSV path for predictions.",
    )
    return parser.parse_args()


def resolve_input_csv(input_csv: Path) -> Path:
    if input_csv.exists():
        return input_csv

    nested_csv = input_csv.parent / "titanic" / input_csv.name
    if nested_csv.exists():
        return nested_csv

    raise FileNotFoundError(f"Missing input CSV at {input_csv}.")


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Missing model artifact at {args.model_path}. Run src/train.py first."
        )

    input_csv = resolve_input_csv(args.input_csv)

    model = load(args.model_path)
    input_df = pd.read_csv(input_csv)

    if not hasattr(model, "feature_cols"):
        raise AttributeError("Model is missing feature_cols metadata.")

    features = input_df[model.feature_cols]
    predictions = model.predict(features)

    output_df = pd.DataFrame(
        {"PassengerId": input_df["PassengerId"], "Survived": predictions}
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
