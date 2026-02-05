# Kaggle Titanic Project

This repository hosts a lightweight, reproducible baseline for the Kaggle Titanic practice
competition. It focuses on a simple scikit-learn pipeline that you can run locally to create
a valid submission file.

## Repository layout

```
.
├── data/                 # Place Kaggle train.csv/test.csv here (not committed)
├── output/               # Generated predictions
├── src/
│   └── train.py          # Baseline training + submission generation
├── requirements.txt
└── README.md
```

## Quickstart

1. **Create a virtual environment (recommended).**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies.**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the data.**

   - Join the competition on Kaggle: https://www.kaggle.com/competitions/titanic
   - Download `train.csv` and `test.csv` from the **Data** tab.
   - Place them in `data/` so you have:

   ```
   data/train.csv
   data/test.csv
   ```

4. **Run the baseline.**

   ```bash
   python src/train.py --data-dir data --output-dir output
   ```

5. **Submit to Kaggle.**

   Upload `output/submission.csv` on the Kaggle submission page.

## Notes

- This baseline uses a straightforward preprocessing pipeline:
  - Numerical columns: median imputation.
  - Categorical columns: most-frequent imputation + one-hot encoding.
- You can improve scores by feature engineering (family size, titles), model tuning,
  or experimenting with other algorithms.

## Next steps

- Try creating a validation split to evaluate locally.
- Explore Kaggle notebooks for ideas and feature engineering techniques.
- Keep track of experiments and scores in a `docs/experiments.md` log.
