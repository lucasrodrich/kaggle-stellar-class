# kaggle-stellar-class

Color-index feature engineering + LightGBM baseline for the Kaggle Playground
Series **S6E6 — Predicting Stellar Class**.

## The competition

[Playground Series S6E6](https://www.kaggle.com/competitions/playground-series-s6e6)
is a multi-class classification task: given photometric and spectroscopic
measurements of an astronomical object, predict its **stellar class** — one of:

| Class    | Meaning                          |
|----------|----------------------------------|
| `GALAXY` | A galaxy                         |
| `QSO`    | A quasar (quasi-stellar object)  |
| `STAR`   | A star                           |

Submissions are scored on **accuracy** (the fraction of objects classified
correctly).

## The approach

1. **Color-index feature engineering.** The five photometric bands `u, g, r, i, z`
   measure brightness through different filters. In astronomy, the *difference*
   between two bands (an object's "color") separates stars, galaxies, and quasars
   far better than raw brightness. We add adjacent color indices (`u_g`, `g_r`, …),
   wider band-skipping indices, per-row magnitude summaries, and redshift
   interactions (redshift being the single strongest physical signal).
2. **LightGBM + 5-fold cross-validation.** A gradient-boosted tree model is trained
   with `StratifiedKFold` (5 folds) so we get an honest accuracy estimate instead of
   an over-optimistic one. The five fold models' probabilities are averaged to form
   the final test prediction.

This baseline reaches roughly **~0.968 cross-validated accuracy**.

## Setup

```bash
python -m venv venv

# Activate the environment:
#   Windows (PowerShell):  venv\Scripts\Activate.ps1
#   Windows (cmd):         venv\Scripts\activate.bat
#   macOS / Linux:         source venv/bin/activate

pip install -r requirements.txt
```

## Get the data

The dataset is **not** committed to this repo (Kaggle redistribution rules + size).
Choose one of:

- **Manual:** download `train.csv` and `test.csv` from the
  [competition data page](https://www.kaggle.com/competitions/playground-series-s6e6/data)
  and place them in `data/`.
- **Kaggle CLI:**

  ```bash
  kaggle competitions download -c playground-series-s6e6 -p data/
  # then unzip the downloaded archive inside data/
  unzip data/playground-series-s6e6.zip -d data/
  ```

After this you should have `data/train.csv` and `data/test.csv`.

## Run

```bash
python src/stellar.py
```

The script prints per-fold accuracy and the overall cross-validated accuracy, then
writes `submission.csv` in the project root, ready to upload to Kaggle.

## Folder structure

```
kaggle-stellar-class/
├── data/            # raw CSVs (train.csv, test.csv) — gitignored, add your own
├── src/             # training scripts
│   └── stellar.py   # LightGBM 5-fold baseline
├── notebooks/       # EDA notebooks (later)
├── submissions/     # generated submissions — gitignored
├── requirements.txt
├── .gitignore
└── README.md
```

## Next steps

- [ ] Exploratory data analysis in `notebooks/` (class balance, feature
      distributions, redshift vs. color separation).
- [ ] Hyperparameter tuning (Optuna) on the LightGBM model.
- [ ] Build a **3-model LightGBM + XGBoost + CatBoost ensemble** and blend the
      out-of-fold probabilities.
- [ ] Add stratified-by-class error analysis to find the hardest objects.
- [ ] Save fold models and OOF predictions for reproducible stacking.
