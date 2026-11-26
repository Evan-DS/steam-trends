# What Makes a Steam Game Successful?

ECE 9603/9063 Data Analytics Foundations project by Evan Dos Santos and Yasin Erfani Pour.

We predict three measures of a Steam game's success (reception, reach and engagement) from 42 launch features (Model A), then test whether the first 30 days of player reviews improve the predictions (Model B).

## Quick start

1. Install Python 3.10 or newer, then the libraries (adjust requirements.txt if regenerating data):

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Review the two source data files for updates in `data/source/`:

   | File | What it is | Size |
   |---|---|---|
   | `games.parquet` | the full Steam catalogue (Steam Games Dataset), without unused text columns |
   | `review_features.parquet` | early-review features of the 2,000-game Model B sample | ~80 KB |

   See Kaggle and Huggling Face links below.

3. Run it:

   ```bash
   python main.py
   ```

It prints its progress with timestamps as it goes and takes about 35 minutes; the tuning step is most of that. When it finishes, open **`output/report.html`** in a browser: it holds every results table and figure on one page.

No internet connection or API key is needed. The same two source files always give the same results.

## What it does

| # | Step | Makes (in `data/derived/`, figures in `output/figures/`) |
|---|---|---|
| 1 | Clean: filter to 33,374 games and build the three success measures | `games_clean.parquet`, `population_funnel.csv` |
| 2 | Build the 42 launch features | `launch_features.parquet` |
| 3 | Model A: baseline vs XGBoost, 5-fold cross-validation, robustness check | `model_a_summary.csv`, `model_a_results.csv`, … |
| 4 | Model B: Model A vs Model B on the same games and 15 folds | `model_b_comparison.csv`, `model_b_fold_scores.csv` |
| 5 | SHAP explanations and the high-reach, low-engagement games | `shap_importance.csv`, … |
| 6 | Hyperparameter tuning and learning curves | `holdout_comparison.csv`, `learning_curves.csv`, … |
| 7 | The results report | `output/report.html` |

Options (none are needed):

```bash
python main.py --slim-games    # shrink an older, full-size games.parquet (see below)
python main.py --list          # show the steps
python main.py --steps 3       # run one step (the earlier steps' files must exist)
python main.py --steps 3-5     # run a range of steps
pytest                         # run the unit tests
```

## Project structure

```
├── main.py                   run this
├── requirements.txt
├── data/
│   ├── source/               the two input files*
│   └── derived/              everything the program makes from them
├── output/                   report.html and figures/
├── src/steam_success/
│   ├── pipeline.py           runs the steps in order and prints progress
│   ├── config.py             folders, random seed, model settings
│   ├── data.py · modeling.py · plotting.py · printing.py · progress.py · kaggle_client.py
│   └── steps/                clean, features, model_a, model_b, explain, tuning, report,
│                             and load_and_join (only used with --download)
└── tests/
```

`python main.py --download` saves the slim version automatically.

## Optional: regenerating the source data

The two source files were built from public datasets:

- **Steam Games Dataset** (Hugging Face): huggingface.co/datasets/FronkonGames/steam-games-dataset
- **Steam Games Reviews 2024** (Kaggle): kaggle.com/datasets/artermiloff/steam-games-reviews-2024

To rebuild them from those sources instead of using the provided files:

```bash
export KAGGLE_API_TOKEN=your_token     # Windows: set KAGGLE_API_TOKEN=your_token
export HF_TOKEN=your_token             # optional: the Hugging Face dataset is public
python main.py --download
```

If the Kaggle token isn't set, the program asks for it. This downloads the game catalogue and, one game at a time, the review files of the 2,000 sampled games, deleting each after reading it (a few GB of downloads in total, not the full 15 GB). It takes a few hours because of Kaggle's rate limits. The datasets are updated by their authors over time, so a fresh download can give slightly different numbers from the provided files.

Keep API tokens out of the code and out of git (`kaggle.json` and `.env` are ignored).