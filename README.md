# Steam Trends

**What Makes a Steam Game Successful?**

An exploratory data analytics project investigating Steam game success through store metadata, player reviews, and predictive modelling.

## Overview

Steam hosts tens of thousands of games, yet their performance varies significantly. Developers must make decisions about pricing, genre, platforms, and release timing without knowing which factors will contribute most to a game's success.

This project investigates how much of a game's success can be predicted using information available at launch, and whether early player feedback provides additional predictive value.

Success will be examined across three dimensions:

- **Reception:** Percentage of positive user reviews.
- **Reach:** Estimated number of game owners.
- **Engagement:** Median player playtime.

The objective is to identify meaningful patterns in game performance, evaluate predictive models, and determine which factors are most strongly associated with each outcome.

## Research Questions

1. How accurately can game success be predicted using launch metadata alone?
2. Which characteristics, including price, genre, release timing, and platform support, are associated with better outcomes?
3. Do early player reviews improve predictions beyond launch metadata?
4. Which gameplay-related themes, such as bugs, performance, story, difficulty, and value for money, are associated with reception and engagement?

## Data Sources

The project will combine two publicly available datasets:

- **Steam Games Dataset:** Game metadata, release information, pricing, review statistics, estimated ownership, and playtime.
- **Steam Reviews 2024:** Player-written reviews used to derive sentiment and gameplay-related features.

The datasets will be joined using Steam App IDs. Exact dataset versions, source links, and applicable licenses will be documented during data acquisition.

## Methodology

### 1. Data Preparation

Clean and integrate the datasets, remove duplicates and non-games, handle missing values, parse release dates, and establish inclusion criteria for reliable analysis.

### 2. Exploratory Data Analysis

Investigate how reception, reach, and engagement vary by genre, price, release year, and other game characteristics. Identify games with high estimated ownership but comparatively low playtime.

### 3. Feature Engineering

Develop predictive features from two sources:

- **Launch metadata:** Price, genres, developer-set categories, release timing, supported platforms and languages, achievements, DLC, and publisher experience.
- **Player feedback:** Sentiment scores, common terms, and topic proportions derived from early review text.

Player-assigned tags will be excluded from launch-only features because they reflect information gathered after release. Review-derived features will use an early review window to reduce information leakage.

### 4. Predictive Modelling

Compare two approaches for each of the three success measures:

- **Model A:** Launch metadata only.
- **Model B:** Launch metadata combined with early review-derived features.

Candidate methods include linear and logistic regression, random forests, and other appropriate baseline models. The final techniques will depend on the target variable and data characteristics.

### 5. Evaluation

Use train/test splits and cross-validation to evaluate predictive performance.

Depending on the target formulation, evaluation will use appropriate regression metrics or classification metrics, including accuracy, macro F1, and confusion matrices where applicable.

The difference between Model A and Model B will help measure the additional predictive value of early player feedback.

### 6. Interpretation

Use feature importance and SHAP values, where appropriate, to identify influential launch characteristics and gameplay-related themes. Translate the results into practical insights for game developers.

## Repository Structure

```text
steam-trends/
├── data/
│   └── README.md
├── notebooks/
├── src/
├── results/
├── README.md
├── requirements.txt
└── .gitignore
```

| Path | Purpose |
|---|---|
| `data/` | Dataset documentation and local data files |
| `notebooks/` | Exploration, visualizations, and experiments |
| `src/` | Reusable processing and modelling code |
| `results/` | Figures, tables, and evaluation outputs |
| `requirements.txt` | Python dependencies |

Large datasets and generated results will generally remain outside version control.

## Getting Started

### Requirements

- Python 3.11 or another compatible Python version
- Git
- Jupyter Notebook or JupyterLab
- The required datasets downloaded separately

### Installation

Clone the repository:

```bash
git clone <repository-url>
cd steam-trends
```

Create and activate a virtual environment.

**Windows (PowerShell):**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

## Data Considerations

- Steam owner counts are estimates rather than exact sales figures.
- Games with too few reviews may be excluded to improve target reliability.
- Player-assigned tags will not be used as launch-time predictors.
- Review-derived features will use early reviews to reduce information leakage.
- Steam does not publish complete refund data, so refund behaviour cannot be measured directly.
- Playtime filtering and target definitions will be finalized before modelling to avoid excluding relevant engagement patterns.

## Team

- Yasin (Dani) Erfani Pour
- Evan Dos Santos

## Project Status

**In progress:** Project setup and data acquisition.

Planned stages include data cleaning, exploratory analysis, feature engineering, predictive modelling, evaluation, and interpretation.

