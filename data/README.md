# Dataset Documentation

This directory contains local copies of the datasets used by Steam Trends.

## Required Datasets

### 1. Steam Games Dataset

Contains game metadata, pricing, release information, review statistics, estimated ownership, and playtime.

- Source: FronkonGames, distributed through Kaggle and/or Hugging Face.
- License: Verify the license for the exact version downloaded; the proposed source identifies CC BY 4.0.

### 2. Steam Reviews 2024

Contains player-written Steam reviews for sentiment analysis and gameplay-topic extraction.

- Source: Kaggle.
- License and usage conditions: Verify the dataset's accompanying documentation.

## Setup

Download the datasets from their original sources and place the required files in this directory.

Do not commit large raw datasets to Git. Record the download date, source URL, version, and any preprocessing steps here.

## Data Processing

The datasets will be joined using Steam App IDs. Cleaning, filtering, feature engineering, and target construction will be documented in the project's notebooks and source code.
