# ISA Project 1 — MovieLens Recommender System

A recommender system built on the MovieLens 1M dataset for the ISA course.

## Getting started

**Option 1 — uv (recommended)**
```bash
uv sync
```

**Option 2 — pip**
```bash
pip install -r requirements.txt
```

## Dataset

The MovieLens 1M dataset is already included locally at `data/raw/ml-1m/` — no download needed.

## Usage

Everything important (EDA, model training, evaluation, results) is in the notebook:

```
notebooks/01_movielens_recsys.ipynb
```

Open it with Jupyter and run all cells from top to bottom.

## Project structure

```
├── data/raw/ml-1m/     <- MovieLens 1M dataset (included)
├── notebooks/          <- Main notebook with all the work
├── project1/           <- Source code (dataset loading, features, training, prediction)
└── models/             <- Saved trained models
```
