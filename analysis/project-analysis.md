# Mini-Project 1: MovieLens Recommendation System — Project Analysis

## 1. Assignment Context

**Goal:** Develop the first of two recommendation-system mini-projects (classic vs. NN later). Deliverables:

- **1.1 EDA and data preprocessing (12 pts):** EDA, data preparation for modeling, iterative justification w.r.t. the dataset.
- **1.2 Modeling and evaluation (8 pts):** RecSys modeling aligned with the quest, quality evaluation.

**Constraints:** Python; executable code (e.g. Jupyter Notebook or cookiecutter-style project); documentation with computed values, outputs, visualizations; due **week 6**.

---

## 2. Dataset Summary (MovieLens 100K)

Location: `movielens-data/ml-100k/`. Tab-separated (not CSV); encoding and delimiters must be handled in the pipeline.

| File       | Description | Format (from README) |
|-----------|-------------|----------------------|
| **u.data** | Ratings     | `user_id \t item_id \t rating \t timestamp` — 100,000 ratings (1–5), 943 users, 1,682 movies; each user ≥20 ratings. |
| **u.user** | Demographics| `user_id \t age \t gender \t occupation \t zip` — age, gender (M/F), occupation, zip. |
| **u.item** | Movies      | `movie_id \t title \t release_date \t video_release \t IMDb_URL \t unknown \t [19 genre flags]` — 19 binary genre columns (e.g. Action, Comedy, Drama). |
| **u.genre** | Genre names | One genre per line (indexed by column in u.item). |
| **u.occupation** | Occupations | One occupation per line. |
| **u1.base / u1.test … u5.base / u5.test** | Splits | 80%/20% splits for 5-fold cross-validation. |
| **ua.base, ua.test, ub.base, ub.test** | Alternative splits | 10 ratings per user in test set. |

**Relevance for your plan:** Demographics (age, gender, occupation) and genre (from u.item) are directly available for a hybrid/cold-start design (SVD + demographics + genre preference).

---

## 3. Chosen Approach: Hybrid — Collaborative Recommendation + Cold-Start via Similar Users

### 3.1 Core idea

- **Primary model:** **Collaborative recommendation** via Matrix Factorization (Surprise SVD) on the user–item–rating matrix for users with sufficient history.
- **Cold-start mitigation:** For new or sparse users, we use a **similar-user strategy**: find one or more users similar to the current user, then recommend movies based on what those similar users like (e.g. their top-rated or SVD-recommended items). This avoids recommending blindly and keeps the logic aligned with collaborative filtering.
- **Enrichment:** Demographics (u.user) and **genre preference** (derived from ratings × movie genres) support both the similar-user search and re-ranking/filtering.

### 3.2 How the pieces fit together

1. **Collaborative recommendation (SVD)**  
   - Input: (user_id, item_id, rating).  
   - Output: predicted ratings; top-N recommendations per user.  
   - Surprise expects a specific format (e.g. user, item, rating); you need a **data pipeline** that loads `u.data` (and optionally the splits) into that format.  
   - Used for **warm users** (users with enough ratings).

2. **Cold start: similar-user strategy**  
   - **Goal:** For a cold user (no or very few ratings), find **similar users** in the existing user base, then recommend movies based on what those similar users like (e.g. aggregate their high-rated items or their SVD top-N).  
   - **Input for cold user:** Can include demographics (age, gender, occupation), stated genre preferences, or both.  
   - **Output:** A ranked list of movies (e.g. most liked by similar users, or most frequently in similar users' top-N).

3. **Finding similar users (under consideration)**  
   - **Idea:** Represent each user by a feature vector (e.g. demographics, genre preference, or both), then find "nearest" users.  
   - **K-means:** One option is to **cluster users with K-means** (on demographics and/or genre preference). For a cold user, assign them to the nearest cluster and treat cluster members as similar users; recommend from the aggregate preferences of that cluster.  
   - **Uncertainty:** K-means is not yet fixed — alternatives include:  
     - **K-Nearest Neighbors (KNN)** on the same user features (explicit "similar users" without hard clusters).  
     - **Cosine similarity** (or other distance) on user preference vectors.  
   - **Practical suggestion:** Prototype with **K-means** first (simple, interpretable clusters); if cold-start quality is weak, try KNN or similarity-based selection and compare.

4. **Demographics (u.user)**  
   - Used to **define user similarity** for cold start (e.g. same or close age, same gender, same occupation).
   - Can be encoded into the user feature vector for K-means or KNN (with appropriate scaling/encoding for categoricals).

5. **Genre preference**  
   - **Derive:** For each user, from (user, item, rating) and (item, genres), compute a preference vector (e.g. average rating per genre, or count of high-rated movies per genre).
   - **Use:**  
     - **Similar-user features:** Include genre preference in the user representation for K-means/KNN so that "similar users" share both demographic and taste profile.
     - **Filter/Boost:** For warm users, re-rank or filter SVD top-N by genre preference.
     - **Cold start:** New user states preferred genres → can be used in the similarity step (e.g. match to users who also prefer those genres) and to filter recommended movies.

### 3.3 End-to-end flow (high level)

1. **Load & merge:** u.data, u.user, u.item → unified tables (users, items, ratings, user_demographics, item_genres).  
2. **EDA:** Distributions (ratings, users, items), sparsity, demographics, genre usage; document in notebook.  
3. **Preprocessing:**  
   - Map ids to Surprise-compatible format; handle missing/duplicates.  
   - Build genre preference per user from training data.  
   - Build user feature representation for similar-user finding (demographics + genre preference; suitable for K-means or KNN).  
4. **Train/validate:** Use built-in splits (e.g. u1.base/u1.test … u5.base/u5.test) or Surprise’s split; train SVD; tune (e.g. n_factors, epochs, lr).  
5. **Similar-user model (cold start):** Fit K-means (or alternative) on user features; for a cold user, identify similar users (same cluster or K-nearest) and aggregate their liked/recommended movies.  
6. **Evaluate:** RMSE/MAE (rating prediction); Precision@K / Recall@K or NDCG (ranking) if you define “relevant” (e.g. rating ≥ 4).  
7. **Recommendation logic:**  
   - **Warm user:** Collaborative (SVD) top-N → (optional) re-rank/filter by genre preference.  
   - **Cold user:** Find similar users (e.g. via K-means cluster or KNN) → recommend movies based on what similar users like (e.g. top-rated or top-N from their SVD).  
8. **Document:** Results, metrics, short justification of choices (alignment with assignment 1.1.C and 1.2.B), and note on K-means vs. alternatives if experimented.

---

## 4. What You Need to Do (Checklist-Oriented)

- **EDA:** Ratings distribution, user/item activity, sparsity matrix (if feasible), demographics (age, gender, occupation), genre distribution in u.item and in rated movies.  
- **Data pipeline:**  
  - Load u.data, u.user, u.item with correct separators and encodings.  
  - Join so that every rating has user and item metadata.  
  - Export to Surprise format (and optionally keep pandas for EDA/hybrid).  
- **Preprocessing:** Train/test split (use existing or Surprise); derive genre preference per user; build user feature representation for similar-user finding (demographics + genre preference); handle cold-start path.  
- **Modeling:** SVD in Surprise for warm users; for cold start: find similar users (e.g. K-means on user features or KNN) and recommend from what similar users like; optional: genre-based re-ranking/filtering.  
- **Evaluation:** RMSE/MAE; optionally Precision@K, Recall@K, NDCG (with a relevance threshold).  
- **Documentation:** Clear narrative, figures, and metric tables in the notebook/report.

---

## 5. Work Division (2 Teammates)

Suggested split so both contribute to EDA, pipeline, and modeling, with clear ownership.

| Area | Person A | Person B |
|------|----------|----------|
| **EDA** | Ratings & sparsity; user/item activity; basic stats. | Demographics (age, gender, occupation); genre distribution and usage. |
| **Data pipeline** | Load u.data, u.user, u.item; merge; Surprise Dataset format; train/test split. | Genre preference computation (per-user); cold-start input format (demographics + genre). |
| **Modeling** | SVD training and hyperparameter tuning (Surprise); evaluation (RMSE/MAE). | Cold-start: similar-user finding (e.g. K-means or KNN on user features), recommend from similar users' likes; re-ranking/filtering by genre. |
| **Evaluation & doc** | Metric tables; rating prediction analysis; reproducibility. | Recommendation quality (e.g. Precision@K/Recall@K if implemented); narrative and justification. |

**Sync points:** Agree on (1) Surprise data format and column names, (2) definition of “relevant” for ranking metrics, (3) cold-start API (inputs: age, gender, occupation, preferred genres). One person can own the notebook structure and the other the helper modules (e.g. `data_loader.py`, `metrics.py`) if you prefer a small package structure.

---

## 6. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Surprise expects different column names / format | Define a single “load_ratings()” that returns (user, item, rating) and use it everywhere. |
| Cold start is under-specified | Implement similar-user strategy: find similar users (K-means cluster or KNN), recommend from their top-rated or SVD top-N; fallback: top popular in preferred genres. |
| Genre re-ranking hurts performance | Compare SVD-only vs. SVD + genre re-rank in a small experiment; document. |
| K-means may not be best for similar-user finding | Prototype with K-means first; if cold-start quality is weak, try KNN or cosine similarity on user features and compare; document the choice. |
| Deadline (week 6) | Prioritize: (1) pipeline + SVD + RMSE/MAE, (2) EDA and justification, (3) genre/demographics as enhancement. |

---

## 7. Outcome for Mini-Project 1

- **Deliverable:** One Jupyter Notebook (or cookiecutter project) containing EDA, data prep, SVD training/evaluation, cold-start via similar users (e.g. K-means/KNN), and (optional) demographics + genre preference.  
- **Best model:** Either the best-tuned SVD or the best variant of SVD + re-ranking/filtering; cold-start path via similar-user recommendations; documented with metrics.  
- **Data pipeline:** Reusable load and preprocessing that can later be reused for deployment (Mini-Project 3).

This analysis should be enough to start implementation and to divide tasks between the two of you. The todo list in `todo-for-agent.md` breaks the same into actionable steps for an agent or for sequential work.
