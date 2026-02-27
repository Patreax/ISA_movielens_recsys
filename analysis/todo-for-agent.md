# Todo List: Mini-Project 1 — Plan for Implementation

Use this as a sequential plan for an agent or for tracking progress. Do not code in this phase; this is the planning document. When implementing, follow the order below and mark items done.

---

## Phase 1: Environment and Data Loading

- [ ] **1.1** Create project structure (e.g. one Jupyter Notebook or a cookiecutter-style layout: `data/`, `notebooks/`, `src/` or similar). Decide: single notebook vs. notebook + scripts.
- [ ] **1.2** Add a data loader that reads `movielens-data/ml-100k/u.data` with correct separator (tab) and columns: user_id, item_id, rating, timestamp. Verify row count (100k) and no corrupt lines.
- [ ] **1.3** Add loaders for `u.user` (user_id, age, gender, occupation, zip) and `u.item` (movie_id, title, release_date, …, 19 genre columns). Use README for exact column order and delimiter (tab for u.data, `|` for u.user/u.item).
- [ ] **1.4** Merge ratings with user and item metadata into a single working dataset (e.g. pandas DataFrames) so each rating has user demographics and movie genres. Document column names and dtypes.

---

## Phase 2: Exploratory Data Analysis (EDA)

- [ ] **2.1** EDA — Ratings: distribution of ratings (1–5), distribution of ratings per user and per movie, sparsity (number of ratings vs. users×movies). Report summary statistics and optionally a small visualization.
- [ ] **2.2** EDA — Users and items: number of users (943), movies (1682), and ratings per user/item (min, max, mean). Identify any very inactive users/items if needed.
- [ ] **2.3** EDA — Demographics: distribution of age, gender, occupation; document how they will be used (cold start, similarity).
- [ ] **2.4** EDA — Genres: distribution of genres in u.item (how many movies per genre); distribution of genres in the rated set (e.g. genre frequency in rated movies). Document for the “genre preference” feature.
- [ ] **2.5** Write a short “quest establishment” summary: what task is being solved (e.g. “predict rating and recommend top-N for warm users via collaborative (SVD); for cold users, find similar users and recommend based on what they like”) and how the data supports it.

---

## Phase 3: Data Preprocessing for Modeling

- [ ] **3.1** Define a single function or module that outputs ratings in Surprise format: (user, item, rating) with consistent types (e.g. string for user/item so Surprise does not reorder). Support both full u.data and split files (e.g. u1.base, u1.test).
- [ ] **3.2** Implement or use existing train/test splits: either load u1.base/u1.test … u5.base/u5.test for 5-fold CV, or use Surprise’s built-in split. Document which split strategy is used.
- [ ] **3.3** Compute genre preference per user from the **training** set only: for each user, aggregate (e.g. average rating or count of high ratings) per genre using u.item genre columns. Store as a per-user vector or dict for re-ranking/filtering and for **similar-user finding** (user feature vector for K-means/KNN).
- [ ] **3.4** Define cold-start input: (age, gender, occupation, list of preferred genres). Document: cold start will use **similar-user strategy** — find similar users (e.g. via K-means cluster or KNN on user features), then recommend movies based on what those similar users like.

---

## Phase 4: SVD Model (Surprise)

- [ ] **4.1** Train Surprise SVD on the chosen train set (e.g. u1.base). Use default or reasonable hyperparameters first (e.g. n_factors=100, n_epochs=20, lr_all=0.005).
- [ ] **4.2** Evaluate on the corresponding test set: RMSE and MAE. Report in the notebook.
- [ ] **4.3** Run 5-fold cross-validation (u1–u5) and report mean and std of RMSE/MAE. Document as “baseline SVD” performance.
- [ ] **4.4** (Optional) Tune SVD: grid search or manual tuning over n_factors, n_epochs, lr_all, reg_all. Report best configuration and metrics.

---

## Phase 5: Recommendation Logic and Optional Enhancements

- [ ] **5.1** Implement top-N recommendation for warm users: from trained SVD, for each test user, get top-N unseen items and (optionally) re-rank or filter by genre preference. Document the logic.
- [ ] **5.2** Implement cold-start recommendation (hybrid / similar-user strategy): (1) Build user feature representation (demographics + genre preference). (2) Find similar users — e.g. **K-means** on user features, assign cold user to nearest cluster and use cluster members as similar users; or **KNN** / similarity on the same features. (3) Recommend movies based on what similar users like (e.g. aggregate their top-rated items or SVD top-N). Fallback: top popular in preferred genres. Document whether K-means or an alternative (KNN, cosine similarity) is used and why.
- [ ] **5.3** (Optional) Compare SVD-only vs. SVD + genre re-ranking on a few users or on ranking metrics; document whether re-ranking helps or hurts.

---

## Phase 6: Evaluation and Quality

- [ ] **6.1** Rating prediction quality: final RMSE and MAE for the chosen model and split; include in the notebook with clear captions.
- [ ] **6.2** (Optional) Ranking quality: define “relevant” (e.g. rating ≥ 4) and compute Precision@K, Recall@K, or NDCG for top-N recommendations if implemented. Use metrics from course materials (e.g. metrics.md).
- [ ] **6.3** Justify the solution: short text on why SVD was chosen, how demographics and genre are used, and how the pipeline is suitable for the dataset (alignment with assignment 1.1.C and 1.2.B).

---

## Phase 7: Documentation and Submission Readiness

- [ ] **7.1** Ensure the notebook (or project) runs from top to bottom without errors; document any required environment (Python version, Surprise, pandas, etc.).
- [ ] **7.2** Add clear section headers and narrative so that EDA (1.1.A), data preparation (1.1.B), justification (1.1.C), modeling (1.2.A), and evaluation (1.2.B) are easy to find.
- [ ] **7.3** Prepare submission: executable code + documentation (values, outputs, visualizations) as required by the assignment; ready for week 6 deadline.

---

## Optional / Future (for deployment in Mini-Project 3)

- [ ] **D.1** Refactor the data pipeline into a small module that can be called from an app (e.g. load ratings, build Surprise dataset, train SVD).
- [ ] **D.2** Expose a simple API or function: `recommend(user_id=None, cold_start_profile=None, top_n=10)` that returns list of movie IDs or titles.

---

**Note:** This is a plan only; no code was written in this step. When implementing, start from Phase 1 and proceed in order; dependencies (e.g. 3.3 depends on 3.2) are implied by the order.
