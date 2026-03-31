# 🎬 OTT Platform User Behavior Analyzer

> Predict churn, forecast watch time, and discover viewer personas using machine learning on Netflix content data.

---

## 📌 Overview

This project analyzes user behavior on an OTT platform (modeled after Netflix) and applies a full machine learning pipeline across three core tasks:

| Task | Type | Goal |
|------|------|------|
| Churn Prediction | Classification | Will this user leave the platform? |
| Watch Time Prediction | Regression | How many hours will this user watch? |
| Viewer Segmentation | Clustering | What kind of viewer is this user? |

Since the source dataset is content-based (titles, genres, ratings), **user behavior is realistically simulated** from it — a standard and accepted technique in data science when user-level data is unavailable.

---

## 📁 Project Structure

```
ott-behavior-analyzer/
│
├── OTT_Platform_User_Behavior_Analyzer.ipynb   # Main notebook (all code)
├── netflix_titles.csv                           # Dataset (download separately)
└── README.md
```

---

## 📊 Dataset

**Source:** [Netflix Movies and TV Shows — Kaggle (shivamb)](https://www.kaggle.com/datasets/shivamb/netflix-shows)

Download `netflix_titles.csv` and place it in the project root before running the notebook.

> ⚠️ If the file is not found, the notebook automatically generates a 500-row demo dataset so all cells still execute.

### Simulated User Features

Since the Kaggle dataset only contains content metadata, the notebook generates a **5,000-user behavioral dataset** with the following features:

| Feature | Description |
|---|---|
| `watch_time` | Total hours watched per month |
| `number_of_sessions` | Login sessions per month |
| `preferred_genre` | Most-watched genre |
| `last_login_days` | Days since last login |
| `subscription_type` | Basic / Standard / Premium |
| `rating_given` | Average rating given (1–5) |
| `tenure_months` | Months as a subscriber |
| `num_profiles` | Number of profiles on account |
| `device_type` | Mobile / TV / Laptop / Tablet |
| `churn` | Target label: 0 = Active, 1 = Churned |

### Engineered Features

| Feature | Description |
|---|---|
| `watch_per_session` | Average hours per session |
| `engagement_score` | Composite score (watch time + sessions + recency + rating) |
| `is_inactive` | Flag: last login > 20 days |
| `is_power_user` | Flag: high watch time AND high sessions |
| `genre_popularity_score` | How popular the user's preferred genre is in the catalog |
| `loyalty_tier` | New / Regular / Loyal based on tenure |

---

## 🧠 ML Tasks

### ✅ Task 1 — Classification: Churn Prediction

**Target:** `churn` (0 = Active, 1 = Churned)

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear model |
| K-Nearest Neighbors | K=7 |
| Support Vector Machine | RBF kernel |
| Decision Tree | max_depth=6 |
| Random Forest | 100 estimators |

**Metrics:** Accuracy · Precision · Recall · F1 Score · Confusion Matrix

---

### ✅ Task 2 — Regression: Watch Time Prediction

**Target:** `watch_time` (hours/month)

| Model | Notes |
|---|---|
| Linear Regression | Baseline |
| Decision Tree Regressor | max_depth=8 |
| Random Forest Regressor | 100 estimators |

**Metrics:** MAE · RMSE · R² Score

---

### ✅ Task 3 — Clustering: Viewer Personas

**Algorithm:** K-Means + Agglomerative Hierarchical Clustering

| Persona | Profile |
|---|---|
| 🍿 Binge Watcher | High watch time, very frequent sessions |
| 😌 Casual Viewer | Low watch time, infrequent sessions |
| 📅 Regular Viewer | Moderate usage, consistent engagement |
| 😴 Lapsed Viewer | High inactivity, churn risk |

**Metrics:** Silhouette Score · Elbow Method · Dendrogram

---

### ⭐ Bonus — Content-Based Recommendation System

Uses **TF-IDF vectorization** on genre, type, rating, and description.  
Computes **cosine similarity** between titles to recommend content based on user's preferred genre.

---

## ⚙️ Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ott-behavior-analyzer.git
cd ott-behavior-analyzer
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

### 3. Add the dataset

Download `netflix_titles.csv` from [Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows) and place it in the project root.

### 4. Launch the notebook

```bash
jupyter notebook OTT_Platform_User_Behavior_Analyzer.ipynb
```

Run all cells top to bottom (`Kernel → Restart & Run All`).

---

## 📈 Sample Outputs

- 📊 Churn rate breakdown by subscription type
- 🔥 Feature importance chart (Random Forest)
- 📉 Actual vs Predicted scatter plot (regression)
- 🗺️ PCA 2D cluster visualization
- 🌳 Hierarchical clustering dendrogram
- 🎬 Personalized content recommendations per user


---

## 📜 License

This project is for educational purposes. Dataset credit: [Shivam Bansal on Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows).
