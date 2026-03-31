"""
predictive_models.py — Module 4
================================
Predict purchase probability using an Ensemble of Syllabus Algorithms.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

FEATURES = ["n_views", "n_carts", "hour", "total_price"]

def engineer_session_features(df: pd.DataFrame) -> pd.DataFrame:
    # Exactly the same as the previous response (must include user_id)
    sessions = (
        df.groupby("user_session")
          .apply(lambda g: pd.Series({
              "user_id":       int(g["user_id"].iloc[0]),
              "n_views":       int((g["event_type"] == "view").sum()),
              "n_carts":       int((g["event_type"] == "cart").sum()),
              "hour":          int(g["hour"].mode().iloc[0]) if not g["hour"].dropna().empty else 0,
              "total_price":   round(float(g["price"].sum()), 2),
              "made_purchase": int((g["event_type"] == "purchase").any()),
          }), include_groups=False).reset_index()
    )
    return sessions

def train_ensemble_model(df: pd.DataFrame):
    """
    Trains LR, DT, SVM, and NB, then combines them via Stacking.
    """
    sessions = engineer_session_features(df)
    X = sessions[FEATURES]
    y = sessions["made_purchase"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Base Learners (All from Syllabus)
    base_estimators = [
        ('lr', LogisticRegression(class_weight='balanced', max_iter=200)),
        ('dt', DecisionTreeClassifier(max_depth=5, class_weight='balanced')),
        ('nb', GaussianNB()),
        ('svm', SVC(probability=True, class_weight='balanced', kernel='rbf')) 
    ]

    # Ensemble Method: Stacking (From Syllabus)
    # The Logistic Regression metaclassifier learns how to best combine the 4 base models
    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(),
        cv=3 
    )
    
    print("[...] Training Stacking Ensemble (LR, DT, NB, SVM)...")
    stack.fit(X_train, y_train)
    
    preds = stack.predict(X_test)
    print(f"[✓] Ensemble    : Trained successfully. Accuracy = {accuracy_score(y_test, preds):.2%}")
    
    return stack, sessions

def predict_purchase_probability(stack: StackingClassifier, n_views: int, n_carts: int, hour: int, total_price: float):
    X = pd.DataFrame([[n_views, n_carts, hour, total_price]], columns=FEATURES)
    pred = int(stack.predict(X)[0])
    prob = float(stack.predict_proba(X)[0][1])
    return pred, prob