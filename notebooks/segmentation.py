"""
segmentation.py — Module 2
============================
Customer Segmentation using PCA, K-Means, and DBSCAN from the CS3201 Syllabus.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
import warnings

warnings.filterwarnings("ignore")

N_CLUSTERS = 4
RANDOM_STATE = 42

def segment_customers(rfm: pd.DataFrame) -> tuple[pd.DataFrame, KMeans, PCA, StandardScaler]:
    """
    Pipeline: Scale -> PCA -> K-Means (Main Segments) -> DBSCAN (Outliers)
    """
    scaler = StandardScaler()
    features = ["recency", "frequency", "monetary"]
    X_scaled = scaler.fit_transform(rfm[features].values.astype(float))

    # 1. Dimensionality Reduction: PCA (From Syllabus)
    # Compressing to 2 components for noise reduction
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    print(f"[✓] PCA Applied : Retained {pca.explained_variance_ratio_.sum():.1%} variance")

    # 2. Unsupervised Learning: K-Means (From Syllabus)
    km = KMeans(n_clusters=N_CLUSTERS, init="k-means++", random_state=RANDOM_STATE)
    rfm_seg = rfm.copy()
    rfm_seg["cluster"] = km.fit_predict(X_pca)

    # 3. Unsupervised Learning: DBSCAN (From Syllabus)
    # Used here strictly for anomaly detection (identifying massive outliers)
    db = DBSCAN(eps=0.5, min_samples=5)
    rfm_seg["dbscan_outlier"] = db.fit_predict(X_pca) # -1 means outlier

    # Label Clusters
    LABELS = {0: "VIP / Champion", 1: "Loyal Customer", 2: "Potential Loyalist", 3: "Churn Risk"}
    
    # Sort clusters by monetary value to assign labels logically
    stats = rfm_seg.groupby("cluster")["monetary"].mean().sort_values(ascending=False).index
    label_map = {cluster_id: LABELS[i] for i, cluster_id in enumerate(stats)}
    rfm_seg["segment"] = rfm_seg["cluster"].map(label_map)

    outlier_count = (rfm_seg["dbscan_outlier"] == -1).sum()
    print(f"[✓] K-Means     : Segmented into {N_CLUSTERS} groups")
    print(f"[✓] DBSCAN      : Flagged {outlier_count} unique outlier customers")

    return rfm_seg, km, pca, scaler