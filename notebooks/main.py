import pandas as pd
import warnings

# Importing your CS3201 syllabus-aligned modules
from preprocessing import load_data, compute_rfm
from segmentation import segment_customers
from recommender import build_knn_recommender
from predictive_models import train_ensemble_model, predict_purchase_probability

warnings.filterwarnings("ignore")

def main():
    print("🚀 Initializing ML Pipeline (CS3201 Edition)...\n")
    print("Please wait while the models are trained (this will take a moment).")
    
    # 1. Load Data
    df = load_data("sample_ecommerce.csv")

    # 2. Segment Customers (PCA, K-Means, DBSCAN)
    rfm = compute_rfm(df)
    rfm_seg, km, pca, scaler = segment_customers(rfm)

    # 3. Build Recommendation Engine (KNN)
    engine = build_knn_recommender(df)

    # 4. Train Predictive Models (Stacking Ensemble)
    stack, sessions = train_ensemble_model(df)

    print("\n" + "="*60)
    print("✅ PIPELINE READY! The system is now interactive.")
    print("="*60 + "\n")

    def analyze_customer(target_user_id: int):
        """Generates the three required outputs for a specific customer."""
        print(f"\n{'='*50}\n 👤 CUSTOMER REPORT: {target_user_id}\n{'='*50}")

        # --- A. GROUPING (SEGMENTATION) ---
        if target_user_id in rfm_seg["user_id"].values:
            user_rfm = rfm_seg[rfm_seg["user_id"] == target_user_id].iloc[0]
            anomaly = "⚠️ (Flagged as Outlier by DBSCAN)" if user_rfm["dbscan_outlier"] == -1 else ""
            print(f"📊 [1] CUSTOMER GROUP: {user_rfm['segment']} {anomaly}")
            print(f"    • Recency   : {user_rfm['recency']} days")
            print(f"    • Frequency : {user_rfm['frequency']} purchases")
            print(f"    • Spent     : ${user_rfm['monetary']:.2f}")
        else:
            print("📊 [1] CUSTOMER GROUP: Unknown (No purchase history yet)")

        # --- B. PRODUCT SUGGESTIONS (KNN) ---
        recs = engine["recommend"](target_user_id, top_n=5)
        print(f"\n🛍️  [2] SUGGESTED PRODUCTS (KNN Similarity):")
        for i, pid in enumerate(recs, 1):
            print(f"    {i}. Product ID: {pid}")

        # --- C. LIKELINESS TO BUY (PREDICTION) ---
        user_sessions = sessions[sessions["user_id"] == target_user_id]
        if not user_sessions.empty:
            latest = user_sessions.iloc[-1]
            pred, prob = predict_purchase_probability(
                stack, latest["n_views"], latest["n_carts"], latest["hour"], latest["total_price"]
            )
            print(f"\n🔮 [3] LIKELINESS TO BUY (Stacking Ensemble):")
            print(f"    • Latest Activity: {int(latest['n_views'])} views, {int(latest['n_carts'])} carts")
            print(f"    • Probability    : {prob:.1%} chance of making a purchase")
            print(f"    • System Verdict : {'WILL BUY 🟢' if pred == 1 else 'WILL NOT BUY 🔴'}")
        else:
            print("\n🔮 [3] LIKELINESS TO BUY: Not enough session data to predict.")
        print("="*50 + "\n")

    # ─── THE INTERACTIVE LOOP ───
    # This loop keeps asking for a customer ID every time it finishes.
    while True:
        # Prompt the user to type an ID
        user_input = input("Enter a User ID to analyze (or type 'quit' to exit): ").strip()
        
        # Check if the user wants to close the program
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting the program. Goodbye! 👋")
            break
            
        # Ensure they typed a valid number
        try:
            target_user_id = int(user_input)
            # Run the analysis for the chosen ID
            analyze_customer(target_user_id)
        except ValueError:
            print("⚠️ Invalid input. Please enter a numeric User ID (e.g., 520088904).")

if __name__ == "__main__":
    main()