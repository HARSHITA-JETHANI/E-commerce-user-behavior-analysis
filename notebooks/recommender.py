"""
recommender.py — Module 3
==========================
Content-Based Recommendation Engine using K-Nearest Neighbors (KNN).
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors

def build_knn_recommender(df: pd.DataFrame) -> dict:
    """
    Builds a product feature space and uses KNN to find similar products.
    """
    # Create unique product profiles
    products = df.drop_duplicates('product_id')[['product_id', 'category_code', 'brand', 'price']].copy()
    products.fillna('unknown', inplace=True)
    products = products.reset_index(drop=True)

    # Encode categorical features
    le_cat = LabelEncoder()
    le_brand = LabelEncoder()
    products['cat_encoded'] = le_cat.fit_transform(products['category_code'])
    products['brand_encoded'] = le_brand.fit_transform(products['brand'])

    # Scale features
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(products[['cat_encoded', 'brand_encoded', 'price']])

    # Supervised/Instance Learning: KNN (From Syllabus)
    knn = NearestNeighbors(n_neighbors=6, metric='euclidean') # 6 because 1st nearest is the item itself
    knn.fit(feature_matrix)
    print(f"[✓] KNN Engine  : Trained on {len(products)} unique products")

    # Keep track of user's most recent interaction for contextual recommendations
    last_interaction = df.groupby('user_id').last()['product_id'].to_dict()
    popular_products = df['product_id'].value_counts().head(5).index.tolist()

    def recommend(user_id: int, top_n: int = 5) -> list:
        # If unknown user, return global popular items
        if user_id not in last_interaction:
            return popular_products[:top_n]
        
        # Get the last product the user viewed/bought
        target_product_id = last_interaction[user_id]
        
        try:
            # Find the product's index in our matrix
            prod_idx = products[products['product_id'] == target_product_id].index[0]
            
            # Use KNN to find the closest matching products
            distances, indices = knn.kneighbors([feature_matrix[prod_idx]], n_neighbors=top_n+1)
            
            # Skip the first index (which is the product itself)
            rec_indices = indices[0][1:] 
            return products.iloc[rec_indices]['product_id'].tolist()
        except IndexError:
            return popular_products[:top_n]

    return {"recommend": recommend}