# ============================================
# E-commerce Behaviour Analysis
# Basic Data Analysis + Visualization
# ============================================

# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Dataset (SAFE for large data)
file_path = "./data/2019-Nov.csv"

try:
    df = pd.read_csv(file_path, nrows=100000)
    print("✅ Dataset loaded successfully!\n")
except FileNotFoundError:
    print("❌ File not found. Check the path.")
    import sys
    sys.exit()

# 3. Basic Info
print("="*50)
print("🔹 DATA OVERVIEW")
print("="*50)

print(df.head(), "\n")
df.info()
print("\nStatistical Summary:")
print(df.describe(percentiles=[0.1,0.25,0.5,0.75,0.9]))
print("\nColumns:", df.columns.tolist())

# 4. Missing Values
print("\n🔹 Missing Values:")
print(df.isnull().sum())

df.fillna(method='ffill', inplace=True)

# ============================================
# 🔥 DATA PREPROCESSING (IMPORTANT ADDITION)
# ============================================

# Convert time column
if 'event_time' in df.columns:
    df['event_time'] = pd.to_datetime(df['event_time'])
    df['hour'] = df['event_time'].dt.hour

# Remove invalid prices
if 'price' in df.columns:
    df = df[df['price'] > 0]

# ============================================
# 📊 VISUALIZATION SECTION
# ============================================

# Event Type Distribution
if 'event_type' in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x='event_type', data=df)
    plt.title("User Actions Distribution")
    plt.show()

# Price Distribution (Histogram + KDE)
if 'price' in df.columns:
    plt.figure(figsize=(8,5))
    sns.histplot(df['price'], bins=50, kde=True)
    plt.title("Price Distribution")
    plt.show()

# Top Categories
if 'category_id' in df.columns:
    top_categories = df['category_id'].value_counts().head(10)
    plt.figure(figsize=(8,5))
    sns.barplot(x=top_categories.values, y=top_categories.index)
    plt.title("Top 10 Categories")
    plt.show()

# User Activity
if 'user_id' in df.columns:
    user_activity = df['user_id'].value_counts().head(10)
    plt.figure(figsize=(8,5))
    sns.barplot(x=user_activity.values, y=user_activity.index)
    plt.title("Top 10 Active Users")
    plt.show()

# Hourly Activity (Advanced Insight)
if 'hour' in df.columns:
    hourly = df['hour'].value_counts().sort_index()
    plt.figure(figsize=(8,5))
    sns.lineplot(x=hourly.index, y=hourly.values)
    plt.title("User Activity by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Activity Count")
    plt.show()

# Correlation Heatmap
numeric_df = df.select_dtypes(include=np.number)

if not numeric_df.empty:
    plt.figure(figsize=(10,7))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

# ============================================
# 📊 BASIC ANALYSIS
# ============================================

if 'event_type' in df.columns:
    print("\n🔹 Event Type Distribution:")
    print(df['event_type'].value_counts())

if 'price' in df.columns:
    print("\n🔹 Price Stats:")
    print("Mean:", df['price'].mean())
    print("Max:", df['price'].max())
    print("Min:", df['price'].min())

if 'user_id' in df.columns:
    print("\n🔹 Unique Users:", df['user_id'].nunique())

# ============================================
# 📊 GROUPED ANALYSIS
# ============================================

if 'user_id' in df.columns and 'price' in df.columns:
    user_spending = df.groupby('user_id')['price'].sum()
    print("\n🔹 Top 5 Users by Spending:")
    print(user_spending.sort_values(ascending=False).head())

if 'category_id' in df.columns:
    print("\n🔹 Top Categories:")
    print(df['category_id'].value_counts().head())

# ============================================
# 🔥 EXTRA INSIGHTS (IMPORTANT)
# ============================================

if 'event_type' in df.columns:
    purchase_rate = (df['event_type'] == 'purchase').mean() * 100
    print(f"\n🔹 Conversion Rate: {purchase_rate:.2f}%")

if 'user_id' in df.columns and 'price' in df.columns:
    avg_spending = df.groupby('user_id')['price'].sum().mean()
    print(f"\n🔹 Avg Spending per User: {avg_spending:.2f}")

# ============================================
# 🧠 NUMPY DEMO
# ============================================

arr = np.array([10, 20, 30, 40, 50])
print("\n🔹 NumPy Demo:")
print("Mean:", np.mean(arr))
print("Std:", np.std(arr))

# ============================================
# 💾 SAVE SAMPLE
# ============================================

df.to_csv("../data/sample_ecommerce.csv", index=False)


print("\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")