import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

# Suppress a future warning from scikit-learn
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn.cluster._kmeans')

# --- Step 1: Load the original data ---
# We use the original training data because we need to see all behaviors, not just conversions.
df = pd.read_csv('data/train_data.csv')
print("--- Original data loaded ---")


# --- Step 2: Create Session-Level Features ---
# We need to aggregate the data so that each row represents one session.
session_df = df.groupby('session_id').agg(
    total_clicks=('order', 'max'),          # The highest order number is the total clicks
    avg_price=('price', 'mean'),            # Average price of items viewed in the session
    distinct_categories=('page1_main_category', 'nunique'), # Number of different categories visited
    sale_page_visited=('page1_main_category', lambda x: 1 if 4 in x.values else 0) # Did they visit the 'sale' page (ID 4)?
).reset_index()

print("--- Session-level features created ---")


# --- Step 3: Prepare Data for Clustering ---
# We'll cluster based on the behavioral features. We drop session_id as it's just an identifier.
features_for_clustering = session_df.drop('session_id', axis=1)

# K-Means is sensitive to the scale of data, so we must scale our features.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_for_clustering)
print("--- Features scaled ---")


# --- Step 4: Build and Train the K-Means Model ---
# We have to choose the number of clusters (k). Let's start with 4, assuming a few distinct shopper types.
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(scaled_features)
print(f"--- K-Means model trained with {k} clusters ---")


# --- Step 5: Analyze the Clusters ---
# Assign the cluster labels back to our session dataframe
session_df['cluster'] = kmeans.labels_

# Let's see how many sessions are in each cluster
print("\n--- Cluster Sizes ---")
print(session_df['cluster'].value_counts())

# Now, let's look at the average behavior for each cluster.
# We un-scale the cluster centers to interpret them in original units (like dollars and clicks).
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_analysis = pd.DataFrame(cluster_centers, columns=features_for_clustering.columns)

print("\n--- Cluster Analysis (Average behavior per cluster) ---")
print(cluster_analysis.round(2))