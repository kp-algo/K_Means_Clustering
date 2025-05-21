import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --- Load Data ---
df = pd.read_csv("clustered_customers.csv")

# --- Feature Groups ---
demographic_features = ['Age', 'Customer_Tenure' ,'Income', 'Kidhome', 'Teenhome', 'Recency']
products_features = ['Wines_ratio','Fruits_ratio','MeatProducts_ratio','Fish_ratio','SweetProducts_ratio','GoldProds_ratio', 'Total_Spending']
place_features = ['Total_Purchases', 'WebPurchases_ratio', 'CatalogPurchases_ratio', 'StorePurchases_ratio']
promotion_features = ['AcceptedCmp3' ,'AcceptedCmp4' ,'AcceptedCmp5' ,"AcceptedCmp1" ,'AcceptedCmp2']

# --- Page Setup ---
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation Dashboard")
st.markdown("Use the controls on the sidebar to explore customer clusters dynamically.")

# --- Sidebar Controls ---
st.sidebar.header("Clustering Controls")
st.sidebar.markdown("Select features from each category:")

selected_demographic_features = st.sidebar.multiselect("Demographic Features", demographic_features, default=['Age', 'Income'])
selected_products_features = st.sidebar.multiselect("Product Features", products_features, default=['Wines_ratio', 'Total_Spending'])
selected_place_features = st.sidebar.multiselect("Place Features", place_features, default=['Total_Purchases'])
selected_promotion_features = st.sidebar.multiselect("Promotion Features", promotion_features, default=['AcceptedCmp1', 'AcceptedCmp2'])



selected_features = selected_demographic_features + selected_products_features + selected_place_features + selected_promotion_features

# --- Clustering Logic ---
if len(selected_features) >= 3:
    st.markdown("Clustering Results")
    
    X = df[selected_features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 4)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df.loc[X.index, 'Cluster'] = cluster_labels

    sil_score = silhouette_score(X_scaled, cluster_labels)
    st.success(f"**Silhouette Score:** {sil_score:.3f} _(Higher is better, max=1)_")

    # --- Cluster Size Chart ---
    st.markdown("Cluster Sizes")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    fig1, ax1 = plt.subplots()
    colors = [matplotlib.colors.to_hex(c) for c in plt.cm.Set1.colors[:n_clusters]]
    cluster_counts.plot(kind='bar', ax=ax1, color=colors)
    ax1.set_xlabel("Cluster")
    ax1.set_ylabel("Number of Customers")
    ax1.set_title("Cluster Sizes")
    st.pyplot(fig1)

    st.markdown("---")

    # --- Cluster Profile Table ---
    st.markdown("Cluster Profile Summary")
    cluster_summary = df.groupby('Cluster')[selected_features].mean()
    st.dataframe(cluster_summary.style.background_gradient(cmap='Blues', axis=1))

    st.markdown("---")

    # --- PCA Plot ---
    st.markdown("PCA: 2D Customer Segments")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='Set1', alpha=0.7)
    ax2.set_xlabel('PCA Component 1')
    ax2.set_ylabel('PCA Component 2')
    ax2.set_title('Customer Segments (PCA Reduced)')
    legend1 = ax2.legend(*scatter.legend_elements(), title="Clusters")
    ax2.add_artist(legend1)
    st.pyplot(fig2)

    st.markdown("---")

    # --- Download CSV ---
    st.markdown("Download Results")
    csv = df.to_csv(index=False).encode()
    st.download_button(
        label="Download Clustered Data",
        data=csv,
        file_name='clustered_customers_labeled.csv',
        mime='text/csv'
    )

else:
    st.warning("Please select at least **three features** for clustering.")
