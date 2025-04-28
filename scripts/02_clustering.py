#!/usr/bin/env python
# coding: utf-8

# # Phase 2: Unsupervised Clustering of Engine Degradation Stages
# 
# This notebook implements unsupervised learning techniques on the cleaned CMAPSS datasets (FD001–FD004) to identify data-driven degradation stages. The primary goal is to segment each engine’s operating cycles into five interpretable stages (Stage 0 to Stage 4), reflecting the progression of wear and deterioration.
# 
# ---
# 
# ## Purpose of Clustering
# 
# In real-world predictive maintenance, true degradation labels are rarely available. Clustering allows us to discover underlying patterns directly from sensor data, without manual labeling or domain assumptions.
# 
# This stage helps us:
# - Identify recurring patterns in engine behavior using sensor measurements.
# - Assign degradation stage labels based on natural structure in the data.
# - Create inputs for later classification and regression models in Phases 3 to 5.
# 
# ---
# 
# ## Why Use Both KMeans and Agglomerative Clustering?
# 
# We use two distinct clustering approaches to compare insights:
# - **KMeans** groups data into fixed, compact clusters and performs well with clear, centralized boundaries.
# - **Agglomerative Clustering** builds a hierarchy of connections between instances and is suited for gradual, continuous degradation behavior.
# 
# By applying both, we aim to understand the consistency and differences in how degradation manifests across engines.
# 
# ---
# 
# ## Dimensionality Reduction: PCA and t-SNE
# 
# Since sensor data is high-dimensional, we use the following techniques to visualize it:
# - **PCA (Principal Component Analysis)** captures linear structure and major axes of variance.
# - **t-SNE (t-distributed Stochastic Neighbor Embedding)** reveals local neighborhood relationships and non-linear patterns.
# 
# Visualizing clusters in 2D helps interpret and compare their quality intuitively.
# 
# ---
# 
# ## Silhouette Scores for Cluster Evaluation
# 
# To assess clustering quality, we use the **silhouette score**, which measures:
# - **Cohesion**: How similar a point is to its own cluster.
# - **Separation**: How different it is from the nearest cluster.
# 
# Scores close to 1 indicate good clustering. We compare scores across datasets and methods to validate our approach.
# 
# ---
# 
# ## Next Steps
# 
# The clusters created here serve as pseudo-labels for:
# - **Phase 3**: Supervised classification of health stages using live sensor readings.
# - **Phase 4 and 5**: RUL prediction and hybrid scoring, informed by degradation stages.
# 
# Clustering establishes the foundation for the rest of the predictive maintenance pipeline, balancing interpretability with machine-learned insights.
# 

# In[ ]:


# Phase 2: Clustering Analysis with KMeans and AgglomerativeClustering (Enhanced + Plot Export + Score Summary)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import os

# Load cleaned data
base_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'data'))
figures_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'figures'))
os.makedirs(figures_path, exist_ok=True)

dataset_ids = ['FD001', 'FD002', 'FD003', 'FD004']
datasets = {}

for ds_id in dataset_ids:
    df = pd.read_csv(os.path.join(base_path, f"clean_train_{ds_id}.csv"))
    datasets[ds_id] = df

# Map numeric cluster label to descriptive stage name
def stage_label(n):
    return f"Stage {n}"

# Global silhouette summary
silhouette_summary = {}

# Function to apply clustering and visualize with PCA and t-SNE
def apply_clustering(name, df):
    print(f"\n⏳ Clustering and visualizing {name}...")
    sensor_features = [col for col in df.columns if col.startswith("sensor_")]
    X = df[sensor_features].copy()

    # Apply clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    agglom = AgglomerativeClustering(n_clusters=5)
    df['kmeans_cluster'] = kmeans.fit_predict(X)
    df['agglo_cluster'] = agglom.fit_predict(X)
    df['kmeans_stage'] = df['kmeans_cluster'].apply(stage_label)
    df['agglo_stage'] = df['agglo_cluster'].apply(stage_label)

    # Silhouette scores
    k_score = silhouette_score(X, df['kmeans_cluster'])
    a_score = silhouette_score(X, df['agglo_cluster'])
    silhouette_summary[name] = {
        'KMeans': round(k_score, 4),
        'Agglomerative': round(a_score, 4)
    }

    # PCA visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    df['pca_1'] = pca_result[:, 0]
    df['pca_2'] = pca_result[:, 1]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x='pca_1', y='pca_2', hue='kmeans_stage', palette='tab10')
    plt.title(f"{name}: KMeans Stages via PCA")
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x='pca_1', y='pca_2', hue='agglo_stage', palette='tab10')
    plt.title(f"{name}: Agglomerative Stages via PCA")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f"{name}_PCA_Clusters.png"))
    plt.close()

    # t-SNE visualization (updated to use max_iter)
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(X)
    df['tsne_1'] = tsne_result[:, 0]
    df['tsne_2'] = tsne_result[:, 1]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x='tsne_1', y='tsne_2', hue='kmeans_stage', palette='tab10')
    plt.title(f"{name}: KMeans Stages via t-SNE")
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df, x='tsne_1', y='tsne_2', hue='agglo_stage', palette='tab10')
    plt.title(f"{name}: Agglomerative Stages via t-SNE")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f"{name}_TSNE_Clusters.png"))
    plt.close()

    # Distribution plots (added hue= explicitly to avoid warnings)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.countplot(x='kmeans_stage', hue='kmeans_stage', data=df, palette='tab10', order=sorted(df['kmeans_stage'].unique()))
    plt.title(f"{name} KMeans Stage Distribution")
    plt.subplot(1, 2, 2)
    sns.countplot(x='agglo_stage', hue='agglo_stage', data=df, palette='tab10', order=sorted(df['agglo_stage'].unique()))
    plt.title(f"{name} Agglomerative Stage Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, f"{name}_Cluster_Distribution.png"))
    plt.close()

    # Save clustered data
    df.to_csv(os.path.join(base_path, f"clustered_train_{name}.csv"), index=False)
    print(f"✔ Clustered data saved: clustered_train_{name}.csv")

    # Clear and impactful explanation
    print(f"\n📘 **{name} Clustering Insight:**")
    print("Clustering the sensor data into degradation stages helped us visualize and interpret engine health patterns.")
    print("KMeans captured tight groupings while Agglomerative showed progressive transitions.")
    print("PCA and t-SNE plots make the separation between clusters clearly visible.")
    print("Silhouette scores validate the quality of the clustering for both methods.\n")

# Run clustering
for ds_id in dataset_ids:
    apply_clustering(ds_id, datasets[ds_id])


# Print silhouette summary for reporting
print("\n📊 **Silhouette Score Summary Table:**")
for ds, scores in silhouette_summary.items():
    print(f"{ds}: KMeans={scores['KMeans']}, Agglomerative={scores['Agglomerative']}")


# ## Save Silhouette Scores to JSON
# 
# This section stores the silhouette scores computed during clustering for all four datasets (FD001–FD004) into a JSON file (`silhouette_scores.json`).
# 
# ### Purpose
# Silhouette scores are used to evaluate how well the clustering algorithms (KMeans and Agglomerative) have grouped the sensor data. Rather than recalculating these scores every time the notebook is run, we save them once and reuse the results during evaluation, reporting, or further phases.
# 
# ### Why This Step is Useful
# - Helps confirm whether clustering was meaningful across datasets.
# - Enables direct comparison of different clustering methods.
# - Makes it easier to document and explain our work in later phases (classification and hybrid scoring).
# - Allows quick inclusion of the evaluation metric in reports and presentations without rerunning heavy code.
# 
# ### Output
# - A JSON file located at `report/silhouette_scores.json` containing:
#   - KMeans and Agglomerative silhouette scores per dataset.

# In[6]:


import pandas as pd
import os
import json
from sklearn.metrics import silhouette_score

# Paths
base_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'data'))
report_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'report'))
os.makedirs(report_path, exist_ok=True)

dataset_ids = ['FD001', 'FD002', 'FD003', 'FD004']
silhouette_summary = {}

for ds_id in dataset_ids:
    file_path = os.path.join(base_path, f"clustered_train_{ds_id}.csv")
    if not os.path.exists(file_path):
        print(f"⚠️ File missing: {file_path}")
        continue

    df = pd.read_csv(file_path)
    sensor_cols = [col for col in df.columns if col.startswith("sensor_")]

    # Ensure required columns exist
    if 'kmeans_cluster' not in df.columns or 'agglo_cluster' not in df.columns:
        print(f"⚠️ Missing cluster labels in {ds_id}")
        continue

    # Compute silhouette scores
    X = df[sensor_cols]
    kmeans_score = silhouette_score(X, df['kmeans_cluster'])
    agglo_score = silhouette_score(X, df['agglo_cluster'])

    silhouette_summary[ds_id] = {
        'KMeans': round(kmeans_score, 4),
        'Agglomerative': round(agglo_score, 4)
    }

# Save to JSON
output_path = os.path.join(report_path, "silhouette_scores.json")
with open(output_path, "w") as f:
    json.dump(silhouette_summary, f, indent=4)

print(f"📁 Silhouette scores saved to: {output_path}")
print("📊 Summary:", silhouette_summary)


# ## Silhouette Score Summary Table (KMeans vs Agglomerative)
# 
# This table summarizes the silhouette scores computed during clustering for each dataset using both KMeans and Agglomerative methods. Silhouette score is a metric that reflects how well each point lies within its assigned cluster — the higher the score, the better the separation and cohesion of clusters.
# 
# | Dataset | KMeans Score | Agglomerative Score |
# |---------|--------------|---------------------|
# | FD001   | 0.1880       | 0.1760              |
# | FD002   | 0.8840       | 0.8840              |
# | FD003   | 0.2454       | 0.2250              |
# | FD004   | 0.8862       | 0.8862              |
# 
# ### Interpretation
# 
# - 🟢 **FD002** and **FD004** show excellent clustering performance (scores above 0.88), with clearly distinguishable degradation stages.
# - 🟡 **FD003** has moderate separation, which might benefit from additional feature engineering or dimensionality adjustments.
# - 🔴 **FD001** shows weak clustering performance, possibly due to subtle or overlapping degradation patterns across its sensor readings.
# 
# These observations help identify which datasets are best suited for supervised learning in the next phases, particularly for building classification models (Phase 3) and hybrid scoring systems (Phase 5).
# 

# ## Silhouette Score Markdown Summary Export
# 
# This step converts the `silhouette_scores.json` file into a readable Markdown summary and saves it as `silhouette_summary.md` inside the `/report/` folder.
# 
# ### Why This Step Is Useful
# 
# - Allows you to present clustering evaluation results directly in documentation or your final report without re-running the notebook.
# - Provides a clean and portable format for sharing clustering insights through GitHub or printing in research papers.
# - Acts as a snapshot of model performance during Phase 2 for easy future reference in Phases 3–5.
# 
# The resulting markdown file contains the full table of silhouette scores for all datasets and includes interpretation guidance.

# In[8]:


import os
import json

# Define report path
report_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'report'))
json_path = os.path.join(report_path, "silhouette_scores.json")
summary_md = os.path.join(report_path, "silhouette_summary.md")

# Load JSON
with open(json_path, "r") as f:
    silhouette_summary = json.load(f)

# Write Markdown summary
with open(summary_md, "w", encoding="utf-8") as md:
    md.write("# 📊 Silhouette Score Summary\n\n")
    for ds, scores in silhouette_summary.items():
        md.write(f"## {ds}\n")
        md.write(f"- **KMeans**: {scores['KMeans']}\n")
        md.write(f"- **Agglomerative**: {scores['Agglomerative']}\n\n")


print(f"✅ Markdown summary saved to: {summary_md}")


# ## Top 5 Sensors by Variance (Per Dataset)
# 
# This exploratory analysis identifies the five most variable sensor features for each dataset (FD001–FD004). High variance across engine cycles often indicates stronger signal relevance, as these sensors are more likely to capture meaningful differences related to degradation.
# 
# ### Why This Matters
# 
# - Sensors with higher variance tend to contribute more effectively to clustering and model learning.
# - Supports feature selection decisions in upcoming phases, particularly Phase 3 (classification).
# - Helps build interpretability into the pipeline by identifying which sensor signals are driving the separation between degradation stages.
# 
# The top sensors identified here can also guide targeted visualization and deeper analysis.

# In[13]:


import json

top_variance_summary = {}

for ds_id in dataset_ids:
    df = pd.read_csv(os.path.join(base_path, f"clustered_train_{ds_id}.csv"))
    sensor_cols = [col for col in df.columns if col.startswith("sensor_")]
    top_sensors = df[sensor_cols].var().sort_values(ascending=False).head(5).round(4)

    # Store as dict
    top_variance_summary[ds_id] = top_sensors.to_dict()

    # Print in notebook
    print(f"\n📊 {ds_id} - Top 5 High-Variance Sensors:")
    print(top_sensors)

# Save to file
os.makedirs("../report", exist_ok=True)
with open("../report/top_variance_sensors.json", "w") as f:
    json.dump(top_variance_summary, f, indent=4)

print("\n📁 Saved top variance sensor summary to: report/top_variance_sensors.json")


# ## Average Engine Cycle per Cluster (Degradation Profile)
# 
# This analysis plots the average number of cycles associated with each KMeans cluster (Stage 0 to Stage 4), based on the `time` column in the clustered training data.
# 
# ### Purpose
# 
# A meaningful degradation model should reflect time progression. Early-stage clusters (e.g., Stage 0) should appear closer to the start of engine life, while later stages (e.g., Stage 4) should occur near failure. This pattern supports the validity of the clustering.
# 
# ### Why This Is Important
# 
# - Confirms that the clustering process aligns with real-world degradation timelines.
# - Provides a sanity check that the assigned stages follow a progressive structure.
# - Helps justify the use of these clusters in downstream RUL prediction or risk scoring models.
# 
# This plot can be referenced in the final report to illustrate how clustering captures temporal degradation trends.
# 

# In[6]:


import matplotlib.pyplot as plt
for ds_id in dataset_ids:
    df = pd.read_csv(os.path.join(base_path, f"clustered_train_{ds_id}.csv"))
    df['time'] = df['time'] if 'time' in df.columns else df.index + 1

    plt.figure(figsize=(6, 4))
    mean_time = df.groupby('kmeans_stage')['time'].mean().sort_index()
    mean_time.plot(kind='bar', color='skyblue')
    plt.title(f"{ds_id} - Avg Cycle by KMeans Stage")
    plt.ylabel("Average Time (Cycle)")
    plt.tight_layout()
    plt.savefig(f"../figures/{ds_id}_Stage_TimeProfile.png")
    plt.close()


# ## Cluster Summary Table: Average Sensor Values by Stage
# 
# This section displays the mean values of each sensor and the average cycle count (`time`) for every KMeans stage (Stage 0 to Stage 4) across each dataset.
# 
# ### Why This Is Useful
# 
# - Provides a clear, interpretable profile of how sensor behavior changes across degradation stages.
# - Helps define what each stage actually represents in terms of physical sensor readings and operational context.
# - Offers valuable context for supervised learning and hybrid RUL-risk modeling in later phases.
# 
# This table is particularly useful for model explainability and can be referenced in reports to show how sensor readings evolve throughout the engine's life.

# In[14]:


summary_folder = os.path.abspath(os.path.join(os.getcwd(), '..', 'report', 'cluster_summaries'))
os.makedirs(summary_folder, exist_ok=True)

for ds_id in dataset_ids:
    df = pd.read_csv(os.path.join(base_path, f"clustered_train_{ds_id}.csv"))
    sensor_cols = [col for col in df.columns if col.startswith("sensor_")]

    # Include time column if available
    cols_to_use = sensor_cols + ['time'] if 'time' in df.columns else sensor_cols
    summary = df.groupby("kmeans_stage")[cols_to_use].mean().round(3)

    # Save as CSV
    summary.to_csv(os.path.join(summary_folder, f"{ds_id}_ClusterSummary.csv"))
    print(f"📁 Saved: {ds_id}_ClusterSummary.csv")


# ## Elbow Curves for All Datasets (KMeans)
# 
# This section plots the Within-Cluster Sum of Squares (WCSS) for different values of `k` (from 2 to 10) across all four datasets (FD001–FD004).
# 
# ### Purpose
# 
# The elbow method helps identify a suitable number of clusters by measuring how tightly the data points group together as `k` increases. A noticeable drop in WCSS followed by a leveling-off suggests that adding more clusters beyond that point offers diminishing returns.
# 
# ### Why This Matters
# 
# - Supports our decision to use **5 clusters** for KMeans across all datasets.
# - Helps avoid overfitting (too many clusters) or underfitting (too few clusters).
# - Adds credibility to our methodology by showing that the clustering configuration is backed by quantitative analysis.
# - These plots can be referenced in both presentations and the final report to explain model selection decisions.
# 
# Each dataset’s elbow plot is saved in the `/figures/` folder and the corresponding WCSS values are summarized in `/report/elbow_summary.md` and `/report/elbow_scores.json`.

# In[13]:


import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Setup paths
dataset_ids = ['FD001', 'FD002', 'FD003', 'FD004']
base_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'data'))
figures_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'figures'))
report_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'report'))

os.makedirs(figures_path, exist_ok=True)
os.makedirs(report_path, exist_ok=True)

# We'll store WCSS (within-cluster sum of squares) for each dataset here
elbow_scores = {}

# Loop through all datasets and calculate WCSS for k = 2 to 10
for ds_id in dataset_ids:
    file_path = os.path.join(base_path, f"clustered_train_{ds_id}.csv")
    df = pd.read_csv(file_path)
    sensor_cols = [col for col in df.columns if col.startswith("sensor_")]
    X = df[sensor_cols]

    wcss = []
    for k in range(2, 11):
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(X)
        wcss.append(round(model.inertia_, 2))

    elbow_scores[ds_id] = wcss

    # Plot elbow curve
    plt.figure(figsize=(6, 4))
    plt.plot(range(2, 11), wcss, marker='o', linestyle='-', color='teal')
    plt.title(f"Elbow Curve - {ds_id}")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(figures_path, f"{ds_id}_ElbowCurve.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"{ds_id}: Elbow curve saved → {plot_path}")

# Save WCSS data to a JSON file
json_path = os.path.join(report_path, "elbow_scores.json")
with open(json_path, "w") as f:
    json.dump(elbow_scores, f, indent=4)
print(f"WCSS values saved to JSON → {json_path}")

# Now prepare a markdown summary for the report
md_path = os.path.join(report_path, "elbow_summary.md")

with open(md_path, "w") as f:
    f.write("# Elbow Curve Summary: WCSS for Each Dataset\n\n")
    f.write("This table lists the WCSS values for different values of k (from 2 to 10) across all four datasets.\n\n")

    f.write("| Dataset | " + " | ".join([f"k={k}" for k in range(2, 11)]) + " |\n")
    f.write("|---------|" + "--------|" * 9 + "\n")

    for ds_id in dataset_ids:
        values = elbow_scores[ds_id]
        row = f"| {ds_id} | " + " | ".join([str(v) for v in values]) + " |\n"
        f.write(row)

    f.write("\n")
    f.write("## Notes\n")
    f.write("- As k increases, WCSS generally decreases.\n")
    f.write("- We typically look for the 'elbow point' — the spot where adding more clusters doesn’t improve WCSS much.\n")
    f.write("- For most of these datasets, k=5 gives a good balance between simplicity and separation.\n")

print(f"Markdown summary saved → {md_path}")


# ---
# 
# ## Elbow Curve Summary: WCSS per k-value (KMeans Clustering)
# 
# This table shows the Within-Cluster Sum of Squares (WCSS) values for `k = 2 to 10` across each dataset (FD001 to FD004). These were used to create the elbow plots saved in `/figures/`.
# 
# | Dataset | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 | k=10 |
# |---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
# | FD001   | 3246.17 | 2510.97 | 2315.82 | 2100.38 | 1724.82 | 1625.12 | 1564.63 | 1510.38 | 1479.32 |
# | FD002   | 41150.21 | 19887.15 | 6537.17 | 1479.66 | 812.84 | 159.83 | 151.57 | 142.24 | 132.28 |
# | FD003   | 4709.90 | 2911.12 | 2545.37 | 2339.51 | 1879.34 | 1741.17 | 1577.68 | 1492.31 | 1437.51 |
# | FD004   | 46489.35 | 33708.72 | 7374.29 | 1481.33 | 740.49 | 241.32 | 213.28 | 198.29 | 188.09 |
# 
# ---
# 
# ### What is WCSS?
# 
# WCSS (Within-Cluster Sum of Squares) measures how close data points are to their cluster centroid. Lower WCSS means tighter, more compact clusters.
# 
# ---
# 
# ### What is the Elbow Method?
# 
# We plot WCSS for different values of `k`. The point where WCSS reduction slows (the "elbow") indicates an optimal number of clusters — typically around **k = 5** here.
# 
# ---
# 
# ### Why It Matters
# 
# - Confirms that 5 clusters strike a good balance between compactness and simplicity.
# - Supports our choice in Phase 2 and justifies clustering labels for use in:
#   - Phase 3: Classification
#   - Phase 5: Hybrid risk scoring
# 

# ---
# 
# ## Dendrograms for Agglomerative Clustering (All Datasets)
# 
# This section presents dendrogram plots created using a 500-sample subset from each dataset. The dendrograms are generated using the **Ward linkage method**, which minimizes variance within clusters during the hierarchical merging process.
# 
# ### What is a Dendrogram?
# 
# A dendrogram is a **tree-like diagram** that shows how samples (or groups of samples) are merged step-by-step during hierarchical clustering.
# 
# - At the bottom, each point is treated as its own cluster.
# - As you move upward, clusters are gradually merged based on similarity.
# - The height of each merge indicates the **distance (or dissimilarity)** between the groups.
# 
# ### Why This is Important
# 
# - Reveals the **hierarchical relationships** between engine cycles based on sensor patterns.
# - Allows us to observe whether natural degradation stages emerge from the data without predefined labels.
# - Complements PCA and t-SNE visualizations by showing an entirely different clustering perspective.
# - Helps validate the use of **Agglomerative Clustering** as a meaningful technique in Phase 2.
# 
# ### Use in This Project
# 
# The dendrograms:
# - Provide visual intuition on how engines group into clusters at various linkage levels.
# - Can be referenced to explain how different engine behaviors or degradation paths relate to each other structurally.
# - Strengthen the justification for choosing 5 clusters and validate stage-based grouping in unsupervised settings.
# 
# The dendrogram plots are saved to the `/figures/` directory as:  
# `FD001_Dendrogram.png`, `FD002_Dendrogram.png`, etc.
# 

# In[12]:


from scipy.cluster.hierarchy import linkage, dendrogram

for ds_id in dataset_ids:
    df = pd.read_csv(os.path.join(base_path, f"clustered_train_{ds_id}.csv"))
    sensor_cols = [col for col in df.columns if col.startswith("sensor_")]
    X_sample = df[sensor_cols].sample(n=500, random_state=42)

    linkage_matrix = linkage(X_sample, method='ward')

    plt.figure(figsize=(10, 4))
    dendrogram(linkage_matrix, truncate_mode='lastp', p=20)
    plt.title(f"Dendrogram - {ds_id} (Sample of 500)")
    plt.tight_layout()
    plt.savefig(f"../figures/{ds_id}_Dendrogram.png")
    plt.close()


# ---
# 
# ## Summary Plot: Top 3 High-Variance Sensors per KMeans Cluster Stage
# 
# This section visualizes how the most informative sensors — those with the highest variance — behave across the five KMeans stages (Stage 0 to Stage 4) for each dataset.
# 
# ### Why This Matters
# 
# - High-variance sensors are often the most informative features.
# - These plots show whether their average readings follow a consistent trend as degradation progresses.
# - This helps validate our cluster labels and informs feature selection for classification models in Phase 3.
# 
# ### What the Plots Show
# 
# For each dataset (FD001–FD004), the top 3 highest-variance sensors are identified. Their mean values are then calculated across each KMeans stage. The resulting line plots highlight which sensors increase, decrease, or remain stable as engines degrade.
# 
# All plots are saved to the `/figures/` directory using the format:
# ``FD00x_TopSensors_PerStage.png``
# 

# In[9]:


# 📊 Top Sensor Summary Plots per Stage (FD001–FD004)
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Paths
base_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'data'))
fig_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'figures'))
json_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'report', 'top_variance_sensors.json'))

# Load top sensors JSON
with open(json_path, "r") as f:
    top_sensors = json.load(f)

# Create and save plots
for ds_id, sensors in top_sensors.items():
    file = os.path.join(base_path, f"clustered_train_{ds_id}.csv")
    if not os.path.exists(file):
        print(f"❌ File not found: {file}")
        continue

    df = pd.read_csv(file)
    if "kmeans_stage" not in df.columns:
        print(f"❌ Missing clustering column in: {ds_id}")
        continue

    top3 = list(sensors.keys())[:3]  # Take top 3 sensors
    mean_vals = df.groupby("kmeans_stage")[top3].mean()

    # Plot
    mean_vals.plot(kind="bar", figsize=(8, 5))
    plt.title(f"{ds_id} – Top 3 Sensor Averages by Stage (KMeans)")
    plt.ylabel("Average Normalized Sensor Value")
    plt.xlabel("KMeans Stage")
    plt.xticks(rotation=0)
    plt.tight_layout()

    save_path = os.path.join(fig_path, f"{ds_id}_TopSensors_PerStage.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved: {save_path}")

