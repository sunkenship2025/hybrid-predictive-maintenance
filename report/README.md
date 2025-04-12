# 📁 /report — Evaluation Metrics & Clustering Insights

This folder consolidates **key clustering evaluation results**, **feature insights**, and **summarized statistics** from Phase 1 and Phase 2. These files are designed to support Phase 3 (Classification), Phase 4 (RUL Prediction), and Phase 5 (Hybrid Risk Modeling).

---

## 📦 Contents

| File / Folder | Description |
|---------------|-------------|
| `silhouette_scores.json` | ✅ Contains silhouette scores for all four datasets (FD001–FD004), calculated using both **KMeans** and **AgglomerativeClustering**. These scores reflect how well-separated the clusters are. |
| `silhouette_summary.md` | 📝 A **human-readable Markdown report** summarizing the silhouette scores, with colored insights and clear commentary on which datasets have good clustering separation. Useful for presentations or final documentation. |
| `top_variance_sensors.json` | ✅ Lists the **top 5 high-variance sensors** per dataset. High variance is often indicative of informative features for classification and regression. |
| `cluster_summaries/` | 📊 Folder containing `FD00x_ClusterSummary.csv` files. Each file summarizes the **mean value of each sensor** across degradation stages (KMeans), useful for interpreting health patterns. |
| `contributors.txt` | (Optional) A list of project contributors or collaborators. Include names and emails if submitting for review or as a team project. |

---

## 🔍 Why This Folder Matters

- Helps evaluate **clustering quality** with reproducible metrics  
- Provides **stage-wise insights** to support supervised learning  
- Improves transparency for model interpretability and final reporting  
- Enables quick lookup of which sensors matter most across phases  
- Markdown summaries like `silhouette_summary.md` are ideal for directly copy-pasting into research reports or submission documents

---

## ✅ Used In

- 📊 Phase 2 – Clustering validation  
- 🧠 Phase 3 – Classification model input feature design  
- 📉 Phase 5 – Hybrid scoring based on degradation cluster context  
- 📚 Final report – Justify dataset reliability and cluster-based strategies  

---
