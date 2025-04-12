# /report — Evaluation Metrics and Clustering Insights

This folder consolidates all evaluation results and feature-based insights generated in Phase 1 and Phase 2 of the project. These resources serve as supporting material for later stages, including supervised classification (Phase 3), RUL regression (Phase 4), and hybrid risk modeling (Phase 5).

---

## Contents

| File or Folder | Description |
|----------------|-------------|
| [`silhouette_scores.json`](./silhouette_scores.json) | Silhouette scores for each dataset (FD001–FD004) using both KMeans and Agglomerative clustering. Indicates how well-separated the clusters are. |
| [`silhouette_summary.md`](./silhouette_summary.md) | Human-readable markdown summary of silhouette scores, including commentary and observations for each dataset. Useful for reporting and presentations. |
| [`elbow_scores.json`](./elbow_scores.json) | WCSS (Within-Cluster Sum of Squares) values for k = 2 to 10 per dataset, used to evaluate optimal cluster count using the elbow method. |
| [`elbow_summary.md`](./elbow_summary.md) | Markdown table of WCSS scores with interpretation. Helps justify the use of 5 clusters in Phase 2. |
| [`top_variance_sensors.json`](./top_variance_sensors.json) | Top 5 high-variance sensors per dataset. These are useful for feature selection, dimensionality reduction, and visualization. |
| [`cluster_summaries/`](./cluster_summaries/) | CSV files showing average sensor values per KMeans stage for each dataset. Used to interpret stage meaning and sensor behavior over time. |
| [`contributors.txt`](./contributors.txt) | (Optional) List of contributors with names, emails, or GitHub handles for collaborative submissions. |

---

## Purpose and Use

This directory contains critical artifacts for evaluation, interpretability, and downstream modeling. These files support:

- Verifying clustering quality with objective metrics (silhouette, WCSS)
- Selecting informative features based on sensor variance
- Interpreting degradation stages using cluster-wise sensor summaries
- Embedding summarized insights into reports or presentations

In particular:
- `elbow_scores.json` and `elbow_summary.md` justify cluster configuration using the elbow method.
- Top sensor behavior (e.g., in `FD00x_TopSensors_PerStage.png`) is derived from `top_variance_sensors.json` and supports diagnostics.
- Markdown files ensure clarity and ease of use for documentation and evaluations.

---

## Referenced In

- **Phase 2**: Clustering evaluation, visualization, and explanation
- **Phase 3**: Feature selection for classification
- **Phase 4–5**: Hybrid scoring and time-to-failure modeling
- **Final Report**: Insight justification, reproducibility, and result interpretation