# /report — Evaluation Metrics and Clustering Insights

This folder consolidates all evaluation results and feature-based insights generated in Phase 1, Phase 2, and Phase 2.5 of the project. These resources serve as supporting material for later stages, including supervised classification (Phase 3), RUL regression (Phase 4), and hybrid risk modeling (Phase 5).

---

## Contents

| File or Folder | Description |
|----------------|-------------|
| [`silhouette_scores.json`](./silhouette_scores.json) | Silhouette scores for each dataset (FD001–FD004), calculated using both KMeans and Agglomerative Clustering. These values help measure how well-separated and well-formed the clusters are. |
| [`silhouette_summary.md`](./silhouette_summary.md) | A Markdown summary explaining the silhouette scores for all datasets, with a short interpretation of how good the clustering was. Useful for documentation and presentations. |
| [`top_variance_sensors.json`](./top_variance_sensors.json) | Lists the top 5 high-variance sensors for each dataset. These sensors are more informative and useful for feature selection in later phases. |
| [`elbow_scores.json`](./elbow_scores.json) | WCSS values for KMeans clustering with `k` ranging from 2 to 10. Used to generate elbow plots for selecting the best number of clusters. |
| [`elbow_summary.md`](./elbow_summary.md) | Markdown summary of the elbow method results, including a full WCSS table and explanation of how the best `k` was chosen. |
| [`cluster_summaries/`](./cluster_summaries/) | Contains per-dataset CSV files showing average sensor values and time per cluster stage (KMeans), which helps in interpreting the stages. |
| [`manual_cluster_verification/`](./manual_cluster_verification/) | (Phase 2.5) Contains analysis results from manual cluster verification, coinciding stage detection, and stage relabeling applied to FD001–FD004 datasets. |
| [`correlation_heatmaps/`](../figures/) | Heatmaps showing sensor correlation before and after normalization for each dataset. These were created to demonstrate the impact of scaling during Phase 1. |
| [`contributors.txt`](./contributors.txt) | Optional file listing team contributors and their contact information. Include if submitting as a group. |

---

## Purpose and Use

This folder makes it easy to track and reuse the evaluation metrics, summaries, and supporting analysis needed for the later phases of the project.

- **Silhouette scores** help validate how well the clustering worked.
- **Variance rankings** guide sensor selection for classification models.
- **Elbow scores** justify the use of five clusters.
- **Cluster summaries** explain how sensor values evolve with degradation.
- **Manual cluster verification** ensures cluster stages represent true degradation progression, improving classification and RUL modeling reliability.
- **Correlation heatmaps** provide preprocessing transparency.
- **Markdown summaries** make it easy to reuse results in the final paper or report.

---

## Referenced In

- Phase 2: Unsupervised clustering evaluation
- Phase 2.5: Manual verification of clustering stages and stage correction
- Phase 3: Sensor selection and stage label usage for classifiers
- Phase 4: Mapping corrected degradation stages to RUL modeling
- Phase 5: Using insights from clustering to score hybrid risk
- Final Report: All scores, corrections, and plots from this folder will be cited and used as figures

