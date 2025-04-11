# 📁 /cluster_summaries — Sensor Behavior Across Clusters

Each CSV summarizes **per-cluster sensor readings and average cycle** for a dataset.

---

## 🧾 Files:
| File | Content |
|------|---------|
| `FD001_ClusterSummary.csv` – `FD004_ClusterSummary.csv` | Mean of each sensor + time for KMeans `Stage 0` to `Stage 4` |

---

## 📊 Usage:
- Visualize sensor degradation per cluster
- Helps link cluster labels to actual physical behavior
- Critical for Phase 3 classification (labeling) and Phase 4 regression modeling

Example insight:
> "In FD001, `sensor_11` and `sensor_12` decrease steadily from Stage 0 to Stage 4 — confirming their relevance for degradation modeling."
