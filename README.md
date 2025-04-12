# 🔧 Hybrid Predictive Maintenance using Enhanced CMAPSS Dataset

This project builds a hybrid system that predicts both **engine degradation stages** and **remaining useful life (RUL)** using machine learning on the NASA CMAPSS datasets.

---

## 📁 Folder Structure

| Folder         | Purpose |
|----------------|---------|
| `data/`        | Contains raw, cleaned, and clustered CSVs for FD001–FD004 |
| `figures/`     | PCA, t-SNE, distribution plots, and degradation profiles |
| `report/`      | JSON files, silhouette scores, contributors.txt, and final outputs |
| `notebooks/`   | Phase-wise notebooks with all code and markdown |
| `docs/`        | *(Optional)* Use this to build your final paper/report |

---

## 🚦 Project Phases

### ✅ Phase 1: Data Cleaning & Normalization
- Removed constant features
- Scaled sensor data using MinMaxScaler
- Saved cleaned data for reuse

### ✅ Phase 2: Clustering + Stage Labeling
- Applied KMeans and Agglomerative clustering (5 stages)
- Visualized clusters using PCA and t-SNE
- Calculated silhouette scores and exported insights

### 🔜 Phase 3: Classification (Next)
- Predict degradation stage using supervised models

---

## 👨‍💻 Team Contributors

Names are listed in `/report/contributors.txt`.

---

## 📄 Final Report
> Will be compiled and submitted after Phase 5 using Typst.

---

## 📅 Timeline
Final submission deadline: **May 5**

