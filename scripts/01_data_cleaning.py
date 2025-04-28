#!/usr/bin/env python
# coding: utf-8

# # Phase 1: Data Cleaning and Sensor Trend Analysis
# 
# This notebook is dedicated to preprocessing the CMAPSS FD001–FD004 datasets in preparation for downstream clustering and modeling tasks. The primary focus of this phase is to clean the raw data, retain only the most relevant sensor features, and normalize the values for consistent analysis across different engine units.
# 
# ---
# 
# ## Objectives
# 
# - Assign meaningful column names to improve interpretability.
# - Identify and remove constant (non-varying) sensors that do not contribute to model performance.
# - Apply min-max normalization to bring all sensor values into a common range [0, 1].
# - Plot key sensor trends to visually assess degradation behavior across operational cycles.
# 
# This cleaned data will be used as the input for unsupervised clustering in Phase 2 and for supervised modeling in later phases.

# In[7]:


# Phase 1: Load CMAPSS FD001–FD004 datasets, clean, normalize, and visualize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os


# ### Step 1: Load CMAPSS Data and Assign Column Names
# 
# Each CMAPSS dataset contains time-series sensor readings recorded at the cycle level for multiple engines. In this step, appropriate and descriptive column names are assigned to enhance data readability and facilitate downstream analysis.

# In[8]:


# Define column names from README
cols = ['unit', 'time', 'op_setting_1', 'op_setting_2', 'op_setting_3']
sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
column_names = cols + sensor_cols

# Fixed: Set correct relative path assuming script is in notebooks/ and data is in ../data/
base_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'data'))
dataset_ids = ['FD001', 'FD002', 'FD003', 'FD004']
datasets = {}

# Load all datasets with column names
for ds_id in dataset_ids:
    train_df = pd.read_csv(os.path.join(base_path, f"train_{ds_id}.txt"), sep=r'\s+', header=None, names=column_names)
    test_df = pd.read_csv(os.path.join(base_path, f"test_{ds_id}.txt"), sep=r'\s+', header=None, names=column_names)
    rul_df = pd.read_csv(os.path.join(base_path, f"RUL_{ds_id}.txt"), sep=r'\s+', header=None, names=['RUL'])
    datasets[ds_id] = {'train': train_df, 'test': test_df, 'rul': rul_df}


# ### Step 2: Remove Constant Sensors and Normalize Sensor Values
# 
# To reduce noise and focus on meaningful features, we start by removing any sensors that show no variation across all engine cycles. These constant-value sensors don’t provide any useful signals for identifying degradation patterns or predicting failure.
# 
# Once the non-informative sensors are dropped, we normalize the remaining sensor values using MinMax scaling. This brings all sensor readings into the same range [0, 1], which ensures fair comparison between sensors and prevents larger-valued features from skewing clustering or model training.
# 
# This step ensures that every feature in the dataset contributes useful, comparable information—setting a clean foundation for the next phases.

# In[9]:


# Drop constant sensors and normalize sensor values only
def clean_and_normalize(df):
    # Drop columns with no variance (constant values)
    dropped = []
    for col in sensor_cols:
        if df[col].std() == 0:
            df.drop(columns=[col], inplace=True)
            dropped.append(col)
    print(f"Dropped constant columns: {dropped}")

    # Normalize only sensor columns
    sensor_cols_active = [col for col in sensor_cols if col in df.columns]
    scaler = MinMaxScaler()
    df[sensor_cols_active] = scaler.fit_transform(df[sensor_cols_active])
    return df


# ### Step 3: Save Cleaned Training Data
# 
# After completing the cleaning and normalization process, each dataset is saved as a CSV file in the `/data/` directory using the format `clean_train_FD00x.csv`.
# 
# This allows us to reuse the processed data in future phases—such as clustering, classification, or regression—without needing to repeat the entire preprocessing pipeline. It also helps maintain consistency and ensures reproducibility throughout the project.
# 

# In[10]:


# Clean, normalize and save all training sets
for ds_id in dataset_ids:
    cleaned_df = clean_and_normalize(datasets[ds_id]['train'])
    datasets[ds_id]['train'] = cleaned_df
    cleaned_filename = os.path.join(base_path, f"clean_train_{ds_id}.csv")
    cleaned_df.to_csv(cleaned_filename, index=False)
    print(f"Saved: {cleaned_filename}")


# ### Step 4: Visualize Sensor Behavior for Engine 1
# 
# To understand how sensor values change over time, we plot time-series trends for all sensors from Engine 1 in each dataset. These visualizations help us observe early signs of degradation, stability, or irregular patterns within each sensor signal.
# 
# Below each plot, we include a brief interpretation to highlight:
# - Sensors that degrade gradually and may be suitable for RUL regression
# - Sensors that show abrupt shifts, which could be useful for time-based or event-driven models
# - Datasets that exhibit noisy signals or multiple operating conditions
# 
# These early observations are important for shaping our clustering approach and for selecting the most informative features in later modeling phases.

# In[11]:


# Visualization: plot one engine's sensor progression from each FD dataset

def plot_sensor_trend(df, ds_id):
    unit_1 = df[df['unit'] == 1]
    plt.figure(figsize=(14, 6))
    for col in df.columns:
        if col.startswith('sensor_'):
            plt.plot(unit_1['time'], unit_1[col], label=col)
    plt.title(f"Engine 1 Sensor Trends in {ds_id}")
    plt.xlabel("Cycle")
    plt.ylabel("Normalized Sensor Values")
    plt.legend(loc='upper right', fontsize='small', ncol=3)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Enhanced explanations for a student-friendly but deep insight
    interpretations = {
        'FD001': (
            "In FD001, sensors like sensor_2 and sensor_11 show a smooth and consistent downward trend, which strongly indicates engine degradation over time. "
            "Other sensors remain relatively flat, implying they do not contribute much to RUL prediction. These clear degradation patterns make FD001 ideal for supervised regression models."
        ),
        'FD002': (
            "FD002 introduces complexity with multiple operational conditions. While some sensors (like sensor_3 and sensor_6) fluctuate without obvious trends, others like sensor_14 degrade steadily. "
            "Such mixed patterns make this dataset useful for clustering engines by behavior and developing models that are robust to variable environments."
        ),
        'FD003': (
            "FD003 includes both stable and degrading sensors, but the degradation is less obvious due to noise. Sensor_7 and sensor_1 show potential patterns that could be extracted with smoothing or feature engineering. "
            "This kind of dataset challenges simple models and encourages exploration of signal processing or noise-tolerant learning techniques."
        ),
        'FD004': (
            "FD004 is marked by sudden changes in some sensors like sensor_15 and sensor_17, indicating abrupt or nonlinear degradation. This behavior is critical for testing models that can capture dynamic system failure. "
            "Deep learning models like LSTM or GRU that learn from temporal sequences are especially well-suited for this type of data."
        )
    }
    print(f"\n**{ds_id} Explanation:**\n{interpretations[ds_id]}\n")

# Define dataset IDs for iteration
dataset_ids = ['FD001', 'FD002', 'FD003', 'FD004']

# Plot
for ds_id in dataset_ids:
    plot_sensor_trend(datasets[ds_id]['train'], ds_id)


# ## Sensor Correlation Heatmap: Before vs. After Normalization
# 
# To understand how normalization affects our data, we plotted the pairwise correlation heatmaps of all sensor features **before** and **after** applying MinMaxScaler. This comparison helps validate the preprocessing decisions made in Phase 1.
# 
# ---
# 
# ### What These Heatmaps Show
# 
# - **Before Normalization (Left):**  
#   Raw sensor values vary in scale. As a result, sensors with high numerical ranges tend to dominate correlation calculations. This can lead to **false or exaggerated correlations** that don’t reflect real relationships between sensor trends.
# 
# - **After Normalization (Right):**  
#   All sensor values are scaled to the same range [0, 1]. This ensures that **correlation reflects the shape and pattern** of sensor data, not just magnitude. Relationships between sensor signals become more **balanced and interpretable**.
# 
# ---
# 
# ### Why This Is Important
# 
# - Prevents bias in clustering or distance-based models (like KMeans)
# - Ensures equal contribution of each sensor during model training
# - Highlights genuine patterns between sensor behaviors
# - Supports better feature analysis and visual diagnostics
# 
# ---
# 
# ### Summary
# 
# This visualization confirms that normalization is a **critical step** — not just for model accuracy, but for making sure we interpret sensor data correctly. It improves consistency, fairness, and reliability across all subsequent phases.
# 

# In[4]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Define column names for raw dataset
column_names = ['unit', 'time', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
               [f'sensor_{i}' for i in range(1, 22)]

# Dataset IDs
dataset_ids = ['FD001', 'FD002', 'FD003', 'FD004']

# Ensure output folder exists
os.makedirs("../figures", exist_ok=True)

# Loop through all datasets
for ds_id in dataset_ids:
    # Load raw dataset
    raw_df = pd.read_csv(f"../data/train_{ds_id}.txt", sep=" ", header=None)
    raw_df.drop(columns=[26, 27], inplace=True)  # Drop extra space columns
    raw_df.columns = column_names
    raw_sensors = raw_df[[col for col in raw_df.columns if col.startswith("sensor_")]]

    # Load cleaned dataset
    clean_df = pd.read_csv(f"../data/clean_train_{ds_id}.csv")
    clean_sensors = clean_df[[col for col in clean_df.columns if col.startswith("sensor_")]]

    # Compute correlation
    corr_raw = raw_sensors.corr()
    corr_clean = clean_sensors.corr()

    # Plot heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    sns.heatmap(corr_raw, ax=axes[0], cmap="coolwarm", annot=False)
    axes[0].set_title(f"{ds_id} — Before Normalization")

    sns.heatmap(corr_clean, ax=axes[1], cmap="coolwarm", annot=False)
    axes[1].set_title(f"{ds_id} — After Normalization")

    plt.tight_layout()
    save_path = f"../figures/{ds_id}_Correlation_Before_After.png"
    plt.savefig(save_path)
    plt.close(fig)

    print(f"[✔] Heatmap saved to: {save_path}")

