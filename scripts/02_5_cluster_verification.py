#!/usr/bin/env python
# coding: utf-8

# ## FD001 - Load Data and Plot Top 5 Sensor Behavior
# 
# In this section, we load the clustered FD001 data and plot the behavior of the top 5 variance sensors (`sensor_11`, `sensor_12`, `sensor_4`, `sensor_2`, and `sensor_21`) across different KMeans cluster stages.  
# The goal is to visually verify the degradation patterns for manual cluster validation.
# 

# In[6]:


# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

# Load clustered FD001 data
data_path = "../data/clustered_train_FD001.csv"
df = pd.read_csv(data_path)

# Show available columns
print(df.columns)

# --- ✅ Paste FD001 Variance Calculation Block here ---
sensor_columns_fd001 = [col for col in df.columns if 'sensor_' in col]
variances_fd001 = df[sensor_columns_fd001].var().sort_values(ascending=False)
print("Top 5 variance sensors for FD001:")
print(variances_fd001.head(5))

top_sensors_fd001 = ['sensor_11', 'sensor_12', 'sensor_4', 'sensor_2', 'sensor_21']

# Create output directory
output_dir = "../figures/manual_cluster_verification/FD001/"
os.makedirs(output_dir, exist_ok=True)

# Correct cycle column
cycle_column = 'time'

# Plot mean sensor values
for sensor in top_sensors_fd001:
    plt.figure(figsize=(10,6))
    sns.lineplot(
        x=cycle_column,
        y=sensor,
        hue='kmeans_stage',
        data=df,
        palette='tab10'
    )
    plt.title(f"{sensor} behavior across KMeans Cluster Stages - FD001")
    plt.xlabel("Cycle Number (Time)")
    plt.ylabel(f"{sensor} Reading (Normalized)")
    plt.legend(title="KMeans Stage")
    plt.grid(True)
    plt.savefig(f"{output_dir}{sensor}_kmeans_stage_fd001.png")
    plt.close()

# --- Then after plotting comes ---
# Relabel Stage 3 and 4
# Save corrected file


# ## FD001 - Coinciding Stage Detection
# 
# Here, we check if any KMeans stages have similar (coinciding) mean sensor readings using a small tolerance threshold.  
# This helps identify if two or more cluster stages are indistinguishable and need correction.
# 

# In[15]:


# --- Coinciding Stage Detection for FD001 ---

import pandas as pd
import numpy as np

# Load FD001 dataset
df_fd001 = pd.read_csv("../data/clustered_train_FD001.csv")

# Tolerance for coincidence
tolerance = 0.01

# Sensor columns
sensor_columns_fd001 = [col for col in df_fd001.columns if 'sensor_' in col]

print("===== FD001 Coinciding Sensors and Stages =====")
for sensor in sensor_columns_fd001:
    stage_means = df_fd001.groupby('kmeans_stage')[sensor].mean().sort_index()
    coinciding_stages = []
    stages = stage_means.index.tolist()

    for i in range(len(stages)):
        for j in range(i+1, len(stages)):
            if abs(stage_means.loc[stages[i]] - stage_means.loc[stages[j]]) < tolerance:
                coinciding_stages.append((stages[i], stages[j]))

    if coinciding_stages:
        print(f"⚡ Sensor {sensor} → coinciding stages: {coinciding_stages}")


# ## FD001 - Manual Relabeling and Correction
# 
# After analyzing the sensor behavior and coinciding stages,  
# we found that **Stage 3 and Stage 4 were wrongly ordered**.
# 
# Hence, we manually swap Stage 3 and Stage 4 to restore a logical degradation sequence:  
# **Stage 0 → Stage 1 → Stage 2 → Stage 3 → Stage 4**
# 
# The corrected FD001 dataset is saved as `corrected_clustered_train_FD001.csv`.
# 

# In[7]:


# --- FD001 Correction (Full) ---

# Step 0: Import pandas
import pandas as pd

# Step 1: Load the clustered FD001 data
df_fd001 = pd.read_csv("../data/clustered_train_FD001.csv")

# Step 2: Create a copy to avoid modifying the original dataframe
df_corrected = df_fd001.copy()

# Step 3: Temporarily relabel Stage 3 to 100
df_corrected['kmeans_stage'] = df_corrected['kmeans_stage'].replace({3: 100})

# Step 4: Relabel Stage 4 to Stage 3
df_corrected['kmeans_stage'] = df_corrected['kmeans_stage'].replace({4: 3})

# Step 5: Relabel temporary 100 (old Stage 3) to Stage 4
df_corrected['kmeans_stage'] = df_corrected['kmeans_stage'].replace({100: 4})

# Step 6: Save corrected file
corrected_data_path = "../data/corrected_clustered_train_FD001.csv"
df_corrected.to_csv(corrected_data_path, index=False)

print("✅ Corrected FD001 clustered data saved successfully!")


# ## FD002 - Top Sensor Selection and Cluster Behavior Plotting
# We load the clustered FD002 data, calculate variance for all sensors, select the top 5, and plot their behavior across KMeans stages to verify degradation patterns.

# In[11]:


# --- FD002 - Cluster Verification (KMeans Stage) ---

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plot style
sns.set(style="whitegrid")

# Load clustered FD002 data
data_path_fd002 = "../data/clustered_train_FD002.csv"
df_fd002 = pd.read_csv(data_path_fd002)

# Show available columns to verify
print(df_fd002.columns)

# --- Step 1: Variance Calculation for FD002 ---

# List all sensor columns
sensor_columns_fd002 = [col for col in df_fd002.columns if 'sensor_' in col]

# Calculate variance
variances_fd002 = df_fd002[sensor_columns_fd002].var().sort_values(ascending=False)
print("Top 5 variance sensors for FD002:")
print(variances_fd002.head(5))

# --- Step 2: (After seeing printed output) Update Top Sensors List ---

# (Placeholder for now - after running you will update this manually)
top_sensors_fd002 = ['sensor_16', 'sensor_1', 'sensor_19', 'sensor_13', 'sensor_12']

# --- Step 3: Define Plotting Function (Perfect) ---

def plot_sensor_behavior_by_stage(df, top_sensors, dataset_code):
    """
    Plots mean sensor behavior grouped by kmeans_stage for a given dataset.

    Args:
        df (DataFrame): Clustered dataset (FD001/FD002/FD003/FD004)
        top_sensors (list): List of top 5 variance sensor names
        dataset_code (str): 'FD001', 'FD002', etc.
    """
    cycle_column = 'time'  # Confirmed from data

    output_dir = f"../figures/manual_cluster_verification/{dataset_code}/"
    os.makedirs(output_dir, exist_ok=True)

    for sensor in top_sensors:
        plt.figure(figsize=(10,6))
        sns.lineplot(
            x=cycle_column,
            y=sensor,
            hue='kmeans_stage',
            data=df,
            palette='tab10'
        )
        plt.title(f"{sensor} behavior across KMeans Cluster Stages - {dataset_code}")
        plt.xlabel("Cycle Number (Time)")
        plt.ylabel(f"{sensor} Reading (Normalized)")
        plt.legend(title="KMeans Stage")
        plt.grid(True)
        plt.savefig(f"{output_dir}{sensor}_kmeans_stage_{dataset_code}.png")
        plt.close()

    print(f"✅ Sensor behavior plots saved for {dataset_code}!")

# --- Step 4: Plot Sensor Behavior for FD002 ---

# (Only after updating correct top_sensors_fd002 list)
plot_sensor_behavior_by_stage(df_fd002, top_sensors_fd002, "FD002")


# ## FD002 - Coinciding Stage Detection
# We check if any stages have similar sensor readings, indicating wrong or merged clustering. If stages are too close, manual relabeling is needed.

# In[16]:


# --- Coinciding Stage Detection for FD002 ---

import pandas as pd
import numpy as np

# Load FD002 dataset
df_fd002 = pd.read_csv("../data/clustered_train_FD002.csv")

# Tolerance for coincidence
tolerance = 0.01

# Sensor columns
sensor_columns_fd002 = [col for col in df_fd002.columns if 'sensor_' in col]

print("===== FD002 Coinciding Sensors and Stages =====")
for sensor in sensor_columns_fd002:
    stage_means = df_fd002.groupby('kmeans_stage')[sensor].mean().sort_index()
    coinciding_stages = []
    stages = stage_means.index.tolist()

    for i in range(len(stages)):
        for j in range(i+1, len(stages)):
            if abs(stage_means.loc[stages[i]] - stage_means.loc[stages[j]]) < tolerance:
                coinciding_stages.append((stages[i], stages[j]))

    if coinciding_stages:
        print(f"⚡ Sensor {sensor} → coinciding stages: {coinciding_stages}")


# ## FD002 - Relabeling Stages Based on Sensor Behavior
# Since Stage 0, 1, 3, and 4 overlap heavily across many sensors, we merge them into a new Healthy Stage (0) and keep Stage 2 as new Degradation Stage (1).

# In[18]:


# --- Relabel FD002 based on sensor review ---
import pandas as pd

# Load clustered FD002 data
df_fd002 = pd.read_csv("../data/clustered_train_FD002.csv")

# Create a corrected copy
df_fd002_corrected = df_fd002.copy()

# Merge Stage 0,1,3,4 → into Stage 0 (new healthy stage)
df_fd002_corrected['kmeans_stage'] = df_fd002_corrected['kmeans_stage'].replace({
    1: 0,
    3: 0,
    4: 0
})

# Change Stage 2 → to Stage 1 (new degradation stage)
df_fd002_corrected['kmeans_stage'] = df_fd002_corrected['kmeans_stage'].replace({
    2: 1
})

# Save corrected file
corrected_data_path_fd002 = "../data/corrected_clustered_train_FD002.csv"
df_fd002_corrected.to_csv(corrected_data_path_fd002, index=False)

print("✅ FD002 relabeling completed and saved successfully!")


# ## 1. Plotting Sensor Behavior
# 
# We plotted the behavior of the top 5 variance sensors for FD003:
# 
# - `sensor_4`
# - `sensor_7`
# - `sensor_11`
# - `sensor_12`
# - `sensor_17`
# 
# Each plot shows how the sensor readings evolve across engine cycles and cluster stages assigned by KMeans.
# 

# In[4]:


# --- FD003 - Cluster Verification (KMeans Stage) ---

# Import necessary libraries (only if needed again)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plot style
sns.set(style="whitegrid")

# Load clustered FD003 data
data_path_fd003 = "../data/clustered_train_FD003.csv"
df_fd003 = pd.read_csv(data_path_fd003)

# Show available columns
print(df_fd003.columns)

# --- Find Top 5 Variance Sensors for FD003 ---
sensor_columns_fd003 = [col for col in df_fd003.columns if 'sensor_' in col]
variances_fd003 = df_fd003[sensor_columns_fd003].var().sort_values(ascending=False)
print("\nTop 5 variance sensors for FD003:")
print(variances_fd003.head(5))

# --- Define top sensors based on actual variance ---
top_sensors_fd003 = variances_fd003.head(5).index.tolist()

# --- Plot Sensor Behavior for FD003 ---
def plot_sensor_behavior_by_stage(df, top_sensors, dataset_code):
    cycle_column = 'time'
    output_dir = f"../figures/manual_cluster_verification/{dataset_code}/"
    os.makedirs(output_dir, exist_ok=True)

    for sensor in top_sensors:
        plt.figure(figsize=(10,6))
        sns.lineplot(
            x=cycle_column,
            y=sensor,
            hue='kmeans_stage',
            data=df,
            palette='tab10'
        )
        plt.title(f"{sensor} behavior across KMeans Cluster Stages - {dataset_code}")
        plt.xlabel("Cycle Number (Time)")
        plt.ylabel(f"{sensor} Reading (Normalized)")
        plt.legend(title="KMeans Stage")
        plt.grid(True)
        plt.savefig(f"{output_dir}{sensor}_kmeans_stage_{dataset_code}.png")
        plt.close()

    print(f"✅ Sensor behavior plots saved for {dataset_code}!")

# Plotting
plot_sensor_behavior_by_stage(df_fd003, top_sensors_fd003, "FD003")

# --- Save Corrected FD003 Clustered Dataset ---
# Save corrected FD003 clustered data (even though no relabeling needed)

corrected_data_path_fd003 = "../data/corrected_clustered_train_FD003.csv"
df_fd003.to_csv(corrected_data_path_fd003, index=False)

print("✅ Corrected FD003 clustered data saved successfully!")


# ## 2. Observations
# 
# - The degradation progression across stages shows a clear and logical increasing or decreasing pattern.
# - No major stage inversion or severe overlap is observed for top sensors.
# - Coincidence was detected only in some lower-variance sensors (`sensor_5`, `sensor_6`, etc.), but **not** in top 5 sensors.
# 

# ## 3. Correction Done
# 
# - No relabeling was necessary for FD003.
# - We saved the `corrected_clustered_train_FD003.csv` after confirming that stage progression was already proper.
# - The proper health degradation sequence remains:  
#   **Stage 0 → Stage 1 → Stage 2 → Stage 3 → Stage 4**  
#   (Normal → Minor Degradation → Moderate → Severe → Failure)
# 
# ✅ FD003 verified and corrected dataset saved successfully!
# 

# In[19]:


# --- Coinciding Stage Detection for FD003 ---

import pandas as pd
import numpy as np

# Load FD003 dataset
df_fd003 = pd.read_csv("../data/clustered_train_FD003.csv")

# Tolerance for coincidence
tolerance = 0.01

# Sensor columns
sensor_columns_fd003 = [col for col in df_fd003.columns if 'sensor_' in col]

print("===== FD003 Coinciding Sensors and Stages =====")
for sensor in sensor_columns_fd003:
    stage_means = df_fd003.groupby('kmeans_stage')[sensor].mean().sort_index()
    coinciding_stages = []
    stages = stage_means.index.tolist()

    for i in range(len(stages)):
        for j in range(i+1, len(stages)):
            if abs(stage_means.loc[stages[i]] - stage_means.loc[stages[j]]) < tolerance:
                coinciding_stages.append((stages[i], stages[j]))

    if coinciding_stages:
        print(f"⚡ Sensor {sensor} → coinciding stages: {coinciding_stages}")


# ## FD004 - Top Sensor Selection and Cluster Behavior Plotting
# We load the clustered FD004 data, calculate variance for all sensors, select the top 5 based on variance, and plot their behavior across KMeans stages to observe the sensor trends and validate cluster separations.
# 

# In[ ]:


# --- FD004 - Cluster Verification (KMeans Stage) ---

# Import necessary libraries (if not already imported earlier)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plot style
sns.set(style="whitegrid")

# Load clustered FD004 data
data_path_fd004 = "../data/clustered_train_FD004.csv"
df_fd004 = pd.read_csv(data_path_fd004)

# Show available columns
print(df_fd004.columns)

# --- Find Top 5 Variance Sensors for FD004 ---
sensor_columns_fd004 = [col for col in df_fd004.columns if 'sensor_' in col]
variances_fd004 = df_fd004[sensor_columns_fd004].var().sort_values(ascending=False)
print("\nTop 5 variance sensors for FD004:")
print(variances_fd004.head(5))

# --- Define top sensors based on actual variance ---
top_sensors_fd004 = variances_fd004.head(5).index.tolist()

# --- Plot Sensor Behavior for FD004 ---
def plot_sensor_behavior_by_stage(df, top_sensors, dataset_code):
    cycle_column = 'time'
    output_dir = f"../figures/manual_cluster_verification/{dataset_code}/"
    os.makedirs(output_dir, exist_ok=True)

    for sensor in top_sensors:
        plt.figure(figsize=(10,6))
        sns.lineplot(
            x=cycle_column,
            y=sensor,
            hue='kmeans_stage',
            data=df,
            palette='tab10'
        )
        plt.title(f"{sensor} behavior across KMeans Cluster Stages - {dataset_code}")
        plt.xlabel("Cycle Number (Time)")
        plt.ylabel(f"{sensor} Reading (Normalized)")
        plt.legend(title="KMeans Stage")
        plt.grid(True)
        plt.savefig(f"{output_dir}{sensor}_kmeans_stage_{dataset_code}.png")
        plt.close()

    print(f"✅ Sensor behavior plots saved for {dataset_code}!")

# --- Step 1: Plotting ---
plot_sensor_behavior_by_stage(df_fd004, top_sensors_fd004, "FD004")

# --- Step 2: Save Corrected FD004 Clustered Dataset ---

corrected_data_path_fd004 = "../data/corrected_clustered_train_FD004.csv"
df_fd004.to_csv(corrected_data_path_fd004, index=False)

print("✅ Corrected FD004 clustered data saved successfully!")


# ## FD004 - Coinciding Stage Detection
# We check if any KMeans stages have very similar average sensor readings. This helps detect cases where multiple stages are statistically indistinguishable and might need merging through manual relabeling.
# 

# In[20]:


# --- Coinciding Stage Detection for FD004 ---

import pandas as pd
import numpy as np

# Load FD004 dataset
df_fd004 = pd.read_csv("../data/clustered_train_FD004.csv")

# Tolerance for coincidence
tolerance = 0.01

# Sensor columns
sensor_columns_fd004 = [col for col in df_fd004.columns if 'sensor_' in col]

print("===== FD004 Coinciding Sensors and Stages =====")
for sensor in sensor_columns_fd004:
    stage_means = df_fd004.groupby('kmeans_stage')[sensor].mean().sort_index()
    coinciding_stages = []
    stages = stage_means.index.tolist()

    for i in range(len(stages)):
        for j in range(i+1, len(stages)):
            if abs(stage_means.loc[stages[i]] - stage_means.loc[stages[j]]) < tolerance:
                coinciding_stages.append((stages[i], stages[j]))

    if coinciding_stages:
        print(f"⚡ Sensor {sensor} → coinciding stages: {coinciding_stages}")


# ## FD004 - Relabeling Coinciding Stages
# Based on the detected coinciding stages, we manually relabel KMeans clusters to merge stages that behave similarly. Specific relabeling rules are applied separately for sensor_13, sensor_16, and sensor_19 to improve clustering quality.

# In[9]:


# --- Relabeling Coinciding Stages for FD004 ---

def relabel_fd004_stages(y_pred, sensor_name):
    y_pred_new = y_pred.copy()

    if sensor_name in ['sensor_13', 'sensor_19']:
        # Merge Stage 0, 1, 2, 4 into one stage (say 0), keep Stage 3 as 1
        y_pred_new = np.where(np.isin(y_pred, [0,1,2,4]), 0, y_pred)  # 0,1,2,4 → 0
        y_pred_new = np.where(y_pred == 3, 1, y_pred_new)              # 3 → 1

    if sensor_name == 'sensor_16':
        # Merge Stage 0, 3, 4 into one stage (say 0), and Stage 1,2 into another (say 1)
        y_pred_new = np.where(np.isin(y_pred, [0,3,4]), 0, y_pred)  # 0,3,4 → 0
        y_pred_new = np.where(np.isin(y_pred, [1,2]), 1, y_pred_new) # 1,2 → 1

    return y_pred_new

# Example usage:
# y_pred_sensor_13 = relabel_fd004_stages(y_pred_sensor_13, 'sensor_13')
# y_pred_sensor_16 = relabel_fd004_stages(y_pred_sensor_16, 'sensor_16')
# y_pred_sensor_19 = relabel_fd004_stages(y_pred_sensor_19, 'sensor_19')

