# /scripts — Project Automation Scripts

This folder contains production-ready Python scripts that were automatically converted from Jupyter notebooks using `nbconvert`.  
These scripts serve the following important purposes:

- Enable command-line execution (for example, `python 01_data_cleaning.py`).
- Allow easy integration into larger automated workflows or batch processing pipelines.
- Provide a backup of the core project logic outside the `.ipynb` notebook format.
- Facilitate faster execution, easier tracking of changes, and better version control using Git.

The scripts directly mirror the phases implemented in the `/notebooks` directory, but they are streamlined and structured for automation and reproducibility.

---

## How These Scripts Were Generated

Each script was generated using the following command in the terminal:

```bash
jupyter nbconvert --to script notebooks/<notebook_name>.ipynb --output-dir="scripts" --output="<script_name>"
```

For example:

```bash
jupyter nbconvert --to script notebooks/01_data_cleaning.ipynb --output-dir="scripts" --output="01_data_cleaning"
```

This ensures that every phase of the project has a corresponding `.py` file available for clean execution.

---

## Available Scripts

| Script | Description |
|--------|-------------|
| `01_data_cleaning.py` | Phase 1: Cleaning and normalizing the raw CMAPSS datasets |
| `02_clustering.py` | Phase 2: Performing clustering (KMeans, Agglomerative) and assigning degradation stages |
| `02_5_cluster_verification.py` | Phase 2.5: Manually verifying cluster labels, detecting coinciding stages, and relabeling where necessary |

---

## Usage

To execute any script directly from the terminal:

```bash
python 01_data_cleaning.py
python 02_clustering.py
python 02_5_cluster_verification.py
```

Make sure that the current working directory is properly set so that relative paths to the `/data/`, `/figures/`, and `/report/` folders work correctly.

---

## Why Have Separate Scripts?

- **Reproducibility**: Anyone can reproduce the key phases without needing Jupyter Notebook.
- **Automation**: The scripts can be used in continuous training pipelines without manual intervention.
- **Version Control**: Easier to track changes in code without including heavy notebook metadata.
- **Backup**: Ensures that the main logic is preserved even if notebooks change during experimentation.
