# 🧠 /scripts — Project Automation Scripts

This folder contains production-ready Python scripts automatically converted from Jupyter notebooks using `nbconvert`. These scripts allow for:

- 🏃‍♂️ Command-line execution (e.g., `python 01_data_cleaning.py`)
- 🛠️ Integration into pipelines or batch jobs
- 🧾 Backup of key logic outside `.ipynb` format
- 📦 Faster execution and cleaner version control

These scripts **mirror the notebook stages** from the `/notebooks` directory but are streamlined for automation and reproducibility.

---

## 🔁 How These Scripts Were Generated

Each script was created using the following command:

```bash
jupyter nbconvert --to script notebooks/<notebook_name>.ipynb
