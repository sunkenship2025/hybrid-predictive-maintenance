import os

# Base path: the actual directory of this script 
# or fallback to current working dir
try:
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT_PATH = os.getcwd()  # For Jupyter

# ✅ Fix: use ROOT_PATH directly as project root
BASE_DIR = os.path.abspath(os.path.join(ROOT_PATH))

# Constants
RANDOM_STATE = 42
FIGURES_DIR = os.path.join(BASE_DIR, 'figures', 'phase3')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORT_DIR = os.path.join(BASE_DIR, 'report')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Dataset files
FD001_FILE = os.path.join(DATA_DIR, 'corrected_clustered_train_FD001.csv')
FD002_FILE = os.path.join(DATA_DIR, 'corrected_clustered_train_FD002.csv')
FD003_FILE = os.path.join(DATA_DIR, 'corrected_clustered_train_FD003.csv')
FD004_FILE = os.path.join(DATA_DIR, 'corrected_clustered_train_FD004.csv')
