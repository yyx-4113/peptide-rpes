"""config.py — path configuration for the peptide-rpes reproduction package.

All paths default to directories inside this repository. Override them with
environment variables if you keep the data / models elsewhere:

    PEPRPES_DATA_DIR   directory containing merged_peptide_library.csv
    PEPRPES_MODEL_DIR  directory for cached PCA latent etc.
    PEPRPES_OUT_DIR    directory for output JSON / arrays
"""
import os

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.environ.get("PEPRPES_DATA_DIR", os.path.join(REPO_DIR, "data"))
MODEL_DIR = os.environ.get("PEPRPES_MODEL_DIR", os.path.join(REPO_DIR, "models"))
OUT_DIR = os.environ.get("PEPRPES_OUT_DIR", os.path.join(REPO_DIR, "output"))

PEPTIDE_CSV = os.path.join(DATA_DIR, "merged_peptide_library.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
