from pathlib import Path

# -----------------------
# Project root
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# -----------------------
# Basic functions
# -----------------------
def ensure_dir(path: Path | str):
    """
    Ensure the directory exists; create it if missing.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_file_exists(path):
    """
    Check if a file exists; raise FileNotFoundError if not.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p.resolve()}")
    return p


# -----------------------
# Folder registry (NO side effects)
# -----------------------
FOLDERS = {
    "raw": DATA_DIR / "raw_audio",
    "preprocessed": DATA_DIR / "preprocessed_audio",
    "embeddings": DATA_DIR / "embeddings",
    "outputs": DATA_DIR / "outputs",
}

# Ensure folders exist safely
for folder in FOLDERS.values():
    ensure_dir(folder)


# -----------------------
# Generic path helper
# -----------------------
def get_path(folder_type: str, filename: str) -> Path:
    """
    Get full path to a file in a predefined folder.
    """
    if folder_type not in FOLDERS:
        raise ValueError(
            f"Unknown folder type: {folder_type}. Must be one of {list(FOLDERS.keys())}"
        )

    # Prevent path duplication bugs
    filename = Path(filename).name

    return FOLDERS[folder_type] / filename
