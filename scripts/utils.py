import os


def ensure_dir(path: str):
    """
    Create directory if it does not exist.
    Safe to call multiple times.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def check_file_exists(path: str):
    """
    Check if a file exists.
    Raises FileNotFoundError if not found.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required file not found: {path}")
