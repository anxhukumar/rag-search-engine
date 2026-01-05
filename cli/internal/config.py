from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DATA_FILE_PATH = PROJECT_ROOT/"data"/"movies.json"