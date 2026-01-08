from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DATA_FILE_PATH = PROJECT_ROOT/"data"/"movies.json"
STOP_WORDS_FILE_PATH = PROJECT_ROOT/"data"/"stop_words.txt"
CACHE_FILE_PATH = PROJECT_ROOT/"cache"
INDEX_CACHE_FILE_PATH = CACHE_FILE_PATH/"index.pkl"
DOCMAP_CACHE_FILE_PATH = CACHE_FILE_PATH/"docmap.pkl"