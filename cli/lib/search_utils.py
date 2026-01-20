import os
from dotenv import load_dotenv
from google import genai

MODEL = "gemini-2.5-flash"

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


BM25_K1 = 1.5
BM25_B = 0.75
SEARCH_LIMIT = 5
TEXT_CHUNK_SIZE = 200
TEXT_CHUNK_OVERLAP = 0
MAX_CHUNK_SIZE = 4
ALPHA_VAL = 0.5
DEFAULT_RRF_K = 60 