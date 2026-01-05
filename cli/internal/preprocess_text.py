from nltk.stem import PorterStemmer
from internal import config
import string

stemmer = PorterStemmer()

def lower_case(text: str) -> str:
    return text.lower()
    
def remove_punctuation(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)

def tokenize(text: str) -> list[str]:
    return text.split()

def remove_stop_words(tokens: list[str]) -> list[str]:
    res = []
    with open(config.STOP_WORDS_FILE_PATH, "r") as file:
        stop_words = file.read()
    stop_words_set = set(stop_words.splitlines())
    for t in tokens:
        if t not in stop_words_set:
            res.append(t)
    return res

def stem_tokens(tokens: list[str]) -> list[str]:
    res = []
    for t in tokens:
        res.append(stemmer.stem(t))
    return res

def preprocess_text(text: str) -> list[str]:
    return stem_tokens(remove_stop_words(tokenize(remove_punctuation(lower_case(text)))))