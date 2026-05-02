import re
from functools import lru_cache

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


STEMMER = StemmerFactory().create_stemmer()


@lru_cache(maxsize=50000)
def stem_word(word: str) -> str:
    return STEMMER.stem(word)


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(stem_word(word) for word in text.split())
