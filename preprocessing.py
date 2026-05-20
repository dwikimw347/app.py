import re
from functools import lru_cache

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


STEMMER = StemmerFactory().create_stemmer()
STOPWORDS = set(StopWordRemoverFactory().get_stop_words())
NEGATION_WORDS = {
    "tidak",
    "tak",
    "bukan",
    "belum",
    "jangan",
    "tanpa",
    "kurang",
    "ga",
    "gak",
    "nggak",
    "enggak",
}
STOPWORDS = STOPWORDS - NEGATION_WORDS


@lru_cache(maxsize=50000)
def stem_word(word: str) -> str:
    return STEMMER.stem(word)


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(stem_word(word) for word in tokens)
