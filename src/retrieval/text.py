"""Classical NLP for queries and recipes (technique area B).

- text normalization (lower-casing, Turkish-aware, punctuation stripping)
- regex extraction of constraints: max total time and diet
- language guess (TR / EN) and Snowball stemming per language
- a small TR->EN keyword lexicon so the lexical baseline can handle Turkish queries
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import snowballstemmer

_STEM = {
    "en": snowballstemmer.stemmer("english"),
    "tr": snowballstemmer.stemmer("turkish"),
}

_TR_CHARS = re.compile(r"[çğıöşü]")
_TR_WORDS = {
    "ve", "ile", "bir", "dakika", "dakikadan", "saat", "az", "kısa", "altında",
    "hızlı", "kolay", "tarif", "tarifi", "istiyorum", "olsun", "için", "et",
    "etsiz", "vejetaryen", "tatlı", "akşam", "yemeği", "yemek",
}

EN_STOP = {
    "a", "an", "the", "and", "or", "with", "without", "for", "of", "in", "on",
    "to", "i", "want", "something", "some", "recipe", "recipes", "please", "me",
    "make", "is", "that", "it", "my", "can", "under", "less", "than", "within",
    "minutes", "minute", "min", "mins", "hour", "hours", "max", "maximum",
}
TR_STOP = {
    "ve", "ile", "bir", "için", "bana", "istiyorum", "olsun", "tarif", "tarifi",
    "bir", "şey", "dakika", "dakikadan", "dk", "saat", "saatten", "az", "kısa",
    "altında", "en", "fazla", "içinde", "olan", "yemek", "yemeği",
}

# Minimal bilingual lexicon (TR -> EN). Applied before stemming so that the
# tf-idf baseline can match Turkish queries against English recipes.
TR_EN = {
    "pizza": "pizza", "hamburger": "hamburger", "burger": "burger",
    "suşi": "sushi", "sushi": "sushi", "ramen": "ramen", "falafel": "falafel",
    "sezar": "caesar", "salata": "salad", "salatası": "salad",
    "pankek": "pancake", "krep": "pancake", "çikolatalı": "chocolate",
    "çikolata": "chocolate", "pasta": "cake", "kek": "cake", "spagetti": "spaghetti",
    "makarna": "pasta", "bolonez": "bolognese", "guakamole": "guacamole",
    "avokado": "avocado", "tavuk": "chicken", "et": "meat", "kıyma": "ground beef",
    "balık": "fish", "somon": "salmon", "peynir": "cheese", "peynirli": "cheese",
    "sebze": "vegetable", "sebzeli": "vegetable", "nohut": "chickpea",
    "domates": "tomato", "mantar": "mushroom", "acı": "spicy", "baharatlı": "spicy",
    "kolay": "easy", "hızlı": "quick", "sağlıklı": "healthy", "hafif": "light",
    "kahvaltı": "breakfast", "tatlı": "dessert", "akşam": "dinner",
    "yumurta": "egg", "yumurtasız": "eggless", "glutensiz": "gluten free",
    "pirinç": "rice", "noodle": "noodle", "erişte": "noodle", "çorba": "soup",
    "fırın": "baked", "fırında": "baked", "kızartma": "fried",
}


@dataclass
class Constraints:
    max_minutes: int | None = None
    diet: set[str] = field(default_factory=set)   # subset of {"vegetarian", "vegan"}

    def is_empty(self) -> bool:
        return self.max_minutes is None and not self.diet


def detect_lang(text: str) -> str:
    t = text.lower()
    if _TR_CHARS.search(t):
        return "tr"
    words = set(re.findall(r"\w+", t))
    return "tr" if len(words & _TR_WORDS) >= 1 and not (words & {"the", "and", "with", "under"}) else "en"


def normalize(text: str) -> str:
    # Turkish dotted/dotless I must be handled before lower().
    text = text.replace("I", "ı").replace("İ", "i") if _TR_CHARS.search(text.lower()) else text
    text = text.lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# --- constraint extraction --------------------------------------------------

_NUM = r"(\d{1,3}(?:[.,]\d)?)"
_TIME_PATTERNS = [
    # EN: under/less than/within/max 30 minutes|min|hours
    (re.compile(rf"(?:under|less than|below|within|in|max(?:imum)?|at most|no more than)\s+{_NUM}\s*(min(?:ute)?s?|h(?:ou)?rs?|hours?)\b"), 1),
    (re.compile(rf"{_NUM}\s*(min(?:ute)?s?|h(?:ou)?rs?|hours?)\s+or less\b"), 1),
    # TR: 30 dakikadan az/kısa, 1 saatten kısa, en fazla 20 dk, 30 dk içinde/altında
    (re.compile(rf"{_NUM}\s*(dakikadan|dk(?:dan)?|saatten)\s+(?:az|kısa|önce)"), 1),
    (re.compile(rf"en fazla\s+{_NUM}\s*(dakika|dk|saat)"), 1),
    (re.compile(rf"{_NUM}\s*(dakika|dk|saat)\s+(?:içinde|altında)"), 1),
]
_QUICK = re.compile(r"\b(quick|fast|hızlı|pratik)\b")

_DIET = {
    "vegan": re.compile(r"\b(vegan|bitkisel)\b"),
    "vegetarian": re.compile(r"\b(vegetarian|veggie|meatless|no meat|vejetaryen|vejeteryan|etsiz)\b"),
}


def _to_minutes(num: str, unit: str) -> int:
    n = float(num.replace(",", "."))
    return int(round(n * 60)) if unit.startswith(("h", "saat")) else int(round(n))


def extract_constraints(text: str) -> Constraints:
    t = normalize(text)
    c = Constraints()
    for pat, _ in _TIME_PATTERNS:
        m = pat.search(t)
        if m:
            c.max_minutes = _to_minutes(m.group(1), m.group(2))
            break
    if c.max_minutes is None and _QUICK.search(t):
        c.max_minutes = 30
    for diet, pat in _DIET.items():
        if pat.search(t):
            c.diet.add(diet)
    if "vegan" in c.diet:
        c.diet.add("vegetarian")
    return c


def strip_constraints(text: str) -> str:
    """Remove the constraint phrases so that only the topical part is ranked."""
    t = normalize(text)
    for pat, _ in _TIME_PATTERNS:
        t = pat.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


# --- tokenization / stemming -------------------------------------------------

def tokenize(text: str, lang: str | None = None, translate: bool = True) -> list[str]:
    lang = lang or detect_lang(text)
    words = re.findall(r"[a-zçğıöşü]+", normalize(text))
    if lang == "tr":
        words = [w for w in words if w not in TR_STOP]
        if translate:
            out: list[str] = []
            for w in words:
                out.extend(TR_EN.get(w, w).split())
            words, lang = out, "en"
    words = [w for w in words if w not in EN_STOP and len(w) > 1]
    return _STEM[lang].stemWords(words)


def analyzer(text: str) -> list[str]:
    """Analyzer passed to sklearn's TfidfVectorizer."""
    return tokenize(text)
