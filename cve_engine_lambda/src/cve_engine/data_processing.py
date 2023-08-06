import logging
import string
import nltk
import cvss
import numpy as np
from typing import Union
from sklearn.feature_extraction.text import CountVectorizer

log = logging.getLogger(__name__)

def desc_preprocess(d: str):
    log.debug("preprocessing description...")
    # setup
    stopwords = set(nltk.corpus.stopwords.words("english"))
    lemmatizer = nltk.stem.WordNetLemmatizer()

    # lowercase
    d = d.lower()
    # remove punctuation
    d = d.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    # tokenize
    tokens = d.split()
    # remove stop words
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def vec_parse_metric(vec: str, metric: str):
    """Given an input cvss vector and metric, extract
    the metric from the vector and return its string value."""
    return cvss.CVSS3(vec).get_value_description(metric)

def clean_cvss_vector(vec: Union[str, float]) -> Union[str, float]:
    # TODO: this is fragile; this should be a NaN check
    if type(vec) is not str: return np.nan
    try:
        return cvss.CVSS3(vec).clean_vector()
    except cvss.exceptions.CVSS3MalformedError:
        pass

    # fix common problems
    assert type(vec) is str
    vec = vec.upper()
    vec = vec.rstrip(".")
    vec = vec.replace(" ", "")
    vec = vec.rstrip("/")
    try:
        vec = "CVSS:3.1/" + vec[vec.index("AV:"):]
    except ValueError:
        pass
    # vec = vec.removeprefix("VECTOR:")
    # if vec.startswith("AV"): vec = "CVSS:3.1/" + vec
    # if vec.startswith("/AV"): vec = "CVSS:3.1" + vec

    # try again
    try:
        return cvss.CVSS3(vec).clean_vector()
    except cvss.exceptions.CVSS3MalformedError:
        return np.nan
 

def create_bow(descs: list[str]) -> tuple[CountVectorizer, np.ndarray]:
    vectorizer = CountVectorizer()
    return vectorizer, vectorizer.fit_transform(descs).toarray()