# bn_utils.py
import re
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize

factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("bn")

def bn_preprocess(text):
    txt = normalizer.normalize(text)
    txt = txt.lower()
    txt = re.sub(r"[^\u0980-\u09FF\s]", "", txt)
    return txt

def bn_tokenizer(text):
    return indic_tokenize.trivial_tokenize(text)
