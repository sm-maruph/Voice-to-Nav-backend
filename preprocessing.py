# preprocessing.py
import re
import string

def bn_preprocess(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Normalize Bengali characters (optional)
    # Add any other Bengali-specific preprocessing here
    return text.lower().strip()