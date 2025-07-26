import json
import pickle
import random
import re
import time
from collections import Counter
from googletrans import Translator
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Bengali preprocessing & tokenization (define before loading pickle!) ---
factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("bn")

def bn_preprocess(text):
    txt = normalizer.normalize(text)
    txt = txt.lower()
    txt = re.sub(r"[^\u0980-\u09FF\s]", "", txt)
    return txt

def bn_tokenizer(text):
    return indic_tokenize.trivial_tokenize(text)

# --- Step 2: Load model and vectorizer ---
with open("nlu_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# --- Step 3: Load data ---
with open("intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts, labels = [], []
for intent in data["intents"]:
    for pat in intent["patterns"]:
        texts.append(pat)
        labels.append(intent["tag"])

print("Original class distribution:", Counter(labels))

# --- Step 4: Data augmentation and balancing ---
translator = Translator()
max_count = max(Counter(labels).values())

def back_translate(text):
    try:
        en = translator.translate(text, src='bn', dest='en').text
        bn = translator.translate(en, src='en', dest='bn').text
        time.sleep(1)  # To avoid rate limits
        return bn
    except Exception as e:
        print(f"[DEBUG] Translation failed for '{text}': {e}")
        return text

by_tag = {}
for t, l in zip(texts, labels):
    by_tag.setdefault(l, []).append(t)

aug_texts, aug_labels = [], []
for tag, examples in by_tag.items():
    cur = examples.copy()
    while len(cur) < max_count:
        sample = random.choice(examples)
        aug = back_translate(sample) if random.random() < 0.5 else sample
        cur.append(aug)
    aug_texts += cur
    aug_labels += [tag]*len(cur)

print("Balanced class distribution:", Counter(aug_labels))

# --- Step 5: Vectorize and predict ---
X = vectorizer.transform(aug_texts)
y_pred = model.predict(X)

# --- Step 6: Report metrics ---
print("\n=== Classification Report (on normalized, augmented, balanced data) ===\n")
print(classification_report(aug_labels, y_pred))

cm = confusion_matrix(aug_labels, y_pred, labels=model.classes_)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=model.classes_,
            yticklabels=model.classes_,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_balanced.png")
plt.show()
