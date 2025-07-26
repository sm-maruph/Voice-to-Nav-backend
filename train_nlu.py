import os
import json
import random
import re
import dill
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from googletrans import Translator

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print("\n🔁 [INFO] Starting NLU training pipeline...")

# === Step 1: Load Data ===
with open("intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts, labels = [], []
for intent in data["intents"]:
    for pat in intent["patterns"]:
        texts.append(pat)
        labels.append(intent["tag"])

print(f"[INFO] Loaded {len(texts)} samples across {len(set(labels))} intent classes.")
print(f"[INFO] Class distribution before balancing: {dict(Counter(labels))}")

# === Step 2: Balance Data (Duplication + Back-Translation) ===
translator = Translator()
max_count = max(Counter(labels).values())

def back_translate(text):
    try:
        en = translator.translate(text, src='bn', dest='en').text
        bn = translator.translate(en, src='en', dest='bn').text
        time.sleep(1)  # prevent API throttling
        return bn
    except Exception:
        return text  # fallback

tagged_texts = {}
for t, l in zip(texts, labels):
    tagged_texts.setdefault(l, []).append(t)

aug_texts, aug_labels = [], []
for tag, samples in tagged_texts.items():
    cur = samples.copy()
    while len(cur) < max_count:
        sample = random.choice(samples)
        aug = back_translate(sample) if random.random() < 0.5 else sample
        cur.append(aug)
    aug_texts += cur
    aug_labels += [tag] * len(cur)

print(f"[INFO] Class distribution after balancing: {dict(Counter(aug_labels))}")

# === Step 3: Bengali Normalizer and Tokenizer ===
factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("bn")

def bn_preprocess(text):
    text = normalizer.normalize(text)
    text = text.lower()
    text = re.sub(r"[^\u0980-\u09FF\s]", "", text)
    return text

def bn_tokenizer(text):
    return indic_tokenize.trivial_tokenize(text)

# === Step 4: Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    aug_texts, aug_labels, test_size=0.2, random_state=42, stratify=aug_labels
)
print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")

# === Step 5: Build Pipeline ===
# Combines preprocessing via TfidfVectorizer and LogisticRegression model
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        preprocessor=bn_preprocess,
        tokenizer=bn_tokenizer,
        ngram_range=(1, 2),
        max_features=5000
    )),
    ("clf", LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced"
    ))
])

# === Step 6: Train Model ===
pipeline.fit(X_train, y_train)
print("[INFO] Model trained successfully.")

# === Step 7: Evaluate Model ===
y_pred = pipeline.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Classification Report Heatmap
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

plt.figure(figsize=(10, 6))
sns.heatmap(df_report.iloc[:-3, :-1], annot=True, cmap="Blues", fmt=".2f")
plt.title("Classification Report Heatmap")
plt.tight_layout()
plt.savefig("classification_report_heatmap.png")
plt.close()
print("[INFO] Classification report heatmap saved as 'classification_report_heatmap.png'.")

# Confusion Matrix (Normalized)
cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues", xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_normalized.png")
plt.close()
print("[INFO] Normalized confusion matrix saved as 'confusion_matrix_normalized.png'.")

# === Step 8: Save Model ===
with open("nlu_pipeline_model.pkl", "wb") as f:
    dill.dump(pipeline, f)
print("\n✅ Pipeline model saved to 'nlu_pipeline_model.pkl'. Training complete.")
