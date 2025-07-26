import pickle

# Load the saved vectorizer and model
with open("nlu_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# Preprocessing and tokenizer must be same as training
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize
import re

factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("bn")

def bn_preprocess(text):
    txt = normalizer.normalize(text)
    txt = txt.lower()
    txt = re.sub(r"[^\u0980-\u09FF\s]", "", txt)
    return txt

def bn_tokenizer(text):
    return indic_tokenize.trivial_tokenize(text)

def predict_intent(text):
    # Vectorize the input text
    X = vectorizer.transform([text])
    # Predict the intent label
    pred = model.predict(X)
    return pred[0]

if __name__ == "__main__":
    print("Enter Bengali sentences to predict intent (type 'exit' to quit):")
    while True:
        text = input("Input: ")
        if text.lower() == "exit":
            break
        preprocessed = bn_preprocess(text)
        intent = predict_intent(preprocessed)
        print(f"Predicted intent: {intent}")
