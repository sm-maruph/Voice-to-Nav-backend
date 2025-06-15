from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ NEW: Import CORS
import pickle

# Load model and vectorizer
with open("nlu_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# Your intent-response mapping
intent_actions = {
    "doctors": "ডাক্তার দেখাও",
    "appointment": "অ্যাপয়েন্টমেন্ট দিন",
    "prescriptions": "প্রেসক্রিপশন দেখাও",
    "medicines": "ঔষধ তালিকা",
    "my_report": "আমার রিপোর্ট",
    "my_booking": "আমার বুকিং",
    "back": "পিছনে যাও"
}

app = Flask(__name__)
CORS(app)  # ✅ NEW: Enable CORS for all routes

@app.route("/predict", methods=["POST"])
def predict_intent():
    data = request.get_json()
    message = data.get("text", "")

    if not message:
        return jsonify({"error": "No input provided"}), 400

    vector = vectorizer.transform([message])
    prediction = model.predict(vector)[0]
    print(f"🧠 Received: {message} → Predicted intent: {prediction}")


    return jsonify({
        "intent": prediction,
        "message": intent_actions.get(prediction, "Unknown command")
    })

if __name__ == "__main__":
    app.run(debug=True, port=5005)

    
