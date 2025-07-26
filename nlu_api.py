from flask import Flask, request, jsonify
from flask_cors import CORS
import dill, re
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize

# 🧠 Step 1: Setup Bengali Preprocessor
factory = IndicNormalizerFactory()
normalizer = factory.get_normalizer("bn")

def bn_preprocess(text):
    txt = normalizer.normalize(text)
    txt = txt.lower()
    txt = re.sub(r"[^\u0980-\u09FF\s]", "", txt)
    return txt

# 🧠 Optional (if needed for training or future model): Bengali tokenizer
def bn_tokenizer(text):
    return indic_tokenize.trivial_tokenize(text)

# 🧠 Step 2: Load Model
dill.settings['recurse'] = True
try:
    with open("nlu_model.pkl", "rb") as f:
        vectorizer, model = dill.load(f)
except Exception as e:
    print(f"❌ Model load failed: {e}")
    raise RuntimeError("Cannot start API without model")

# 🧠 Step 3: Define Intent Map (Response + Route)
intent_map = {
    "find_doctor": {
        "response": "ডাক্তার দেখাও",
        "url": "/find"
    },
    "prescriptions": {
        "response": "প্রেসক্রিপশন দেখাও",
        "url": "/dashboard/user/pres"
    },
    "home": {
        "response": "হোমপেজে ফিরে যাও",
        "url": "/"
    },
    "appointment": {
        "response": "অ্যাপয়েন্টমেন্ট দিন",
        "url": "/dashboard/user/appointment"
    },
    "medicines": {
        "response": "ঔষধ তালিকা",
        "url": "/dashboard/user/medicines"
    },
    "my_report": {
        "response": "আমার রিপোর্ট",
        "url": "/dashboard/user/report"
    },
    "my_booking": {
        "response": "আমার বুকিং",
        "url": "/dashboard/user/bookings"
    },
    "edit_profile": {
        "response": "প্রোফাইল এডিট করো",
        "url": "/dashboard/user/settings"
    },
    "join_call": {
        "response": "কল জয়েন করো",
        "url": "/dashboard/user/meeting"
    },
    "back": {
        "response": "পিছনে যাও",
        "url": "back"
    }
}


app = Flask(__name__)
CORS(app)
@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "OK", "message": "Voice-to-Nav backend is live 🚀"}), 200
# 🔍 Step 4: Predict Intent Endpoint
@app.route("/predict", methods=["POST"])
def predict_intent():
    if not request.is_json:
        return jsonify({"error": "Request must be in JSON format"}), 415

    data = request.get_json()
    message = data.get("text", "").strip()

    if not message:
        return jsonify({"error": "No input provided"}), 400

    try:
        cleaned = bn_preprocess(message)
        vector = vectorizer.transform([cleaned])
        intent = model.predict(vector)[0]

        print(f"📥 Received: '{message}' → 🧹 Cleaned: '{cleaned}' → 🎯 Intent: '{intent}'")

        intent_data = intent_map.get(intent)
        if not intent_data:
            return jsonify({
                "intent": "unknown",
                "message": "দুঃখিত, আমি বুঝতে পারিনি",
                "url": None
            })

        return jsonify({
            "intent": intent,
            "message": intent_data["response"],
            "url": intent_data["url"]
        })

    except Exception as e:
        print(f"⚠️ Server error: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
