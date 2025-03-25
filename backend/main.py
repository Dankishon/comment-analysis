from flask import Flask, jsonify, request, render_template, send_from_directory
from process_data import load_json, analyze_sentiments, classify_texts, train_model, save_model
import os

# Создание Flask-приложения
app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

MODEL_PATH = "data/model.pkl"
VECTORIZER_PATH = "data/vectorizer.pkl"

# Глобальные переменные для модели и векторизатора
model = None
vectorizer = None

# ✅ Загружаем модель при старте
def load_model_on_startup():
    global model, vectorizer
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        print("🔁 Загружаем существующую модель и векторизатор...")
        model, vectorizer = train_model(training_file=None, model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH)
    else:
        print("⚠️ Модель не найдена. Обучите модель через API /api/train")

load_model_on_startup()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/api/data', methods=['GET'])
def get_data():
    try:
        data = load_json("data/test_vk_post_results.json")
        processed_data = analyze_sentiments(data)
        return jsonify(processed_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/classify', methods=['POST'])
def classify():
    try:
        global model, vectorizer
        request_data = request.get_json()
        texts = request_data.get("texts", [])
        if not texts:
            return jsonify({"error": "No texts provided"}), 400

        classifications = classify_texts(texts, model, vectorizer)
        return jsonify(classifications)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train():
    try:
        global model, vectorizer

        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty file name"}), 400

        file_path = os.path.join("data", file.filename)
        file.save(file_path)

        model, vectorizer = train_model(training_file=file_path)
        save_model(model, vectorizer, MODEL_PATH, VECTORIZER_PATH)

        return jsonify({"message": "Model trained successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(os.path.join(app.root_path, '../frontend/static'), path)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
