from flask import Flask, jsonify, request, render_template, send_from_directory
from process_data import (
    load_json,
    analyze_sentiments,
    classify_texts,
    train_model,
    save_model,
    load_model
)
import os
import base64
import io
import json

# Создание Flask-приложения
app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

MODEL_PATH = "data/model.pkl"
VECTORIZER_PATH = "data/vectorizer.pkl"

# Глобальные переменные модели и векторизатора
model = None
vectorizer = None

# Загружаем модель при запуске сервера
def load_model_on_startup():
    global model, vectorizer
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        print("🔁 Загружаем существующую модель и векторизатор...")
        model, vectorizer = load_model(MODEL_PATH, VECTORIZER_PATH)
    else:
        print("⚠️ Модель не найдена. Обучите модель через интерфейс загрузки.")

load_model_on_startup()

@app.route('/')
def home():
    return render_template("index.html")

# Получение данных с автоматическим анализом тональности
@app.route('/api/data', methods=['GET'])
def get_data():
    try:
        data = load_json("data/test_vk_post_results.json")
        processed_data = analyze_sentiments(data)
        return jsonify(processed_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Классификация текстов с помощью обученной модели
@app.route('/api/classify', methods=['POST'])
def classify():
    try:
        global model, vectorizer
        request_data = request.get_json()
        texts = request_data.get("texts", [])
        if not texts:
            return jsonify({"error": "No texts provided"}), 400
        if model is None or vectorizer is None:
            return jsonify({"error": "Model not loaded"}), 500

        classifications = classify_texts(texts, model, vectorizer)
        return jsonify(classifications)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Обучение модели с новыми данными (через base64 в JSON)
@app.route('/api/train', methods=['POST'])
def train():
    try:
        global model, vectorizer

        data = request.get_json()
        if not data or 'file_content' not in data:
            return jsonify({"error": "No file content provided"}), 400

        file_content = data['file_content']
        decoded = base64.b64decode(file_content)
        training_data = json.load(io.StringIO(decoded.decode('utf-8')))

        # Сохраняем во временный файл (можно убрать при желании)
        temp_file_path = os.path.join("data", "temp_training_data.json")
        with open(temp_file_path, "w", encoding="utf-8") as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)

        model, vectorizer = train_model(training_file=temp_file_path)
        save_model(model, vectorizer, MODEL_PATH, VECTORIZER_PATH)

        return jsonify({"message": "Model trained successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(os.path.join(app.root_path, '../frontend/static'), path)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
