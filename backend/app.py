from flask import Flask, jsonify, request, render_template
from process_data import load_json, analyze_sentiments, classify_texts

app = Flask(__name__, template_folder="../frontend/templates")


@app.route('/')
def home():
    """
    Рендер главной страницы.
    """
    return render_template("index.html")


@app.route('/api/data', methods=['GET'])
def get_data():
    """
    API для предоставления обработанных данных.
    """
    try:
        data = load_json("data/comments.json")
        processed_data = analyze_sentiments(data)
        return jsonify(processed_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/classify', methods=['POST'])
def classify():
    """
    API для классификации текста по эмоциям.
    """
    try:
        request_data = request.get_json()
        texts = request_data.get("texts", [])
        if not texts:
            return jsonify({"error": "No texts provided"}), 400

        classifications = classify_texts(texts)
        return jsonify(classifications)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_texts():
    """
    API для анализа тональности текста.
    """
    try:
        request_data = request.get_json()
        texts = request_data.get("texts", [])
        if not texts:
            return jsonify({"error": "No texts provided"}), 400

        # Анализ тональности
        processed_data = analyze_sentiments([{"text": text} for text in texts])
        return jsonify({"results": processed_data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)
