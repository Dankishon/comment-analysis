from flask import Flask, jsonify, render_template
from process_data import load_json, analyze_sentiments
import os

app = Flask(__name__, template_folder="../frontend/templates")

# Путь к JSON-файлу
DATA_FILE = os.path.join(os.getcwd(), "data", "test_vk_post.json")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/data", methods=["GET"])
def get_data():
    try:
        # Загрузка и обработка данных
        df = load_json(DATA_FILE)
        df = analyze_sentiments(df)
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/classify", methods=["GET"])
def classify_data():
    texts = load_json(DATA_FILE)
    classified_data = classify_texts([entry['text'] for entry in texts])
    return jsonify(classified_data)



if __name__ == "__main__":
    app.run(debug=True)

