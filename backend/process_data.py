import json
import os
import pickle
import matplotlib
matplotlib.use("Agg")  # ✅ Без GUI (важно для macOS/серверов)
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from deeppavlov import build_model, configs
from deeppavlov.core.commands.utils import parse_config
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier

# === ИНИЦИАЛИЗАЦИЯ МОДЕЛИ DEEPPAVLOV ===
config_path = configs.classifiers.rusentiment_bert
config = parse_config(config_path)
config["device"] = "cpu"
sentiment_model = build_model(config_path, download=True)

# === ОБУЧЕНИЕ КЛАССИФИКАТОРА ===
def train_model(training_file, model_path=None, vectorizer_path=None):
    data = load_training_data(training_file)
    texts = [entry['text'] for entry in data]
    labels = [entry['label'] for entry in data]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    model = CatBoostClassifier(verbose=0)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Classification report:\n", classification_report(y_test, predictions))

    # Сохраняем модель, если указано
    if model_path and vectorizer_path:
        save_model(model, vectorizer, model_path, vectorizer_path)

    return model, vectorizer

def save_model(model, vectorizer, model_path, vectorizer_path):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

def load_model(model_path, vectorizer_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# === ЗАГРУЗКА И СОХРАНЕНИЕ ДАННЫХ ===
def load_json(file_path):
    with open(file_path, encoding="utf-8") as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def load_training_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

# === АНАЛИЗ ТОНАЛЬНОСТИ ===
def analyze_sentiment(text):
    predictions = sentiment_model([text])
    sentiment_label = predictions[0][0]
    sentiment_confidence = predictions[1][0] if len(predictions) > 1 else None
    return sentiment_label, sentiment_confidence

def analyze_sentiments(data):
    sentiment_mapping = {"p": "positive", "n": "negative", "neu": "neutral"}
    processed_data = []
    for entry in data:
        text = entry.get("text", "")
        if text:
            sentiment_label, sentiment_score = analyze_sentiment(text)
            sentiment_label = sentiment_mapping.get(sentiment_label, "neutral")
            entry["sentiment_label"] = sentiment_label
            entry["sentiment_score"] = sentiment_score
        processed_data.append(entry)
    return processed_data

def analyze_test_data(input_file, output_file):
    data = load_json(input_file)
    processed_data = analyze_sentiments(data["items"])
    save_json(processed_data, output_file)

# === КЛАССИФИКАЦИЯ ТЕКСТОВ С ПОМОЩЬЮ ОБУЧЕННОЙ МОДЕЛИ ===
model, vectorizer = None, None

def classify_texts(texts, loaded_model=None, loaded_vectorizer=None):
    global model, vectorizer
    if loaded_model and loaded_vectorizer:
        model = loaded_model
        vectorizer = loaded_vectorizer

    if model is None or vectorizer is None:
        model, vectorizer = train_model("data/training_data.json")

    X = vectorizer.transform(texts)
    predictions = model.predict(X)
    return [{"text": text, "predicted_label": pred} for text, pred in zip(texts, predictions)]

# === ОБЛАКО СЛОВ ===
def generate_wordcloud(df, sentiment_label, output_dir="../frontend/static"):
    texts = df[df["sentiment_label"] == sentiment_label]["text"].dropna().tolist()
    if not texts:
        return None

    combined_text = " ".join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color="white", collocations=False).generate(combined_text)

    # 📁 Убедимся, что папка существует
    os.makedirs(output_dir, exist_ok=True)

    filename = f"wordcloud_{sentiment_label}.png"
    filepath = os.path.join(output_dir, filename)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(filepath)
    plt.close()

    return filename

