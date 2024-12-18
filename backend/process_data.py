import json
from deeppavlov import build_model, configs

# Инициализация предобученной модели для анализа тональности
sentiment_model = build_model(configs.classifiers.rusentiment_bert, download=True)

def load_json(file_path):
    """
    Загружает JSON-файл и возвращает данные.
    """
    with open(file_path, encoding="utf-8") as file:
        return json.load(file)

def analyze_sentiment(text):
    """
    Анализ тональности текста с использованием DeepPavlov.
    Возвращает метку тональности и вероятность.
    """
    predictions = sentiment_model([text])
    sentiment_label = predictions[0][0]  # Метка тональности (positive, negative, neutral)
    sentiment_confidence = predictions[1][0]  # Вероятность предсказания
    return sentiment_label, sentiment_confidence

def analyze_sentiments(data):
    """
    Добавляет метку тональности и уверенность модели к каждому элементу данных.
    """
    processed_data = []
    for entry in data:
        text = entry.get("text", "")
        if text:
            # Анализ тональности
            sentiment_label, sentiment_score = analyze_sentiment(text)
            entry["sentiment_label"] = sentiment_label
            entry["sentiment_score"] = sentiment_score
        processed_data.append(entry)
    return processed_data

def save_json(data, file_path):
    """
    Сохраняет данные в JSON-файл.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
