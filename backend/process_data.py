import json
from deeppavlov import build_model, configs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier


# Инициализация модели DeepPavlov
sentiment_model = build_model(configs.classifiers.rusentiment_bert, download=True)


def load_json(file_path):
    """Загружает JSON-файл и возвращает данные."""
    with open(file_path, encoding="utf-8") as file:
        return json.load(file)


def analyze_sentiment(text):
    """Анализ тональности текста с использованием DeepPavlov."""
    predictions = sentiment_model([text])
    sentiment_label = predictions[0][0]  # Метка тональности (positive, negative, neutral)
    sentiment_confidence = predictions[1][0]  # Вероятность предсказания
    return sentiment_label, sentiment_confidence


def analyze_sentiments(data):
    """Добавляет метку тональности и уверенность модели к каждому элементу данных."""
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
    """Сохраняет данные в JSON-файл."""
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def load_training_data(file_path):
    """Загружает обучающие данные."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def train_model(training_file):
    """Обучение кастомной модели на основе CatBoost."""
    # Загрузка обучающих данных
    data = load_training_data(training_file)
    texts = [entry["text"] for entry in data]
    labels = [entry["label"] for entry in data]

    # Преобразование текста в векторное представление
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    # Разделение данных на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    # Обучение модели (CatBoost)
    model = CatBoostClassifier(verbose=0)
    model.fit(X_train, y_train)

    # Оценка точности модели
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

    return model, vectorizer


# Инициализация кастомной модели (отложенная загрузка)
model, vectorizer = None, None


def classify_texts(texts):
    """Классификация текстов с помощью кастомной модели."""
    global model, vectorizer
    if model is None or vectorizer is None:
        model, vectorizer = train_model("data/training_data.json")

    X = vectorizer.transform(texts)
    predictions = model.predict(X)

    return [{"text": text, "predicted_label": pred} for text, pred in zip(texts, predictions)]
