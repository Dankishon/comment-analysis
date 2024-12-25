import json
from deeppavlov import build_model, configs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier
from deeppavlov.core.commands.utils import parse_config

def train_model(training_file):
    # Загрузка обучающих данных
    data = load_training_data(training_file)
    texts = [entry['text'] for entry in data]
    labels = [entry['label'] for entry in data]

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

# Загрузите конфигурацию
config_path = configs.classifiers.rusentiment_bert
config = parse_config(config_path)

# Установите использование CPU
config["device"] = "cpu"

# Инициализация модели
sentiment_model = build_model(config_path, download=True)

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
    print("Predictions:", predictions)  # Отладочный вывод
    sentiment_label = predictions[0][0]  # Метка тональности (p, n, neu)
    
    # Проверка и обработка вероятности
    sentiment_confidence = predictions[1][0] if len(predictions) > 1 else None
    
    return sentiment_label, sentiment_confidence

def analyze_sentiments(data):
    """
    Добавляет метку тональности и уверенность модели к каждому элементу данных.
    """
    sentiment_mapping = {"p": "positive", "n": "negative", "neu": "neutral"}
    processed_data = []
    for entry in data:
        text = entry.get("text", "")
        if text:
            # Анализ тональности
            sentiment_label, sentiment_score = analyze_sentiment(text)
            # Преобразование метки в формат, ожидаемый дашбордом
            sentiment_label = sentiment_mapping.get(sentiment_label, "neutral")
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

def load_training_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

# Инициализация обученной модели
model, vectorizer = None, None

def classify_texts(texts):
    global model, vectorizer
    if model is None or vectorizer is None:
        model, vectorizer = train_model("data/training_data.json")

    X = vectorizer.transform(texts)
    predictions = model.predict(X)

    return [{"text": text, "predicted_label": pred} for text, pred in zip(texts, predictions)]

def analyze_test_data(input_file, output_file):
    """
    Загружает данные из входного файла, анализирует их и сохраняет результаты в выходной файл.
    """
    data = load_json(input_file)
    processed_data = analyze_sentiments(data["items"])
    save_json(processed_data, output_file)
