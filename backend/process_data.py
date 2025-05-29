import json
import os
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier
from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import pandas as pd

# === ИНИЦИАЛИЗАЦИЯ Huggingface МОДЕЛИ ===
MODEL_NAME = "cointegrated/rubert-tiny2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# === АНАЛИЗ ТОНАЛЬНОСТИ ===
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=-1)
    pred_label = torch.argmax(probs).item()
    confidence = probs[0][pred_label].item()
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    return label_map.get(pred_label, "neutral"), confidence

def analyze_sentiments(data):
    processed_data = []
    for entry in data:
        text = entry.get("text", "")
        if text:
            sentiment_label, sentiment_score = analyze_sentiment(text)
            entry["sentiment_label"] = sentiment_label
            entry["sentiment_score"] = sentiment_score
        processed_data.append(entry)
    return processed_data

# === ОБУЧЕНИЕ И КЛАССИФИКАЦИЯ CatBoost ===
def load_training_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def train_model(training_file, model_path=None, vectorizer_path=None):
    data = load_training_data(training_file)
    texts = [entry['text'] for entry in data]
    labels = [entry['label'] for entry in data]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    model_cb = CatBoostClassifier(verbose=0)
    model_cb.fit(X_train, y_train)
    predictions = model_cb.predict(X_test)
    print("Classification report:\n", classification_report(y_test, predictions))
    if model_path and vectorizer_path:
        with open(model_path, "wb") as f:
            pickle.dump(model_cb, f)
        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)
    return model_cb, vectorizer

# === ЗАГРУЗКА/СОХРАНЕНИЕ ДАННЫХ ===
def load_json(file_path):
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# === КЛАССИФИКАЦИЯ ТЕКСТОВ ===
model_cb, vectorizer_cb = None, None
def classify_texts(texts, loaded_model=None, loaded_vectorizer=None):
    global model_cb, vectorizer_cb
    if loaded_model and loaded_vectorizer:
        model_cb = loaded_model
        vectorizer_cb = loaded_vectorizer
    if model_cb is None or vectorizer_cb is None:
        model_cb, vectorizer_cb = train_model("data/training_data.json")
    X = vectorizer_cb.transform(texts)
    predictions = model_cb.predict(X)
    return [{"text": text, "predicted_label": pred} for text, pred in zip(texts, predictions)]

# === WORDCLOUD ===
def generate_wordcloud(df, sentiment_label, output_dir="../frontend/static"):
    texts = df[df["sentiment_label"] == sentiment_label]["text"].dropna().tolist()
    if not texts:
        return None
    combined_text = " ".join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color="white", collocations=False).generate(combined_text)
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

# === BERTOPIC - ДИНАМИЧЕСКИЙ АНАЛИЗ ТЕМ ===
def get_topics_distribution(df):
    texts = df["text"].dropna().tolist()
    if not texts:
        return pd.DataFrame(columns=["Topic", "Count"])
    
    # Исправленный список стоп-слов
    russian_stop_words = text.ENGLISH_STOP_WORDS.union([
        'это', 'в', 'на', 'и', 'но', 'да', 'что', 'как', 'так', 'же', 'бы', 'за', 'по', 'из', 'у', 'о', 'к', 'с', 'со', 
        'до', 'для', 'при', 'без', 'не', 'его', 'ее', 'их', 'мы', 'вы', 'они', 'я', 'он', 'она', 'ты', 'оно', 'ли', 
        'или', 'был', 'были', 'быть', 'есть', 'нет', 'может', 'очень', 'сам', 'там', 'тут', 'тогда', 'сейчас'
    ])

    vectorizer_model = TfidfVectorizer(stop_words=russian_stop_words)
    topic_model = BERTopic(vectorizer_model=vectorizer_model, language="multilingual")
    topics, _ = topic_model.fit_transform(texts)
    topic_counts = pd.Series(topics).value_counts().reset_index()
    topic_counts.columns = ["Topic", "Count"]
    topic_counts["Name"] = topic_counts["Topic"].apply(lambda t: str(topic_model.get_topic(t)))
    return topic_counts
