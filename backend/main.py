import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import re
from datetime import datetime
import nltk
import os
nltk.download('stopwords')
from nltk.corpus import stopwords
import json
import pickle

# === Настройки ===
DATA_PATH = "data/test_vk_post.json"
MODEL_PATH = "data/model.pkl"
VECTORIZER_PATH = "data/vectorizer.pkl"
RUSENTILEX_PATH = "data/rusentilex_2017.txt"
EVENT_DATE = datetime(2024, 1, 15)

# Папка для сохранения PNG
OUTPUT_DIR = "../frontend/static/diagrams/"

# Проверка и создание папки
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Загрузка данных ===
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)
items = data['items']
df = pd.DataFrame(items)
df['date'] = pd.to_datetime(df['date'], unit='s')

# === Загрузка модели CatBoost из pickle ===
with open(MODEL_PATH, 'rb') as f:
    cat_model = pickle.load(f)
vectorizer = pd.read_pickle(VECTORIZER_PATH)

# === Загрузка RuSentiLex ===
pos_words, neg_words = set(), set()
with open(RUSENTILEX_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        parts = [x.strip() for x in line.strip().split(',')]
        if len(parts) >= 4:
            lemma = parts[2].lower()
            sentiment = parts[3].lower()
            if sentiment == 'positive':
                pos_words.add(lemma)
            elif sentiment == 'negative':
                neg_words.add(lemma)

def lexicon_sentiment_label(text):
    words = re.findall(r'\w+', text.lower())
    pos_count = sum(1 for w in words if w in pos_words)
    neg_count = sum(1 for w in words if w in neg_words)
    if pos_count > neg_count:
        return 'positive'
    elif neg_count > pos_count:
        return 'negative'
    else:
        return 'neutral'

# === Гибридный анализ ===
texts = df['text'].astype(str).tolist()
X = vectorizer.transform(texts)
cat_preds = cat_model.predict(X)
df['catboost'] = np.where(cat_preds == 1, 'positive', 'negative')
df['lexicon'] = df['text'].apply(lexicon_sentiment_label)
df['final'] = df.apply(lambda row: row['catboost'] if row['catboost'] == row['lexicon'] else row['lexicon'], axis=1)

# === Тематический анализ ===
stopwords_ru = stopwords.words('russian')
vectorizer_model = CountVectorizer(stop_words=stopwords_ru)
topic_model = BERTopic(language="multilingual", vectorizer_model=vectorizer_model)
topics, probs = topic_model.fit_transform(texts)
topics_info = topic_model.get_topic_info()

plt.figure(figsize=(8,4))
top_topics = topics_info[topics_info.Topic != -1].nlargest(10, 'Count')
plt.bar(top_topics['Topic'].astype(str), top_topics['Count'])
plt.title("Распределение тем")
plt.xlabel("Тема")
plt.ylabel("Количество сообщений")
plt.savefig(f"{OUTPUT_DIR}topics_distribution.png")
plt.close()

# === Динамика тональности ===
df['sentiment_score'] = df['final'].map({'positive': 1, 'negative': -1, 'neutral': 0})
daily_sentiment = df.groupby(df['date'].dt.date)['sentiment_score'].mean()
plt.figure(figsize=(10,5))
daily_sentiment.plot()
plt.title("Динамика тональности")
plt.xlabel("Дата")
plt.ylabel("Средний скор (–1..+1)")
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}sentiment_timeline.png")
plt.close()

# === Сравнение до/после события ===
before = df[df['date'] < EVENT_DATE]
after = df[df['date'] >= EVENT_DATE]
before_pos = (before['final'] == 'positive').mean() * 100
after_pos = (after['final'] == 'positive').mean() * 100
plt.figure(figsize=(5,5))
plt.bar(['До', 'После'], [before_pos, after_pos], color=['gray','orange'])
plt.ylabel("Доля позитивных (%)")
plt.title("Сравнение до/после события")
for i, val in enumerate([before_pos, after_pos]):
    plt.text(i, val + 1, f"{val:.1f}%", ha='center', fontweight='bold')
plt.savefig(f"{OUTPUT_DIR}sentiment_before_after.png")
plt.close()

# === Word Cloud ===
all_text = " ".join(texts)
wordcloud = WordCloud(width=800, height=600, background_color="white",
                      stopwords=stopwords_ru, collocations=False).generate(all_text)
wordcloud.to_file(f"{OUTPUT_DIR}comments_wordcloud.png")

print("Анализ завершён. Результаты сохранены в папке frontend/static/diagrams/")
