import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# === Параметры ===
TRAIN_DATA_PATH = "data/training_data.json"
MODEL_OUTPUT_PATH = "data/model.pkl"
VECTORIZER_OUTPUT_PATH = "data/vectorizer.pkl"

# === Загрузка данных ===
df = pd.read_json(TRAIN_DATA_PATH)
X_texts = df['text'].astype(str).tolist()

# Преобразование текстовых меток в числа
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
y = df['label'].map(label_map)

# === Векторизация текста ===
vectorizer = CountVectorizer(max_features=5000)
X_vectors = vectorizer.fit_transform(X_texts)

# === Обучение CatBoost ===
model = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, verbose=100)
model.fit(X_vectors, y)

# === Сохранение модели и векторизатора ===
model.save_model("data/model.cbm")  # можно использовать .cbm
with open(MODEL_OUTPUT_PATH, 'wb') as f:
    pickle.dump(model, f)
with open(VECTORIZER_OUTPUT_PATH, 'wb') as f:
    pickle.dump(vectorizer, f)

print("Модель и векторизатор успешно сохранены.")

