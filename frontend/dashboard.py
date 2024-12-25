import sys
import os
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

# Добавляем путь к backend в sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from process_data import analyze_sentiments, load_json

# Создание Dash-приложения
app = dash.Dash(__name__)
app.title = "Text Analysis Dashboard"

# Путь к тестовому файлу
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEST_FILE_PATH = os.path.join(BASE_DIR, 'data', 'test_vk_post.json')

# Макет приложения
app.layout = html.Div([
    html.H1("Text Analysis Dashboard"),
    html.P("This dashboard helps analyze sentiment in social media texts."),

    # Кнопка для анализа данных
    html.Button("Analyze Test Data", id="analyze-button", n_clicks=0),

    # Вывод графиков и таблиц
    html.Div(id="dashboard-content")
])

# Функция для обновления дашборда
@app.callback(
    Output("dashboard-content", "children"),
    Input("analyze-button", "n_clicks")
)
def update_dashboard(n_clicks):
    if n_clicks > 0:
        try:
            # Загрузка и анализ тестового файла
            data = load_json(TEST_FILE_PATH)
            processed_data = analyze_sentiments(data["items"])

            # Преобразование данных в DataFrame
            df = pd.DataFrame(processed_data)

            # Проверка наличия необходимых колонок
            required_columns = ["text", "sentiment_label"]
            for column in required_columns:
                if column not in df.columns:
                    raise ValueError(f"Missing required column: {column}")

            # Фильтрация данных по тональностям
            positive_texts = df[df["sentiment_label"] == "positive"]
            neutral_texts = df[df["sentiment_label"] == "neutral"]
            negative_texts = df[df["sentiment_label"] == "negative"]

            # Построение графика распределения тональностей
            sentiment_counts = df["sentiment_label"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]
            fig = px.bar(sentiment_counts, x="Sentiment", y="Count", title="Sentiment Distribution")

            # Возвращение содержимого дашборда
            return html.Div([
                dcc.Graph(figure=fig),

                html.H3("Positive Comments"),
                html.Ul([html.Li(comment) for comment in positive_texts["text"].head(10)]),

                html.H3("Neutral Comments"),
                html.Ul([html.Li(comment) for comment in neutral_texts["text"].head(10)]),

                html.H3("Negative Comments"),
                html.Ul([html.Li(comment) for comment in negative_texts["text"].head(10)])
            ])
        except Exception as e:
            return html.Div([
                html.H3("An error occurred"),
                html.P(str(e))
            ])

    return html.Div([
        html.P("Click the button to analyze the test data.")
    ])

# Запуск приложения
if __name__ == "__main__":
    app.run_server(debug=True)
