import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import requests

# Создание Dash-приложения
app = dash.Dash(__name__)
app.title = "Text Analysis Dashboard"

# URL для API
API_URL = "http://127.0.0.1:5000/api/data"

# Макет приложения
app.layout = html.Div([
    html.H1("Text Analysis Dashboard"),
    html.P("This dashboard helps analyze sentiment in social media texts."),

    # Кнопка для обновления данных
    html.Button("Refresh Data", id="refresh-button", n_clicks=0),

    # Вывод графиков и таблиц
    html.Div(id="dashboard-content")
])

# Функция для обновления данных
@app.callback(
    Output("dashboard-content", "children"),
    Input("refresh-button", "n_clicks")
)
def update_dashboard(n_clicks):
    try:
        # Запрос данных из API
        response = requests.get(API_URL)
        response.raise_for_status()  # Проверка статуса ответа

        data = response.json()
        
        if not isinstance(data, list):
            raise ValueError("Expected a list of dictionaries from API.")

        # Преобразование данных в DataFrame
        df = pd.DataFrame(data)

        # Проверка необходимых колонок
        required_columns = ["text", "sentiment_label"]
        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"Missing required column: {column}")

        # Фильтрация данных по тональностям
        positive_texts = df[df["sentiment_label"] == "positive"]
        neutral_texts = df[df["sentiment_label"] == "neutral"]
        negative_texts = df[df["sentiment_label"] == "negative"]

        # Построение графиков
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

    except requests.exceptions.RequestException as e:
        return html.Div([
            html.H3("Error fetching data from API"),
            html.P(str(e))
        ])
    except Exception as e:
        return html.Div([
            html.H3("An error occurred"),
            html.P(str(e))
        ])

# Запуск приложения
if __name__ == "__main__":
    app.run_server(debug=True)
