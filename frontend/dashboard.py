import sys
import os
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

app = dash.Dash(
    __name__,
    external_stylesheets=[
        "/frontend/static/styles.css"  # Убедитесь, что путь правильный
    ]
)


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
    html.H1("Text Analysis Dashboard", style={"textAlign": "center", "color": "#4CAF50"}),
    html.P("Analyze social media texts with sentiment analysis.", style={"textAlign": "center"}),

    # Кнопка для анализа данных
    html.Div([
        html.Button("Analyze Test Data", id="analyze-button", n_clicks=0, style={"margin": "10px", "padding": "10px", "backgroundColor": "#4CAF50", "color": "white"})
    ], style={"textAlign": "center"}),

    # Вывод графиков
    html.Div(id="dashboard-content"),

    # Вкладки для отображения списков комментариев
    dcc.Tabs(id="tabs", value="positive", children=[
        dcc.Tab(label="😊 Positive Comments", value="positive", style={"color": "green"}),
        dcc.Tab(label="😐 Neutral Comments", value="neutral", style={"color": "blue"}),
        dcc.Tab(label="😡 Negative Comments", value="negative", style={"color": "red"}),
    ]),
    html.Div(id="comments-content")
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
            global df
            df = pd.DataFrame(processed_data)

            # Проверка наличия необходимых колонок
            required_columns = ["text", "sentiment_label"]
            for column in required_columns:
                if column not in df.columns:
                    raise ValueError(f"Missing required column: {column}")

            # Построение графика распределения тональностей
            sentiment_counts = df["sentiment_label"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]
            fig = px.bar(sentiment_counts, x="Sentiment", y="Count", title="Sentiment Distribution", color="Sentiment")

            return html.Div([
                dcc.Graph(figure=fig)
            ])
        except Exception as e:
            return html.Div([
                html.H3("An error occurred during analysis", style={"color": "red"}),
                html.P(str(e))
            ])
    return html.Div([
        html.P("Click the button to analyze the test data.", style={"textAlign": "center"})
    ])

# Функция для отображения комментариев
@app.callback(
    Output("comments-content", "children"),
    Input("tabs", "value")
)
def display_comments(tab_value):
    if "df" not in globals():
        return html.Div([
            html.P("Please analyze the data first by clicking the 'Analyze Test Data' button.", style={"textAlign": "center", "color": "gray"})
        ])
    try:
        # Фильтрация данных по выбранной тональности
        filtered_df = df[df["sentiment_label"] == tab_value]
        comments_list = filtered_df["text"].tolist()

        # Генерация списка комментариев
        if not comments_list:
            return html.Div([
                html.H3(f"No comments found for {tab_value.capitalize()} sentiment", style={"textAlign": "center"})
            ])

        return html.Div([
            html.H3(f"{tab_value.capitalize()} Comments", style={"textAlign": "center"}),
            html.Ul([html.Li(comment) for comment in comments_list], style={"padding": "10px"})
        ])
    except Exception as e:
        return html.Div([
            html.H3("An error occurred while displaying comments", style={"color": "red"}),
            html.P(str(e))
        ])

# Запуск приложения
if __name__ == "__main__":
    app.run_server(debug=True)
