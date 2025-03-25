import sys
import os
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import requests
import datetime

# Добавляем путь к backend в sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from process_data import analyze_sentiments, load_json

# Создание Dash-приложения
app = dash.Dash(
    __name__,
    external_stylesheets=[
        "/frontend/static/styles.css"
    ]
)
app.title = "Text Analysis Dashboard"

# Путь к тестовому файлу
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEST_FILE_PATH = os.path.join(BASE_DIR, 'data', 'test_vk_post.json')

# URL для обучения модели
TRAIN_API_URL = "http://127.0.0.1:5000/api/train"

# Макет приложения
app.layout = html.Div([
    html.H1("Text Analysis Dashboard", style={"textAlign": "center", "color": "#3a3a3a"}),
    html.P("Analyze social media texts with sentiment analysis.", style={"textAlign": "center", "color": "#7f8c8d"}),

    # Компонент для загрузки пользовательских данных
    html.Div([
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select File")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=False
        ),
        html.Button("Start Training", id="train-button", n_clicks=0, style={
            "margin": "10px", "padding": "10px", "backgroundColor": "#6c7a89", "color": "white"
        }),
        html.Div(id="training-status", style={"textAlign": "center", "color": "#7f8c8d"})
    ]),

    html.Div([
        html.Button(
            "Analyze Test Data",
            id="analyze-button",
            n_clicks=0,
            style={"margin": "10px", "padding": "10px", "backgroundColor": "#6c7a89", "color": "white"}
        )
    ], style={"textAlign": "center"}),

    html.Div(id="dashboard-content"),

    dcc.Tabs(id="tabs", value="positive", children=[
        dcc.Tab(label="Positive Comments", value="positive", style={"color": "#6c7a89"}),
        dcc.Tab(label="Neutral Comments", value="neutral", style={"color": "#5a6b7d"}),
        dcc.Tab(label="Negative Comments", value="negative", style={"color": "#d9534f"}),
    ]),
    html.Div(id="comments-content")
])

@app.callback(
    Output("training-status", "children"),
    Input("train-button", "n_clicks"),
    State("upload-data", "contents"),
    State("upload-data", "filename")
)
def train_model(n_clicks, contents, filename):
    if n_clicks > 0 and contents:
        try:
            content_type, content_string = contents.split(",")
            response = requests.post(TRAIN_API_URL, json={"file_name": filename, "file_content": content_string})
            response.raise_for_status()
            return html.Div("Training completed successfully.", style={"color": "#28a745"})
        except requests.exceptions.RequestException as e:
            return html.Div(f"Error during training: {str(e)}", style={"color": "#d9534f"})
    return html.Div("Upload a file and click 'Start Training' to begin.")

@app.callback(
    Output("dashboard-content", "children"),
    Input("analyze-button", "n_clicks")
)
def update_dashboard(n_clicks):
    if n_clicks > 0:
        try:
            data = load_json(TEST_FILE_PATH)
            processed_data = analyze_sentiments(data["items"])
            global df
            df = pd.DataFrame(processed_data)

            # Обработка времени
            df["date"] = pd.to_datetime(df["date"], unit="s")
            df["day"] = df["date"].dt.date

            # График распределения тональностей
            sentiment_counts = df["sentiment_label"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]
            fig1 = px.bar(sentiment_counts, x="Sentiment", y="Count", title="Sentiment Distribution")

            # График лайков по дням
            df["likes"] = df["likes"].apply(lambda x: x.get("count", 0) if isinstance(x, dict) else 0)
            likes_per_day = df.groupby("day")["likes"].sum().reset_index()
            fig2 = px.line(likes_per_day, x="day", y="likes", markers=True, title="Likes Over Time")

            # График комментариев по дням
            df["comments"] = df["comments"].apply(lambda x: x.get("count", 0) if isinstance(x, dict) else 0)
            comments_per_day = df.groupby("day")["comments"].sum().reset_index()
            fig3 = px.line(comments_per_day, x="day", y="comments", markers=True, title="Comments Over Time")

            # График тональностей по дням
            sentiment_by_day = df.groupby(["day", "sentiment_label"]).size().reset_index(name="count")
            fig4 = px.line(
                sentiment_by_day,
                x="day",
                y="count",
                color="sentiment_label",
                markers=True,
                title="Sentiment Over Time"
            )

            return html.Div([
                dcc.Graph(figure=fig1),
                dcc.Graph(figure=fig2),
                dcc.Graph(figure=fig3),
                dcc.Graph(figure=fig4)
            ])
        except Exception as e:
            return html.Div([
                html.H3("An error occurred during analysis", style={"color": "#d9534f"}),
                html.P(str(e))
            ])
    return html.Div("Click the button to analyze the test data.")

@app.callback(
    Output("comments-content", "children"),
    Input("tabs", "value")
)
def display_comments(tab_value):
    if "df" not in globals():
        return html.Div("Please analyze the data first.")
    filtered_df = df[df["sentiment_label"] == tab_value]
    comments_list = filtered_df["text"].tolist()
    return html.Div([html.Ul([html.Li(comment) for comment in comments_list])])

if __name__ == "__main__":
    app.run_server(debug=True)
