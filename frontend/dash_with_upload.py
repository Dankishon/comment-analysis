import sys
import os
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import requests
import json
import io
import base64

# Добавляем путь к backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
from process_data import analyze_sentiments, generate_wordcloud

# Инициализация Dash-приложения
app = dash.Dash(__name__, external_stylesheets=["/frontend/static/styles.css"])
app.title = "Text Analysis Dashboard"

# Макет интерфейса
app.layout = html.Div([
    html.H1("Text Analysis Dashboard", style={"textAlign": "center", "color": "#3a3a3a"}),
    html.P("Analyze social media texts with sentiment analysis.", style={"textAlign": "center", "color": "#7f8c8d"}),

    html.H2("📚 Upload Training Data"),
    dcc.Upload(
        id="upload-data",
        children=html.Div(["Drag and Drop or ", html.A("Select Training File")]),
        style={
            "width": "100%", "height": "60px", "lineHeight": "60px",
            "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
            "textAlign": "center", "margin": "10px"
        },
        multiple=False
    ),
    html.Button("Start Training", id="train-button", n_clicks=0, style={
        "margin": "10px", "padding": "10px", "backgroundColor": "#6c7a89", "color": "white"
    }),
    dcc.Loading(
        id="loading-train",
        type="circle",
        children=html.Div(id="training-status"),
        color="#6c7a89"
    ),

    html.H2("📄 Upload Test File for Analysis"),
    dcc.Upload(
        id="upload-test-data",
        children=html.Div(["Drag and Drop or ", html.A("Select Test File")]),
        style={
            "width": "100%", "height": "60px", "lineHeight": "60px",
            "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "5px",
            "textAlign": "center", "margin": "10px"
        },
        multiple=False
    ),
    html.Div(id="upload-status"),
    dcc.Store(id="uploaded-test-data"),

    html.Div([
        html.Button(
            "Analyze Uploaded Data",
            id="analyze-button",
            n_clicks=0,
            style={"margin": "10px", "padding": "10px", "backgroundColor": "#6c7a89", "color": "white"}
        )
    ], style={"textAlign": "center"}),

    dcc.Loading(
        id="loading-dashboard",
        type="default",
        children=html.Div(id="dashboard-content"),
        color="#6c7a89"
    ),

    html.H2("Word Clouds", style={"textAlign": "center", "marginTop": "30px"}),
    dcc.Loading(
        id="loading-clouds",
        type="default",
        children=html.Div(id="wordclouds", style={"display": "flex", "justifyContent": "center", "gap": "40px"}),
        color="#6c7a89"
    ),

    dcc.Tabs(id="tabs", value="positive", children=[
        dcc.Tab(label="Positive Comments", value="positive", style={"color": "#6c7a89"}),
        dcc.Tab(label="Neutral Comments", value="neutral", style={"color": "#5a6b7d"}),
        dcc.Tab(label="Negative Comments", value="negative", style={"color": "#d9534f"}),
    ]),
    html.Div(id="comments-content")
])

# Callback: показать имя загруженного тестового файла
@app.callback(
    Output("upload-status", "children"),
    Input("upload-test-data", "filename")
)
def show_filename(name):
    if name:
        return html.Div(f"📄 Загружен файл: {name}", style={"marginLeft": "10px", "color": "#28a745"})
    return ""

# Store test data
@app.callback(
    Output("uploaded-test-data", "data"),
    Input("upload-test-data", "contents")
)
def store_uploaded_test_data(contents):
    if contents is None:
        return None
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        data = json.load(io.StringIO(decoded.decode("utf-8")))
        return data
    except Exception:
        return None

# Обучение модели
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
            response = requests.post("http://127.0.0.1:5000/api/train", json={
                "file_name": filename,
                "file_content": content_string
            })
            response.raise_for_status()
            return html.Div("✅ Training completed successfully.", style={"color": "#28a745"})
        except requests.exceptions.RequestException as e:
            return html.Div(f"❌ Error during training: {str(e)}", style={"color": "#d9534f"})
    return html.Div("Upload a file and click 'Start Training' to begin.")

# Анализ данных
@app.callback(
    Output("dashboard-content", "children"),
    Output("wordclouds", "children"),
    Input("analyze-button", "n_clicks"),
    State("uploaded-test-data", "data")
)
def update_dashboard(n_clicks, data):
    if n_clicks > 0:
        try:
            if data is None:
                return html.Div("⚠️ Сначала загрузите файл с тестовыми комментариями."), html.Div()

            if isinstance(data, dict) and "items" in data:
                items = data["items"]
            elif isinstance(data, list):
                items = data
            else:
                return html.Div("⚠️ Неверный формат данных. Ожидался словарь с ключом 'items'."), html.Div()

            processed_data = analyze_sentiments(items)
            global df
            df = pd.DataFrame(processed_data)

            df["date"] = pd.to_datetime(df["date"], unit="s")
            df["day"] = df["date"].dt.date
            df["likes"] = df["likes"].apply(lambda x: x.get("count", 0) if isinstance(x, dict) else 0)
            df["comments"] = df["comments"].apply(lambda x: x.get("count", 0) if isinstance(x, dict) else 0)

            fig1 = px.bar(df["sentiment_label"].value_counts().reset_index(), x="index", y="sentiment_label", 
                          labels={"index": "Sentiment", "sentiment_label": "Count"}, title="Sentiment Distribution")
            fig2 = px.line(df.groupby("day")["likes"].sum().reset_index(), x="day", y="likes", title="Likes Over Time")
            fig3 = px.line(df.groupby("day")["comments"].sum().reset_index(), x="day", y="comments", title="Comments Over Time")
            fig4 = px.line(df.groupby(["day", "sentiment_label"]).size().reset_index(name="count"),
                           x="day", y="count", color="sentiment_label", title="Sentiment Over Time")

            cloud_positive = generate_wordcloud(df, "positive")
            cloud_neutral = generate_wordcloud(df, "neutral")
            cloud_negative = generate_wordcloud(df, "negative")

            wordclouds = html.Div([
                html.Div([html.H4("Positive"), html.Img(src=f"/static/{cloud_positive}", style={"width": "250px"})]),
                html.Div([html.H4("Neutral"), html.Img(src=f"/static/{cloud_neutral}", style={"width": "250px"})]),
                html.Div([html.H4("Negative"), html.Img(src=f"/static/{cloud_negative}", style={"width": "250px"})])
            ], style={"display": "flex", "justifyContent": "center", "gap": "40px"})

            return html.Div([
                dcc.Graph(figure=fig1),
                dcc.Graph(figure=fig2),
                dcc.Graph(figure=fig3),
                dcc.Graph(figure=fig4)
            ]), wordclouds

        except Exception as e:
            return html.Div([
                html.H3("❌ An error occurred during analysis", style={"color": "#d9534f"}),
                html.P(str(e))
            ]), html.Div()

    return html.Div("Click the button to analyze the uploaded data."), html.Div()

# Комментарии по вкладкам
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
