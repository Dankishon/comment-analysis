import sys
import os
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import pandas as pd
import requests

# Добавляем путь к backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
from process_data import analyze_sentiments, load_json, generate_wordcloud, get_topics_distribution

# Инициализация Dash
app = dash.Dash(__name__, external_stylesheets=["/frontend/static/styles.css"])
app.title = "Text Analysis Dashboard"

# Пути
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEST_FILE_PATH = os.path.join(BASE_DIR, 'data', 'test_vk_post.json')
TRAIN_API_URL = "http://127.0.0.1:5000/api/train"

# Макет интерфейса
app.layout = html.Div([
    html.H1("Text Analysis Dashboard", style={"textAlign": "center", "color": "#3a3a3a"}),
    html.P("Analyze social media texts with sentiment analysis and topic modeling.", style={"textAlign": "center", "color": "#7f8c8d"}),

    html.Div([
        dcc.Upload(id="upload-data", children=html.Div(["Drag and Drop or ", html.A("Select File")]), style={
            "width": "100%", "height": "60px", "lineHeight": "60px", "borderWidth": "1px",
            "borderStyle": "dashed", "borderRadius": "5px", "textAlign": "center", "margin": "10px",
        }, multiple=False),
        html.Button("Start Training", id="train-button", n_clicks=0, style={
            "margin": "10px", "padding": "10px", "backgroundColor": "#6c7a89", "color": "white"
        }),
        dcc.Loading(id="loading-train", type="circle", children=html.Div(id="training-status"), color="#6c7a89")
    ]),

    html.Div([
        html.Button("Analyze Test Data", id="analyze-button", n_clicks=0, style={
            "margin": "10px", "padding": "10px", "backgroundColor": "#6c7a89", "color": "white"
        })
    ], style={"textAlign": "center"}),

    dcc.Loading(id="loading-dashboard", type="default", children=html.Div(id="dashboard-content"), color="#6c7a89"),

    html.H2("Word Clouds", style={"textAlign": "center", "marginTop": "30px"}),
    dcc.Loading(id="loading-clouds", type="default", children=html.Div(id="wordclouds", style={
        "display": "flex", "justifyContent": "center", "gap": "40px"
    }), color="#6c7a89"),

    html.H2("Dynamic Topic Analysis", style={"textAlign": "center", "marginTop": "30px"}),
    dcc.Loading(id="loading-topics", type="default", children=html.Div(id="topics-content"), color="#6c7a89"),

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
    Output("wordclouds", "children"),
    Output("topics-content", "children"),
    Input("analyze-button", "n_clicks")
)
def update_dashboard(n_clicks):
    if n_clicks > 0:
        try:
            data = load_json(TEST_FILE_PATH)
            processed_data = analyze_sentiments(data["items"])
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

            topics_df = get_topics_distribution(df)
            if topics_df.empty:
                topics_content = html.Div("No topics found.")
            else:
                topics_content = dash_table.DataTable(
                    data=topics_df.to_dict("records"),
                    columns=[{"name": i, "id": i} for i in topics_df.columns],
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "padding": "5px"},
                    style_header={"backgroundColor": "#6c7a89", "color": "white"}
                )

            return html.Div([dcc.Graph(figure=fig1), dcc.Graph(figure=fig2), dcc.Graph(figure=fig3), dcc.Graph(figure=fig4)]), wordclouds, topics_content
        except Exception as e:
            return html.Div([html.H3("An error occurred during analysis", style={"color": "#d9534f"}), html.P(str(e))]), html.Div(), html.Div()
    return html.Div("Click the button to analyze the test data."), html.Div(), html.Div()

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
    app.run(debug=True)
