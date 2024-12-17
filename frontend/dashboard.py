import dash
from dash import html, dcc
import requests
import pandas as pd
import plotly.express as px

# Получение данных с бэкенда
response = requests.get("http://127.0.0.1:5000/api/data")
data = response.json()

# Преобразование данных в DataFrame
df = pd.DataFrame(data)

# Создание графиков
likes_bar = px.bar(
    df, 
    x="id", 
    y="likes_count", 
    title="Количество лайков по постам",
    labels={"likes_count": "Лайки", "id": "ID поста"}
)

comments_bar = px.bar(
    df, 
    x="id", 
    y="comments_count", 
    title="Количество комментариев по постам",
    labels={"comments_count": "Комментарии", "id": "ID поста"}
)

# Инициализация Dash-приложения
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Аналитика постов из JSON"),
    dcc.Graph(figure=likes_bar),
    dcc.Graph(figure=comments_bar)
])

if __name__ == "__main__":
    app.run_server(debug=True)
