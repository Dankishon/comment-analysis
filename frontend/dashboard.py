import dash
from dash import dcc, html
import pandas as pd
import requests

app = dash.Dash(__name__)

# Пример данных (замените на загрузку из Flask API)
data = requests.get('http://127.0.0.1:5000/api/data').json()
df = pd.DataFrame(data)

app.layout = html.Div([
    html.H1("Text Analysis Dashboard"),
    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': df['id'], 'y': df['likes_count'], 'type': 'bar', 'name': 'Likes'},
                {'x': df['id'], 'y': df['comments_count'], 'type': 'bar', 'name': 'Comments'},
            ],
            'layout': {
                'title': 'Likes and Comments per Post'
            }
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
