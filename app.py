from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

app = Dash()

df = pd.read_csv("data/CIA_DATA.csv")

print("Checkpoint 1 ⇣")

fig = px.bar(df, x="Country", y="Population_Below_Poverty_Line_percent", barmode="group")

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
             Dash: A web application tatata
             '''),
    
    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run(debug=True)