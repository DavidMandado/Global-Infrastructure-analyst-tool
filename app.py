from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

app = Dash()

df = pd.read_csv("data/CIA_DATA.csv")

print("Checkpoint 1 ⇣")

