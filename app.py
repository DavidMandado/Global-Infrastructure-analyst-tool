from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd

# -----------------------------------------------------------------------------
# App + theme
# -----------------------------------------------------------------------------
app = Dash(__name__)

THEME = {
    "background": "#05060A",
    "panel": "#111827",
    "panel_alt": "#020617",
    "text": "#F9FAFB",
    "muted_text": "#9CA3AF",
    "accent": "#22C55E",   # green
}

# Plotly template for consistent styling
pio.templates["infra_dark"] = go.layout.Template(
    layout=dict(
        paper_bgcolor=THEME["panel"],
        plot_bgcolor=THEME["panel"],
        font=dict(
            family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            color=THEME["text"],
            size=12,
        ),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
        ),
    )
)
pio.templates.default = "infra_dark"

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
df = pd.read_csv("data/CIA_DATA.csv")

# Choose a metric for map color
if "Real_GDP_per_Capita_USD" in df.columns:
    color_col = "Real_GDP_per_Capita_USD"
else:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    color_col = numeric_cols[0] if numeric_cols else None

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
map_fig = px.choropleth(
    df,
    locations="Country",             # adjust if your column is named differently
    locationmode="country names",
    color=color_col,
    color_continuous_scale="Viridis",
    title=f"World map colored by {color_col}" if color_col else "World map",
    template="infra_dark",
)
map_fig.update_layout(height=320)

# -----------------------------------------------------------------------------
# Layout: 3 x 2 grid of blocks
# -----------------------------------------------------------------------------
app.layout = html.Div(
    className="app-page",
    children=[
        html.H1(
            "Global Infrastructure Investment Atlas (Demo Layout)",
            className="app-title",
        ),

        # Small text area that will show the selected country from the map
        html.Div(
            id="selected-country",
            className="selected-country",
            children="Click a country on the map to select it.",
        ),

        # Grid container
        html.Div(
            className="app-grid",
            children=[
                # Block 1: World map
                html.Div(
                    className="panel",
                    children=[
                        html.H3("World Map", className="panel-title"),
                        dcc.Graph(
                            id="world-map",
                            figure=map_fig,
                            className="panel-content",
                        ),
                    ],
                ),

                # Block 2: empty placeholder for future view
                html.Div(
                    className="panel",
                    children=[
                        html.H3("View 2", className="panel-title"),
                        html.Div(
                            "Placeholder for future view 2",
                            className="panel-placeholder",
                        ),
                    ],
                ),

                # Block 3: empty placeholder for future view
                html.Div(
                    className="panel",
                    children=[
                        html.H3("View 3", className="panel-title"),
                        html.Div(
                            "Placeholder for future view 3",
                            className="panel-placeholder",
                        ),
                    ],
                ),

                # Block 4: empty placeholder for future view
                html.Div(
                    className="panel",
                    children=[
                        html.H3("View 4", className="panel-title"),
                        html.Div(
                            "Placeholder for future view 4",
                            className="panel-placeholder",
                        ),
                    ],
                ),

                # Block 5: empty placeholder for future view
                html.Div(
                    className="panel",
                    children=[
                        html.H3("View 5", className="panel-title"),
                        html.Div(
                            "Placeholder for future view 5",
                            className="panel-placeholder",
                        ),
                    ],
                ),

                # Block 6: empty placeholder for future view
                html.Div(
                    className="panel",
                    children=[
                        html.H3("View 6", className="panel-title"),
                        html.Div(
                            "Placeholder for future view 6",
                            className="panel-placeholder",
                        ),
                    ],
                ),
            ],
        ),
    ],
)

# -----------------------------------------------------------------------------
# Simple interaction: click on map → show selected country name
# -----------------------------------------------------------------------------
@app.callback(
    Output("selected-country", "children"),
    Input("world-map", "clickData"),
)
def update_selected_country(clickData):
    if clickData and "points" in clickData:
        # For choropleth, country name is usually in "location"
        country = clickData["points"][0].get("location", "Unknown")
        return f"Selected country: {country}"
    return "Click a country on the map to select it."


if __name__ == "__main__":
    app.run(debug=True)
