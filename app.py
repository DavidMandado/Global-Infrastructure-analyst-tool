from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = Dash(__name__)

THEME = {
    "background": "#05060A",
    "panel": "#111827",
    "panel_alt": "#020617",
    "text": "#F9FAFB",
    "muted_text": "#9CA3AF",
}

# -----------------------------------------------------------------------------
# Plotly theme/template (keeps visuals consistent across all future plots)
# -----------------------------------------------------------------------------
pio.templates["infra_dark"] = go.layout.Template(
    layout=dict(
        paper_bgcolor=THEME["panel"],
        plot_bgcolor=THEME["panel"],
        font=dict(
            family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            color=THEME["text"],
            size=12,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
)
pio.templates.default = "infra_dark"

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
df = pd.read_csv("data/CIA_DATA.csv")

COUNTRY_COL = "Country"
DEFAULT_METRIC = "Real_GDP_per_Capita_USD" if "Real_GDP_per_Capita_USD" in df.columns else None

# A simple metric dropdown list (numeric columns only)
numeric_cols = df.select_dtypes(include="number").columns.tolist()
metric_options = [{"label": c, "value": c} for c in numeric_cols]

if DEFAULT_METRIC is None and numeric_cols:
    DEFAULT_METRIC = numeric_cols[0]

# -----------------------------------------------------------------------------
# Figure builder
# -----------------------------------------------------------------------------
def make_map(metric: str):
    """Create a choropleth map for the given metric."""
    if not metric or metric not in df.columns:
        # empty figure fallback
        fig = go.Figure()
        fig.update_layout(template="infra_dark")
        return fig

    fig = px.choropleth(
        df,
        locations=COUNTRY_COL,
        locationmode="country names",
        color=metric,
        color_continuous_scale="Viridis",
        template="infra_dark",
    )

    # Make it blend into the card and behave like a selector
    fig.update_layout(
        coloraxis_showscale=False,
        showlegend=False,
        dragmode="zoom",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.update_geos(
        bgcolor=THEME["panel"],
        showocean=True,
        oceancolor=THEME["panel"],
        showland=True,
        landcolor=THEME["panel_alt"],
        showframe=False,
        showcountries=True,
        countrycolor="rgba(255,255,255,0.12)",
        projection_type="equirectangular",
        projection_scale=1.65,
        center=dict(lat=20, lon=0),
    )
    fig.update_traces(marker_line_width=0)
    return fig


# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------
app.layout = html.Div(
    className="app-page",
    children=[
        html.Div(
            id="selected-country",
            className="selected-country",
            children="Click a country on the map to see more information.",
        ),

        # Top row: map + main panel
        html.Div(
            className="top-row",
            children=[
                # LEFT: Map panel
                html.Div(
                    className="panel panel-tight",
                    children=[
                        html.H3("World Map", className="panel-title"),
                        html.Div(
                            style={"padding": "0 12px 8px 12px"},
                            children=[
                                dcc.Dropdown(
                                    id="map-metric",
                                    options=metric_options,
                                    value=DEFAULT_METRIC,
                                    clearable=False,
                                    searchable=True,
                                    placeholder="Select a metric to color the map",
                                )
                            ],
                        ),
                        dcc.Graph(
                            id="world-map",
                            className="panel-content",
                            figure=make_map(DEFAULT_METRIC),
                            style={"height": "100%", "width": "100%"},
                            config={
                                "displayModeBar": False,
                                "scrollZoom": True,
                                "doubleClick": "reset",
                                "responsive": True,
                            },
                        ),
                    ],
                ),

                # RIGHT: Main panel (placeholder for PCP)
                html.Div(
                    className="panel",
                    children=[
                        html.H3("Global Overview", className="panel-title"),
                        html.Div(
                            "Placeholder: Parallel Coordinates Plot (PCP) will go here.",
                            className="panel-placeholder",
                        ),
                    ],
                ),
            ],
        ),

        # Lower grid: placeholders
        html.Div(
            className="app-grid2",
            children=[
                html.Div(className="panel", children=[html.H3("View 1", className="panel-title"),
                                                      html.Div("Placeholder", className="panel-placeholder")]),
                html.Div(className="panel", children=[html.H3("View 2", className="panel-title"),
                                                      html.Div("Placeholder", className="panel-placeholder")]),
                html.Div(className="panel", children=[html.H3("View 3", className="panel-title"),
                                                      html.Div("Placeholder", className="panel-placeholder")]),
                html.Div(className="panel", children=[html.H3("View 4", className="panel-title"),
                                                      html.Div("Placeholder", className="panel-placeholder")]),
                html.Div(className="panel", children=[html.H3("View 5", className="panel-title"),
                                                      html.Div("Placeholder", className="panel-placeholder")]),
                html.Div(className="panel", children=[html.H3("View 6", className="panel-title"),
                                                      html.Div("Placeholder", className="panel-placeholder")]),
            ],
        ),
    ],
)

# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
@app.callback(
    Output("world-map", "figure"),
    Input("map-metric", "value"),
)
def update_map(metric):
    return make_map(metric)


@app.callback(
    Output("selected-country", "children"),
    Input("world-map", "clickData"),
)
def update_selected_country(clickData):
    if clickData and "points" in clickData and clickData["points"]:
        country = clickData["points"][0].get("location", "Unknown")
        return f"Selected country: {country}"
    return "Click a country on the map to see more information."


if __name__ == "__main__":
    app.run(debug=True)
