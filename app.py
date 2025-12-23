from dash import Dash, html, dcc, Input, Output, State, callback_context
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import re

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = Dash(__name__)

THEME = {
    "background": "#F6F7FB",   # page background (very light gray)
    "panel": "#FFFFFF",        # card background
    "panel_alt": "#F3F4F6",    # subtle alt background
    "text": "#111827",         # near-black text
    "muted_text": "#6B7280",   # gray text
    "border": "rgba(17,24,39,0.10)",
}

# -----------------------------------------------------------------------------
# Plotly template (light)
# -----------------------------------------------------------------------------
pio.templates["infra_light"] = go.layout.Template(
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
pio.templates.default = "infra_light"

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
df = pd.read_csv("data/CIA_DATA.csv")

COUNTRY_COL = "Country"
DEFAULT_METRIC = "Real_GDP_per_Capita_USD" if "Real_GDP_per_Capita_USD" in df.columns else None
numeric_cols = df.select_dtypes(include="number").columns.tolist()

def pretty_label(col: str) -> str:
    s = col.replace("_", " ").strip()
    s = re.sub(r"\s+", " ", s)
    acronyms = {"gdp", "usd", "ppp", "co2", "km", "kw"}
    words = []
    for w in s.split(" "):
        if w.lower() in acronyms:
            words.append(w.upper())
        else:
            words.append(w.capitalize())
    return " ".join(words)

LABELS = {
    "Real_GDP_per_Capita_USD": "GDP per Capita (USD)",
    "Real_GDP_PPP_billion_USD": "GDP (PPP, $B)",
    "Unemployment_Rate_percent": "Unemployment Rate (%)",
    "Youth_Unemployment_Rate_percent": "Youth Unemployment (%)",
    "Public_Debt_percent_of_GDP": "Public Debt (% of GDP)",
    "electricity_access_percent": "Electricity Access (%)",
    "electricity_generating_capacity_kW": "Electricity Capacity (kW)",
    "roadways_km": "Roadways (km)",
    "railways_km": "Railways (km)",
    "mobile_cellular_subscriptions_total": "Mobile Subscriptions (total)",
    "broadband_fixed_subscriptions_total": "Fixed Broadband (total)",
    "internet_users_total": "Internet Users (total)",
}

GROUP_RULES = [
    ("Economy", ["gdp", "exports", "imports", "inflation", "debt", "budget", "poverty", "unemployment"]),
    ("Demographics", ["population", "birth", "death", "median_age", "growth", "literacy", "migration"]),
    ("Energy", ["electric", "electricity", "coal", "petroleum", "natural_gas", "emissions", "carbon"]),
    ("Transportation", ["road", "roadways", "rail", "railways", "airport", "airports", "waterway", "pipeline"]),
    ("Communications", ["mobile", "broadband", "internet", "telephone", "subscriptions"]),
    ("Other", []),
]

def assign_group(col: str) -> str:
    lc = col.lower()
    for group_name, keys in GROUP_RULES:
        if group_name == "Other":
            continue
        if any(k in lc for k in keys):
            return group_name
    return "Other"

def label_for(c: str) -> str:
    return LABELS.get(c, pretty_label(c))

# Bucket columns into groups
buckets = {g[0]: [] for g in GROUP_RULES}
for c in numeric_cols:
    buckets[assign_group(c)].append(c)

for g in buckets:
    buckets[g] = sorted(buckets[g], key=label_for)

if DEFAULT_METRIC is None and numeric_cols:
    DEFAULT_METRIC = numeric_cols[0]

# -----------------------------------------------------------------------------
# Figure builder
# -----------------------------------------------------------------------------
def make_map(metric: str):
    if not metric or metric not in df.columns:
        fig = go.Figure()
        fig.update_layout(template="infra_light")
        return fig

    s = pd.to_numeric(df[metric], errors="coerce")
    s_valid = s.dropna()
    if len(s_valid) == 0:
        fig = go.Figure()
        fig.update_layout(template="infra_light")
        return fig

    vmin = s_valid.quantile(0.05)
    vmax = s_valid.quantile(0.95)
    if vmin == vmax:
        vmin = s_valid.min()
        vmax = s_valid.max()

    fig = px.choropleth(
        df,
        locations=COUNTRY_COL,
        locationmode="country names",
        color=metric,
        color_continuous_scale="Viridis",
        range_color=[vmin, vmax],
        template="infra_light",
    )

    fig.update_traces(
        hovertemplate="<b>%{location}</b><br>"
                      f"{label_for(metric)}: %{{z}}<br>"
                      f"Color range: [{vmin:.2f}, {vmax:.2f}] (5–95th pct)<extra></extra>",
        marker_line_width=0
    )

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
        countrycolor="rgba(17,24,39,0.10)",
        projection_type="equirectangular",
        projection_scale=1.65,
        center=dict(lat=20, lon=0),
    )

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

        html.Div(
            className="top-row",
            children=[
                # LEFT: Map panel
                html.Div(
                    className="panel panel-tight",
                    children=[
                        html.Div(
                            className="map-controls",
                            children=[
                                dcc.Dropdown(
                                    id="metric-group",
                                    className="light-dropdown",
                                    options=[{"label": g[0], "value": g[0]} for g in GROUP_RULES if g[0] != "Other"],
                                    value="Economy",
                                    clearable=False,
                                    searchable=False,
                                ),
                                dcc.Dropdown(
                                    id="map-metric",
                                    className="light-dropdown",
                                    options=[],
                                    value=DEFAULT_METRIC,
                                    clearable=False,
                                    searchable=True,
                                    placeholder="Select a metric…",
                                ),
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
                                "responsive": False,  # Change from True to False
                            },
                        ),
                    ],
                ),

                # RIGHT: Main panel placeholder
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
    Output("map-metric", "options"),
    Output("map-metric", "value"),
    Input("metric-group", "value"),
)
def update_metric_dropdown(group_name):
    cols = buckets.get(group_name, [])
    opts = [{"label": label_for(c), "value": c} for c in cols]
    val = DEFAULT_METRIC if DEFAULT_METRIC in cols else (cols[0] if cols else None)
    return opts, val

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
