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

def _to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def transform_for_pcp(df_in: pd.DataFrame, cols: list[str], mode: str) -> pd.DataFrame:
    """
    Returns a DataFrame with the selected cols transformed for comparability.
    - pct: percentile (0..1)
    - z: z-score
    - log: log10(positive only -> others become NaN)
    - raw: numeric as-is
    """
    out = pd.DataFrame({COUNTRY_COL: df_in[COUNTRY_COL]})
    for c in cols:
        s = _to_numeric_series(df_in[c])

        if mode == "pct":
            # percentile ranks in [0,1]
            out[c] = s.rank(pct=True)
        elif mode == "z":
            mu = s.mean(skipna=True)
            sd = s.std(skipna=True)
            out[c] = (s - mu) / (sd if sd and sd > 0 else 1.0)
        elif mode == "log":
            out[c] = s.where(s > 0).apply(lambda v: None if pd.isna(v) else __import__("math").log10(v))
        else:
            out[c] = s

    return out

def apply_constraints(df_in: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    """
    constraints format: {col: constraintrange}
    constraintrange can be [lo, hi] or list of ranges [[lo, hi], [lo2, hi2]]
    """
    if not constraints:
        return df_in

    d = df_in.copy()
    for col, cr in constraints.items():
        if col not in d.columns or cr is None:
            continue

        s = d[col]
        if isinstance(cr, list) and len(cr) == 2 and not isinstance(cr[0], list):
            lo, hi = cr
            d = d[(s >= lo) & (s <= hi)]
        elif isinstance(cr, list) and len(cr) > 0 and isinstance(cr[0], list):
            # union of multiple ranges
            mask = False
            for lo, hi in cr:
                mask = mask | ((s >= lo) & (s <= hi))
            d = d[mask]

    return d

def parse_pcp_constraints(relayout: dict, dims: list[str]) -> dict:
    """
    PCP brush events come via relayoutData with keys like:
    'dimensions[2].constraintrange': [a,b]
    """
    if not relayout:
        return {}

    constraints = {}
    for k, v in relayout.items():
        if "constraintrange" not in k:
            continue
        m = re.match(r"dimensions\[(\d+)\]\.constraintrange", k)
        if not m:
            continue
        idx = int(m.group(1))
        if 0 <= idx < len(dims):
            constraints[dims[idx]] = v
    return constraints

def make_pcp(dims: list[str], color_metric: str, scale_mode: str, constraints: dict):
    """
    Builds a go.Parcoords PCP (best for interaction + brush).
    The constraints are applied to compute selected countries, but also shown on axes.
    """
    if not dims:
        fig = go.Figure()
        fig.update_layout(template="infra_light")
        return fig, []

    # Transform for comparability
    td = transform_for_pcp(df, dims + ([color_metric] if color_metric and color_metric not in dims else []), scale_mode)

    # Apply constraints in transformed space
    td_filtered = apply_constraints(td, constraints)
    selected_countries = td_filtered[COUNTRY_COL].dropna().tolist()

    # line colors
    if color_metric and color_metric in td.columns:
        color_vals = td[color_metric]
    else:
        color_vals = td[dims[0]]

    # build dimensions with optional constraintrange
    dim_objs = []
    for c in dims:
        dim = dict(
            label=label_for(c),
            values=td[c],
        )
        if constraints and c in constraints:
            dim["constraintrange"] = constraints[c]
        dim_objs.append(dim)

    fig = go.Figure(
        data=[
            go.Parcoords(
                line=dict(
                    color=color_vals,
                    colorscale="Viridis",
                    showscale=False,
                ),
                dimensions=dim_objs,
                labelfont=dict(color=THEME["text"]),
                tickfont=dict(color=THEME["muted_text"]),
            )
        ]
    )

    fig.update_layout(
        template="infra_light",
        margin=dict(l=20, r=20, t=10, b=10),
    )

    return fig, selected_countries

def make_map_with_selection(metric: str, selected_countries: list[str]):
    """
    Keep the normal (full) choropleth colors exactly like make_map(),
    and only add an outline on selected countries.
    """
    # 1) Start from the original "good looking" map (same scaling, same colors)
    fig = make_map(metric)

    # 2) If nothing selected, return the normal map
    if not selected_countries:
        return fig

    # 3) Add an outline trace for the selected countries (transparent fill)
    outline = go.Choropleth(
        locations=selected_countries,
        locationmode="country names",
        z=[1] * len(selected_countries),  # dummy values
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],  # fully transparent fill
        showscale=False,
        hoverinfo="skip",
        marker=dict(
            line=dict(color="rgba(59,130,246,0.95)", width=2),  # blue outline (change if you want)
        ),
    )

    fig.add_trace(outline)
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

                # RIGHT: Main panel = PCP (the “main” dashboard plot)
                html.Div(
                    className="panel panel-tight",
                    children=[

                        # Stores for linking interactions
                        dcc.Store(id="pcp-constraints", data={}),
                        dcc.Store(id="pcp-selected-countries", data=[]),

                        html.Div(
                            className="pcp-controls",
                            children=[
                                html.Div(
                                    children=[
                                        html.Label("PCP metrics (axes)", className="control-label"),
                                        dcc.Dropdown(
                                            id="pcp-dims",
                                            className="light-dropdown",
                                            options=[{"label": label_for(c), "value": c} for c in numeric_cols],
                                            value=[c for c in [
                                                "Real_GDP_per_Capita_USD",
                                                "Unemployment_Rate_percent",
                                                "electricity_access_percent",
                                                "roadways_km",
                                                "mobile_cellular_subscriptions_total",
                                            ] if c in df.columns][:5],
                                            multi=True,
                                            clearable=False,
                                            placeholder="Select dimensions…",
                                        ),
                                    ],
                                ),

                                html.Div(
                                    children=[
                                        html.Label("Color lines by", className="control-label"),
                                        dcc.Dropdown(
                                            id="pcp-color",
                                            className="light-dropdown",
                                            options=[{"label": label_for(c), "value": c} for c in numeric_cols],
                                            value="Real_GDP_per_Capita_USD" if "Real_GDP_per_Capita_USD" in df.columns else (numeric_cols[0] if numeric_cols else None),
                                            clearable=False,
                                        ),
                                    ],
                                ),

                                html.Div(
                                    children=[
                                        html.Label("Scale", className="control-label"),
                                        dcc.RadioItems(
                                            id="pcp-scale",
                                            className="radio-row",
                                            options=[
                                                {"label": "Percentile", "value": "pct"},
                                                {"label": "Z-score", "value": "z"},
                                                {"label": "Log10", "value": "log"},
                                                {"label": "Raw", "value": "raw"},
                                            ],
                                            value="pct",
                                            inline=True,
                                        ),
                                    ],
                                ),

                                html.Div(
                                    className="pcp-actions",
                                    children=[
                                        html.Button("Reset selection", id="pcp-reset", n_clicks=0, className="btn"),
                                        html.Div(id="pcp-status", className="status-pill", children="Selected: all countries"),
                                    ],
                                ),
                            ],
                        ),

                        dcc.Graph(
                            id="pcp",
                            className="panel-content",
                            config={"displayModeBar": False, "responsive": True},
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
    Output("pcp", "figure"),
    Output("pcp-selected-countries", "data"),
    Output("pcp-constraints", "data"),
    Output("pcp-status", "children"),
    Output("world-map", "figure"),
    Input("pcp-dims", "value"),
    Input("pcp-color", "value"),
    Input("pcp-scale", "value"),
    Input("pcp", "relayoutData"),
    Input("pcp-reset", "n_clicks"),
    Input("map-metric", "value"),
    State("pcp-constraints", "data"),
)
def update_pcp_and_map(dims, color_metric, scale_mode, relayout, reset_clicks, map_metric, stored_constraints):
    # determine what triggered
    trig = (callback_context.triggered[0]["prop_id"] if callback_context.triggered else "")

    # reset button clears constraints
    if "pcp-reset" in trig:
        constraints = {}
    elif "pcp.relayoutData" in trig:
        # update constraints from brush events
        constraints = parse_pcp_constraints(relayout, dims or [])
        # merge with stored (so constraints persist if multiple axes brushed)
        merged = dict(stored_constraints or {})
        merged.update(constraints)
        constraints = merged
    else:
        constraints = stored_constraints or {}

    pcp_fig, selected = make_pcp(dims or [], color_metric, scale_mode, constraints)

    # status text
    if selected:
        status = f"Selected: {len(selected)} countries"
    else:
        status = "Selected: all countries"

    # linked map highlight
    map_fig = make_map_with_selection(map_metric, selected)

    return pcp_fig, selected, constraints, status, map_fig


if __name__ == "__main__":
    app.run(debug=True)
