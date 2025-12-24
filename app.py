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
# Plotly template (light)  ✅ REQUIRED because you use template="infra_light"
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

# ------------------------------------------------------------------
# Global color encoding (used across summary plots)
# ------------------------------------------------------------------

GLOBAL_COLOR_METRIC = (
    "Real_GDP_per_Capita_USD"
    if "Real_GDP_per_Capita_USD" in df.columns
    else (numeric_cols[0] if numeric_cols else None)
)

GLOBAL_COLORSCALE = "Viridis"  # perceptually uniform, investor-safe


def col_as_series(frame: pd.DataFrame, col: str) -> pd.Series:
    """
    Returns a 1D Series even if `col` is duplicated in the DataFrame.
    If duplicates exist, we take the first occurrence.
    """
    obj = frame.loc[:, col]  # Series if unique, DataFrame if duplicated
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    return obj

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

def make_opp_risk_scatter(x_metric: str, y_metric: str, selected_countries: list[str], color_metric=None):
    # Guards
    if not x_metric or not y_metric or x_metric not in df.columns or y_metric not in df.columns:
        fig = go.Figure()
        fig.update_layout(template="infra_light")
        return fig

    selected_countries = selected_countries or []

    # Pick color metric: PCP selection > global fallback
    if not color_metric or color_metric not in df.columns:
        color_metric = GLOBAL_COLOR_METRIC if (GLOBAL_COLOR_METRIC and GLOBAL_COLOR_METRIC in df.columns) else None

    # Build numeric series safely (duplicate headers handled)
    x = pd.to_numeric(col_as_series(df, x_metric), errors="coerce")
    y = pd.to_numeric(col_as_series(df, y_metric), errors="coerce")

    plot_df = pd.DataFrame({COUNTRY_COL: df[COUNTRY_COL], "_x": x, "_y": y})
    plot_df = plot_df.dropna(subset=["_x", "_y"])

    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(template="infra_light")
        return fig

    use_color = False
    if color_metric:
        c = pd.to_numeric(col_as_series(df, color_metric), errors="coerce")
        plot_df["_c"] = c

        # Only use color if we have enough non-NaN values; otherwise, fallback to plain scatter
        non_na = plot_df["_c"].notna().sum()
        if non_na >= max(10, int(0.25 * len(plot_df))):
            use_color = True
            plot_col = plot_df.dropna(subset=["_c"])
        else:
            use_color = False

    if use_color:
        fig = px.scatter(
            plot_col,
            x="_x",
            y="_y",
            color="_c",
            color_continuous_scale=GLOBAL_COLORSCALE,
            hover_name=COUNTRY_COL,
            labels={
                "_x": label_for(x_metric),
                "_y": label_for(y_metric),
                "_c": label_for(color_metric),
            },
            template="infra_light",
        )
        # Hide colorbar (keeps chart clean in cards). You can turn it on later.
        fig.update_layout(coloraxis_showscale=False)
    else:
        fig = px.scatter(
            plot_df,
            x="_x",
            y="_y",
            hover_name=COUNTRY_COL,
            labels={"_x": label_for(x_metric), "_y": label_for(y_metric)},
            template="infra_light",
        )
        fig.update_traces(marker=dict(color="rgba(17,24,39,0.55)"))

    # Marker style without destroying marker.color arrays
    for tr in fig.data:
        if tr.type == "scatter" and tr.mode == "markers":
            tr.marker.size = 8
            tr.marker.opacity = 0.85

    # Selected countries: ring overlay (doesn't overwrite underlying color)
    if selected_countries:
        sel = plot_df[plot_df[COUNTRY_COL].isin(selected_countries)]
        if len(sel) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sel["_x"],
                    y=sel["_y"],
                    mode="markers",
                    marker=dict(
                        size=14,
                        color="rgba(0,0,0,0)",
                        line=dict(width=2.5, color="rgba(59,130,246,0.95)"),
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig = card_layout(fig, x_title=label_for(x_metric), y_title=label_for(y_metric))
    return fig


def make_ranked_bar(metric: str, selected_countries: list[str], top_n=10):
    if not metric or metric not in df.columns:
        fig = go.Figure()
        fig.update_layout(template="infra_light")
        return fig

    selected_countries = selected_countries or []

    values = pd.to_numeric(col_as_series(df, metric), errors="coerce")
    plot_df = pd.DataFrame({COUNTRY_COL: df[COUNTRY_COL], "_v": values}).dropna(subset=["_v"])
    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(template="infra_light")
        return fig

    plot_df = plot_df.sort_values("_v", ascending=False)
    top = plot_df.head(top_n)
    bottom = plot_df.tail(top_n)
    out = pd.concat([top, bottom], axis=0)

    # Reverse so largest appears at top in horizontal bars
    out = out.iloc[::-1]

    fig = px.bar(
        out,
        x="_v",
        y=COUNTRY_COL,
        orientation="h",
        color="_v",  # color by ranked value (best readability)
        color_continuous_scale=GLOBAL_COLORSCALE,
        template="infra_light",
        labels={"_v": label_for(metric), COUNTRY_COL: ""},
    )

    fig.update_layout(
        coloraxis_showscale=False,
        barmode="overlay",
    )

    # Selected overlay as solid blue bars (Plotly bars don't support marker.size)
    if selected_countries:
        sel = out[out[COUNTRY_COL].isin(selected_countries)]
        if len(sel) > 0:
            fig.add_trace(
                go.Bar(
                    x=sel["_v"],
                    y=sel[COUNTRY_COL],
                    orientation="h",
                    marker=dict(color="rgba(59,130,246,0.95)", line=dict(color="white", width=1.5)),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>" + f"{label_for(metric)}: %{{x}}<extra></extra>",
        opacity=0.90,
    )

    fig = card_layout(fig, x_title=label_for(metric))
    return fig


def make_distribution(metric: str, selected_country=None):
    if not metric or metric not in df.columns:
        fig = go.Figure()
        fig.update_layout(template="infra_light")
        return fig

    s = pd.to_numeric(df[metric], errors="coerce").dropna()

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=s,
            nbinsx=30,
            marker_color="rgba(17,24,39,0.35)",
            hovertemplate=f"{label_for(metric)} range<br>Count: %{{y}}<extra></extra>",
        )
    )

    # Median & quartiles
    for q in [0.25, 0.5, 0.75]:
        fig.add_vline(
            x=s.quantile(q),
            line_width=1,
            line_dash="dot",
            line_color=THEME["muted_text"],
        )

    # Selected country marker + soft fill
    if selected_country:
        ser_all = col_as_series(df, metric)
        ser = pd.to_numeric(ser_all[df[COUNTRY_COL] == selected_country], errors="coerce")
        if len(ser) > 0:
            val = ser.iloc[0]
            if pd.notna(val):
                fig.add_vrect(x0=val, x1=val, fillcolor="rgba(59,130,246,0.15)", line_width=0)
                fig.add_vline(x=val, line_width=2, line_color="rgba(59,130,246,0.95)")


    fig.update_layout(
        template="infra_light",
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )

    fig = card_layout(fig, x_title=label_for(metric), y_title="Countries")
    return fig


def card_layout(fig, *, x_title=None, y_title=None):
    fig.update_layout(
        margin=dict(l=48, r=16, t=8, b=40),  # tuned for your panel size
        xaxis=dict(
            title=None,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title=None,
            tickfont=dict(size=11),
        ),
    )

    # Subtle axis labels as annotations (optional but recommended)
    annotations = []

    if x_title:
        annotations.append(
            dict(
                text=x_title,
                x=0.5,
                y=-0.18,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=11, color=THEME["muted_text"]),
            )
        )

    if y_title:
        annotations.append(
            dict(
                text=y_title,
                x=-0.12,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                textangle=-90,
                font=dict(size=11, color=THEME["muted_text"]),
            )
        )

    fig.update_layout(annotations=annotations)
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
                                "responsive": True,
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
                                    className="pcp-scale",
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
                # -----------------------------
                # Row 1 (full width): Scatter + Bar (2 cols each)
                # -----------------------------
                html.Div(
                    className="panel",
                    style={"gridColumn": "span 2"},
                    children=[
                        html.Div("Opportunity vs Risk (Global)", className="panel-title"),
                        dcc.Graph(
                            id="opp-risk-scatter",
                            className="panel-content",
                            config={"displayModeBar": False, "responsive": True},
                        ),
                    ],
                ),
                html.Div(
                    className="panel",
                    style={"gridColumn": "span 2"},
                    children=[
                        html.Div("Top / Bottom Countries", className="panel-title"),
                        dcc.Graph(
                            id="ranked-bar",
                            className="panel-content",
                            config={"displayModeBar": False, "responsive": True},
                        ),
                    ],
                ),

                # -----------------------------
                # Row 2 (full width): Distribution + Companion panel
                # -----------------------------
                html.Div(
                    className="panel",
                    style={"gridColumn": "span 2"},
                    children=[
                        html.Div("Global Distribution Context", className="panel-title"),
                        dcc.Graph(
                            id="metric-distribution",
                            className="panel-content",
                            config={"displayModeBar": False, "responsive": True},
                        ),
                    ],
                ),
                html.Div(
                    className="panel",
                    style={"gridColumn": "span 2"},
                    children=[
                        html.Div("View 4 (placeholder)", className="panel-title"),
                        html.Div("Placeholder", className="panel-placeholder"),
                    ],
                ),

                # -----------------------------
                # Row 3 (4 panels)
                # -----------------------------
                html.Div(className="panel", children=[html.Div("View 5", className="panel-title"), html.Div("Placeholder", className="panel-placeholder")]),
                html.Div(className="panel", children=[html.Div("View 6", className="panel-title"), html.Div("Placeholder", className="panel-placeholder")]),
                html.Div(className="panel", children=[html.Div("View 7", className="panel-title"), html.Div("Placeholder", className="panel-placeholder")]),
                html.Div(className="panel", children=[html.Div("View 8", className="panel-title"), html.Div("Placeholder", className="panel-placeholder")]),

                # -----------------------------
                # Row 4 (4 panels)
                # -----------------------------
                html.Div(className="panel", children=[html.Div("View 9", className="panel-title"), html.Div("Placeholder", className="panel-placeholder")]),
                html.Div(className="panel", children=[html.Div("View 10", className="panel-title"), html.Div("Placeholder", className="panel-placeholder")]),
                html.Div(className="panel", children=[html.Div("View 11", className="panel-title"), html.Div("Placeholder", className="panel-placeholder")]),
                html.Div(className="panel", children=[html.Div("View 12", className="panel-title"), html.Div("Placeholder", className="panel-placeholder")]),
            ],
        )
    ])

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

@app.callback(
    Output("opp-risk-scatter", "figure"),
    Output("ranked-bar", "figure"),
    Output("metric-distribution", "figure"),
    Input("map-metric", "value"),
    Input("pcp-selected-countries", "data"),
    Input("world-map", "clickData"),
    Input("pcp-color", "value"),
)
def update_global_summary(metric, selected_countries, map_click, pcp_color):
    # Define proxies (can later be user-selectable)
    opportunity_metric = metric
    risk_metric = "Public_Debt_percent_of_GDP" if "Public_Debt_percent_of_GDP" in df.columns else metric

    selected_country = None
    if map_click:
        selected_country = map_click["points"][0]["location"]

    scatter = make_opp_risk_scatter(opportunity_metric, risk_metric, selected_countries, color_metric=pcp_color)
    bars = make_ranked_bar(metric, selected_countries)
    dist = make_distribution(metric, selected_country)

    return scatter, bars, dist



if __name__ == "__main__":
    app.run(debug=True)