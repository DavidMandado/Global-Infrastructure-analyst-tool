from dash import Dash, html, dcc, Input, Output, State, callback_context
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import re
from typing import Optional


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
# Data (load + clean + derive)
# -----------------------------------------------------------------------------
COUNTRY_COL = "Country"

NON_NUMERIC_COLS = {
    "Country",
    "Fiscal_Year",
    "internet_country_code",
    "Geographic_Coordinates",
    "Capital",
    "Capital_Coordinates",
    "Government_Type",
}

def _parse_numeric_series(s: pd.Series) -> pd.Series:
    """
    Converts strings like:
      '652,230 sq km' -> 652230
      '2.26%'         -> 2.26
      '-40 m'         -> -40
      '0 km'          -> 0
    Keeps the first numeric token found; NaN if none.
    """
    ss = s.astype("string").str.strip()
    ss = ss.replace(
        {
            "": pd.NA, "NA": pd.NA, "N/A": pd.NA, "n/a": pd.NA,
            "--": pd.NA, "-": pd.NA, "None": pd.NA,
        }
    )
    ss = ss.str.replace(",", "", regex=False)
    ss = ss.str.replace("−", "-", regex=False)  # unicode minus
    extracted = ss.str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")

def coerce_numeric_like_columns(df_in: pd.DataFrame, min_success: float = 0.60) -> pd.DataFrame:
    """
    For object/string columns (excluding NON_NUMERIC_COLS), try to parse numbers.
    If parsing succeeds for >= min_success fraction of non-null cells, convert column to numeric.
    """
    df_out = df_in.copy()
    for col in df_out.columns:
        if col in NON_NUMERIC_COLS:
            continue
        if df_out[col].dtype == "object" or str(df_out[col].dtype).startswith("string"):
            converted = _parse_numeric_series(df_out[col])
            non_null = df_out[col].notna().sum()
            if non_null == 0:
                continue
            success = converted.notna().sum() / non_null
            if success >= min_success:
                df_out[col] = converted
    return df_out

def add_derived_metrics(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Adds per-capita/per-100/per-1k metrics used in infra analysis.
    """
    df_out = df_in.copy()

    pop = pd.to_numeric(df_out.get("Total_Population"), errors="coerce")
    if pop is None:
        return df_out

    def safe_rate(num: pd.Series, den: pd.Series) -> pd.Series:
        den2 = den.replace(0, np.nan)
        return num / den2

    # Subscriptions per 100 people
    for src, dst in [
        ("mobile_cellular_subscriptions_total", "mobile_subscriptions_per_100"),
        ("telephone_fixed_subscriptions_total", "fixed_telephone_per_100"),
        ("broadband_fixed_subscriptions_total", "fixed_broadband_per_100"),
        ("internet_users_total", "internet_users_per_100"),
    ]:
        if src in df_out.columns:
            num = pd.to_numeric(df_out[src], errors="coerce")
            df_out[dst] = safe_rate(num, pop) * 100.0

    # Length-based infrastructure per 1k people
    for src, dst in [
        ("roadways_km", "roadways_km_per_1k"),
        ("railways_km", "railways_km_per_1k"),
        ("waterways_km", "waterways_km_per_1k"),
        ("gas_pipelines_km", "gas_pipelines_km_per_1k"),
        ("oil_pipelines_km", "oil_pipelines_km_per_1k"),
        ("refined_products_pipelines_km", "refined_products_pipelines_km_per_1k"),
        ("water_pipelines_km", "water_pipelines_km_per_1k"),
    ]:
        if src in df_out.columns:
            num = pd.to_numeric(df_out[src], errors="coerce")
            df_out[dst] = safe_rate(num, pop) * 1000.0

    # Airports/heliports per 1M people
    for src, dst in [
        ("airports_paved_runways_count", "airports_paved_per_1M"),
        ("airports_unpaved_runways_count", "airports_unpaved_per_1M"),
        ("heliports_count", "heliports_per_1M"),
    ]:
        if src in df_out.columns:
            num = pd.to_numeric(df_out[src], errors="coerce")
            df_out[dst] = safe_rate(num, pop) * 1_000_000.0

    # Electricity capacity per capita (kW/person)
    if "electricity_generating_capacity_kW" in df_out.columns:
        num = pd.to_numeric(df_out["electricity_generating_capacity_kW"], errors="coerce")
        df_out["electricity_capacity_kW_per_capita"] = safe_rate(num, pop)

    # CO2 tons per capita (Mt -> metric tons)
    if "carbon_dioxide_emissions_Mt" in df_out.columns:
        num = pd.to_numeric(df_out["carbon_dioxide_emissions_Mt"], errors="coerce")
        df_out["co2_tons_per_capita"] = safe_rate(num * 1_000_000.0, pop)

    # Population density (people per km^2), prefer Land_Area else Area_Total
    for area_col in ["Land_Area", "Area_Total"]:
        if area_col in df_out.columns:
            area = pd.to_numeric(df_out[area_col], errors="coerce").replace(0, np.nan)
            df_out["population_density_per_km2"] = pop / area
            break

    return df_out

def add_percentiles_and_gaps(df_in: pd.DataFrame, gdp_col: str = "Real_GDP_per_Capita_USD") -> pd.DataFrame:
    """
    Adds:
      - gdp_pc_pct (percentile rank of GDP per capita)
      - <metric>_pct for key infra metrics
      - gap_<metric>_vs_gdp = <metric>_pct - gdp_pc_pct
      - infra_score_pct_mean and gap_infra_score_vs_gdp
    """
    df_out = df_in.copy()
    if gdp_col not in df_out.columns:
        return df_out

    gdp = pd.to_numeric(df_out[gdp_col], errors="coerce")
    df_out["gdp_pc_pct"] = gdp.rank(pct=True)

    infra_metrics = [
        "fixed_broadband_per_100",
        "mobile_subscriptions_per_100",
        "internet_users_per_100",
        "electricity_capacity_kW_per_capita",
        "roadways_km_per_1k",
        "railways_km_per_1k",
        "co2_tons_per_capita",
    ]

    pct_cols = []
    for m in infra_metrics:
        if m in df_out.columns:
            df_out[m + "_pct"] = pd.to_numeric(df_out[m], errors="coerce").rank(pct=True)
            pct_cols.append(m + "_pct")
            df_out["gap_" + m + "_vs_gdp"] = df_out[m + "_pct"] - df_out["gdp_pc_pct"]

    if pct_cols:
        df_out["infra_score_pct_mean"] = df_out[pct_cols].mean(axis=1, skipna=True)
        df_out["gap_infra_score_vs_gdp"] = df_out["infra_score_pct_mean"] - df_out["gdp_pc_pct"]

    return df_out

def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    df0 = pd.read_csv(csv_path)
    df1 = coerce_numeric_like_columns(df0)
    df2 = add_derived_metrics(df1)
    df3 = add_percentiles_and_gaps(df2)
    return df3

df = load_and_prepare_data("data/CIA_DATA.csv")

# -----------------------------------------------------------------------------
# Derived peer groups (no new data required)
# -----------------------------------------------------------------------------
if "gdp_pc_pct" in df.columns:
    df["income_band"] = pd.cut(
        df["gdp_pc_pct"],
        bins=[-0.01, 0.25, 0.75, 1.01],
        labels=["Low income", "Middle income", "High income"],
    )
else:
    df["income_band"] = pd.Series([pd.NA] * len(df), index=df.index)

POP_COL = "Total_Population" if "Total_Population" in df.columns else ("population" if "population" in df.columns else None)
if POP_COL is not None:
    df["population_band"] = pd.cut(
        pd.to_numeric(df[POP_COL], errors="coerce"),
        bins=[0, 5e6, 25e6, 100e6, 2e9],
        labels=[
            "Small (<5M)",
            "Medium (5–25M)",
            "Large (25–100M)",
            "Very large (>100M)",
        ],
    )
else:
    df["population_band"] = pd.Series([pd.NA] * len(df), index=df.index)

INCOME_BAND_ORDER = ["Low income", "Middle income", "High income"]
POP_BAND_ORDER = ["Small (<5M)", "Medium (5–25M)", "Large (25–100M)", "Very large (>100M)"]

INCOME_BAND_OPTIONS = [
    {"label": b, "value": b}
    for b in INCOME_BAND_ORDER
    if b in set(df["income_band"].astype("string").dropna().unique())
]
POP_BAND_OPTIONS = [
    {"label": b, "value": b}
    for b in POP_BAND_ORDER
    if b in set(df["population_band"].astype("string").dropna().unique())
]

def apply_peer_filters(base_df: pd.DataFrame, income_bands, population_bands) -> pd.DataFrame:
    out = base_df
    if income_bands:
        out = out[out["income_band"].isin(income_bands)]
    if population_bands:
        out = out[out["population_band"].isin(population_bands)]
    return out


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
    "mobile_subscriptions_per_100": "Mobile Subscriptions (per 100 people)",
    "fixed_telephone_per_100": "Fixed Telephone (per 100 people)",
    "fixed_broadband_per_100": "Fixed Broadband (per 100 people)",
    "internet_users_per_100": "Internet Users (per 100 people)",

    "roadways_km_per_1k": "Roadways (km per 1k people)",
    "railways_km_per_1k": "Railways (km per 1k people)",
    "waterways_km_per_1k": "Waterways (km per 1k people)",
    "gas_pipelines_km_per_1k": "Gas Pipelines (km per 1k people)",
    "oil_pipelines_km_per_1k": "Oil Pipelines (km per 1k people)",
    "refined_products_pipelines_km_per_1k": "Refined Product Pipelines (km per 1k people)",
    "water_pipelines_km_per_1k": "Water Pipelines (km per 1k people)",

    "airports_paved_per_1M": "Paved Airports (per 1M people)",
    "airports_unpaved_per_1M": "Unpaved Airports (per 1M people)",
    "heliports_per_1M": "Heliports (per 1M people)",

    "electricity_capacity_kW_per_capita": "Electricity Capacity (kW per person)",
    "co2_tons_per_capita": "CO₂ (tons per person)",
    "population_density_per_km2": "Population Density (per km²)",

    "gdp_pc_pct": "GDP per Capita (percentile)",
    "infra_score_pct_mean": "Infrastructure Score (mean percentile)",
    "gap_infra_score_vs_gdp": "Gap: Infra Score − GDP (pct points)",
}

GROUP_RULES = [
    ("Derived", ["gap_", "_pct", "score", "per_100", "per_capita", "per_1k", "per_1M", "density"]),
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

buckets = {g[0]: [] for g in GROUP_RULES}
for c in numeric_cols:
    buckets[assign_group(c)].append(c)

for g in buckets:
    buckets[g] = sorted(buckets[g], key=label_for)

if DEFAULT_METRIC is None and numeric_cols:
    DEFAULT_METRIC = numeric_cols[0]

# -----------------------------------------------------------------------------
# Global color encoding (used across summary plots)
# -----------------------------------------------------------------------------
GLOBAL_COLOR_METRIC = (
    "Real_GDP_per_Capita_USD"
    if "Real_GDP_per_Capita_USD" in df.columns
    else (numeric_cols[0] if numeric_cols else None)
)
GLOBAL_COLORSCALE = "Viridis"

def col_as_series(frame: pd.DataFrame, col: str) -> pd.Series:
    """
    Returns a 1D Series even if `col` is duplicated in the DataFrame.
    If duplicates exist, take the first occurrence.
    """
    obj = frame.loc[:, col]
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    return obj

# -----------------------------------------------------------------------------
# Figure builders
# -----------------------------------------------------------------------------
def make_map(metric: str, data_df: Optional[pd.DataFrame] = None):
    data_df = df if data_df is None else data_df

    if not metric or metric not in data_df.columns:
        fig = go.Figure()
        fig.update_layout(template="infra_light")
        return fig

    s = pd.to_numeric(col_as_series(data_df, metric), errors="coerce")
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
        data_df,
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
        marker_line_width=0,
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
    Returns a DataFrame with selected cols transformed for comparability.
    - pct: percentile (0..1)
    - z: z-score
    - log: log10(positive only -> others become NaN)
    - raw: numeric as-is
    """
    out = pd.DataFrame({COUNTRY_COL: df_in[COUNTRY_COL]})
    for c in cols:
        s = _to_numeric_series(df_in[c])

        if mode == "pct":
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
            mask = False
            for lo, hi in cr:
                mask = mask | ((s >= lo) & (s <= hi))
            d = d[mask]
    return d

def _unwrap_single_trace_value(v):
    if isinstance(v, list) and len(v) == 1:
        return v[0]
    return v

def parse_pcp_constraints(restyle, dims, existing: Optional[dict] = None) -> dict:
    existing = existing or {}
    if not restyle:
        return {}

    if isinstance(restyle, (list, tuple)) and len(restyle) > 0 and isinstance(restyle[0], dict):
        changes = restyle[0]
    elif isinstance(restyle, dict):
        changes = restyle
    else:
        return {}

    updates = {}
    for k, v in changes.items():
        if "constraintrange" not in k:
            continue

        m = re.match(r"dimensions\[(\d+)\]\.constraintrange(?:\[(\d+)\])?$", k)
        if not m:
            continue

        dim_idx = int(m.group(1))
        bound_idx = m.group(2)
        if dim_idx < 0 or dim_idx >= len(dims):
            continue

        dim_name = dims[dim_idx]
        v = _unwrap_single_trace_value(v)

        if bound_idx is None:
            updates[dim_name] = v
        else:
            bi = int(bound_idx)
            cur = existing.get(dim_name)
            cur = _unwrap_single_trace_value(cur)

            if not (isinstance(cur, list) and len(cur) == 2):
                cur = [None, None]

            vv = _unwrap_single_trace_value(v)
            cur2 = list(cur)
            cur2[bi] = vv
            updates[dim_name] = cur2

    return updates

def clean_pcp_constraints(constraints: dict, dims: list[str]) -> dict:
    if not constraints:
        return {}
    dims = dims or []
    out = {}
    for k, v in (constraints or {}).items():
        if k not in dims:
            continue
        if v is None:
            continue
        if isinstance(v, list) and len(v) == 0:
            continue
        out[k] = v
    return out

def make_pcp(
    dims: list[str],
    color_metric: str,
    scale_mode: str,
    constraints: dict,
    data_df: Optional[pd.DataFrame] = None,
):
    """
    Builds a go.Parcoords PCP.
    Constraints are applied to compute selected countries and shown on axes.
    """
    data_df = df if data_df is None else data_df

    if not dims:
        fig = go.Figure()
        fig.update_layout(
            template="infra_light",
            margin=dict(l=20, r=20, t=10, b=10),
            dragmode="pan",
        )
        return fig, []

    cols = dims + ([color_metric] if color_metric and color_metric not in dims else [])
    td = transform_for_pcp(data_df, cols, scale_mode)

    if not constraints:
        selected_countries = []
    else:
        td_filtered = apply_constraints(td, constraints)
        selected_countries = td_filtered[COUNTRY_COL].dropna().tolist()

    if color_metric and color_metric in td.columns:
        color_vals = td[color_metric]
    else:
        color_vals = td[dims[0]]

    dim_objs = []
    for c in dims:
        if c not in td.columns:
            continue
        dim = dict(label=label_for(c), values=td[c])
        if constraints and c in constraints and constraints[c] is not None:
            dim["constraintrange"] = constraints[c]
        dim_objs.append(dim)

    fig = go.Figure(
        data=[
            go.Parcoords(
                line=dict(color=color_vals, colorscale="Viridis", showscale=False),
                dimensions=dim_objs,
                labelfont=dict(color=THEME["text"]),
                tickfont=dict(color=THEME["muted_text"]),
            )
        ]
    )
    fig.update_layout(template="infra_light", margin=dict(l=20, r=20, t=10, b=10))
    return fig, selected_countries

def make_map_with_selection(
    metric: str,
    selected_countries: list[str],
    data_df: Optional[pd.DataFrame] = None,
):
    """
    Choropleth based on `data_df`, plus an outline on selected countries.
    """
    data_df = df if data_df is None else data_df
    fig = make_map(metric, data_df=data_df)

    if not selected_countries:
        return fig

    peer_set = set(data_df[COUNTRY_COL].dropna().tolist())
    selected_countries = [c for c in selected_countries if c in peer_set]
    if not selected_countries:
        return fig

    outline = go.Choropleth(
        locations=selected_countries,
        locationmode="country names",
        z=[1] * len(selected_countries),
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
        showscale=False,
        hoverinfo="skip",
        marker=dict(line=dict(color="rgba(59,130,246,0.95)", width=2)),
    )

    fig.add_trace(outline)
    return fig


def make_opp_risk_scatter(
    x_metric: str,
    y_metric: str,
    selected_countries: list[str],
    color_metric=None,
    data_df: Optional[pd.DataFrame] = None,
):
    data_df = df if data_df is None else data_df

    if not x_metric or not y_metric or x_metric not in data_df.columns or y_metric not in data_df.columns:
        fig = go.Figure()
        fig.update_layout(template="infra_light")
        return fig

    selected_countries = selected_countries or []

    if not color_metric or color_metric not in data_df.columns:
        color_metric = GLOBAL_COLOR_METRIC if (GLOBAL_COLOR_METRIC and GLOBAL_COLOR_METRIC in data_df.columns) else None

    x = pd.to_numeric(col_as_series(data_df, x_metric), errors="coerce")
    y = pd.to_numeric(col_as_series(data_df, y_metric), errors="coerce")

    plot_df = pd.DataFrame({COUNTRY_COL: data_df[COUNTRY_COL], "_x": x, "_y": y}).dropna(subset=["_x", "_y"])
    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(template="infra_light")
        return fig

    use_color = False
    plot_col = plot_df
    if color_metric:
        c = pd.to_numeric(col_as_series(data_df, color_metric), errors="coerce")
        plot_df["_c"] = c
        non_na = plot_df["_c"].notna().sum()
        if non_na >= max(10, int(0.25 * len(plot_df))):
            use_color = True
            plot_col = plot_df.dropna(subset=["_c"])

    if use_color:
        fig = px.scatter(
            plot_col,
            x="_x",
            y="_y",
            color="_c",
            color_continuous_scale=GLOBAL_COLORSCALE,
            hover_name=COUNTRY_COL,
            custom_data=[COUNTRY_COL],  # ✅ ADD THIS
            labels={"_x": label_for(x_metric), "_y": label_for(y_metric), "_c": label_for(color_metric)},
            template="infra_light",
        )
        fig.update_layout(coloraxis_showscale=False)
    else:
        fig = px.scatter(
            plot_df,
            x="_x",
            y="_y",
            hover_name=COUNTRY_COL,
            custom_data=[COUNTRY_COL],  # ✅ ADD THIS
            labels={"_x": label_for(x_metric), "_y": label_for(y_metric)},
            template="infra_light",
        )
        fig.update_traces(marker=dict(color="rgba(17,24,39,0.55)"))

    # ✅ Enable selection + keep it clean
    fig.update_layout(
        dragmode="select",          # box-select by drag (works even with hidden modebar)
        clickmode="event+select",   # click selects a point
    )

    # ✅ Remove outlines on normal points (keeps colors readable)
    fig.update_traces(
        marker=dict(size=8, opacity=0.85, line=dict(width=0)),
        selector=dict(mode="markers"),
    )
    
    
    fig = card_layout(fig, x_title=label_for(x_metric), y_title=label_for(y_metric))
    return fig

def make_ranked_bar(metric: str, selected_countries: list[str], top_n=10, data_df: Optional[pd.DataFrame] = None):
    data_df = df if data_df is None else data_df

    if not metric or metric not in data_df.columns:
        fig = go.Figure()
        fig.update_layout(template="infra_light")
        return fig

    selected_countries = selected_countries or []

    values = pd.to_numeric(col_as_series(data_df, metric), errors="coerce")
    plot_df = pd.DataFrame({COUNTRY_COL: data_df[COUNTRY_COL], "_v": values}).dropna(subset=["_v"])
    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(template="infra_light")
        return fig

    plot_df = plot_df.sort_values("_v", ascending=False)
    top = plot_df.head(top_n)
    bottom = plot_df.tail(top_n)
    out = pd.concat([top, bottom], axis=0)

    out = out.iloc[::-1]

    fig = px.bar(
        out,
        x="_v",
        y=COUNTRY_COL,
        orientation="h",
        color="_v",
        color_continuous_scale=GLOBAL_COLORSCALE,
        template="infra_light",
        labels={"_v": label_for(metric), COUNTRY_COL: ""},
    )

    fig.update_layout(coloraxis_showscale=False, barmode="overlay")

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

def make_distribution(metric: str, selected_country=None, data_df: Optional[pd.DataFrame] = None):
    data_df = df if data_df is None else data_df

    if not metric or metric not in data_df.columns:
        fig = go.Figure()
        fig.update_layout(template="infra_light")
        return fig

    s = pd.to_numeric(col_as_series(data_df, metric), errors="coerce").dropna()
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=s,
            nbinsx=30,
            marker_color="rgba(17,24,39,0.35)",
            hovertemplate=f"{label_for(metric)} range<br>Count: %{{y}}<extra></extra>",
        )
    )

    for q in [0.25, 0.5, 0.75]:
        fig.add_vline(
            x=s.quantile(q),
            line_width=1,
            line_dash="dot",
            line_color=THEME["muted_text"],
        )

    if selected_country:
        ser_all = col_as_series(data_df, metric)
        ser = pd.to_numeric(ser_all[data_df[COUNTRY_COL] == selected_country], errors="coerce")
        if len(ser) > 0:
            val = ser.iloc[0]
            if pd.notna(val):
                fig.add_vrect(x0=val, x1=val, fillcolor="rgba(59,130,246,0.15)", line_width=0)
                fig.add_vline(x=val, line_width=2, line_color="rgba(59,130,246,0.95)")

    fig.update_layout(template="infra_light", showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    fig = card_layout(fig, x_title=label_for(metric), y_title="Countries")
    return fig

def make_gap_scatter(
    selected_countries: list[str],
    selected_country: Optional[str] = None,
    data_df: Optional[pd.DataFrame] = None,
):
    data_df = df if data_df is None else data_df

    required = {"gdp_pc_pct", "infra_score_pct_mean", "gap_infra_score_vs_gdp"}
    if not required.issubset(data_df.columns):
        fig = go.Figure()
        fig.update_layout(template="infra_light")
        return fig

    selected_countries = selected_countries or []

    plot_df = data_df[[COUNTRY_COL, "gdp_pc_pct", "infra_score_pct_mean", "gap_infra_score_vs_gdp"]].copy()
    plot_df = plot_df.dropna(subset=["gdp_pc_pct", "infra_score_pct_mean"])
    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(template="infra_light")
        return fig

    plot_df["_gdp_pp"] = pd.to_numeric(plot_df["gdp_pc_pct"], errors="coerce") * 100.0
    plot_df["_infra_pp"] = pd.to_numeric(plot_df["infra_score_pct_mean"], errors="coerce") * 100.0
    plot_df["_gap_pp"] = pd.to_numeric(plot_df["gap_infra_score_vs_gdp"], errors="coerce") * 100.0

    fig = px.scatter(
        plot_df,
        x="_gdp_pp",
        y="_infra_pp",
        color="_gap_pp",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        hover_name=COUNTRY_COL,
        labels={
            "_gdp_pp": "GDP per Capita (percentile points)",
            "_infra_pp": "Infrastructure Score (percentile points)",
            "_gap_pp": "Gap (Infra − GDP) (pp)",
        },
        template="infra_light",
    )

    fig.add_shape(
        type="line",
        x0=0, y0=0, x1=100, y1=100,
        xref="x", yref="y",
        line=dict(width=1, dash="dot", color="rgba(107,114,128,0.9)"),
        layer="below",
    )

    if selected_countries:
        sel = plot_df[plot_df[COUNTRY_COL].isin(selected_countries)]
        if len(sel) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sel["_gdp_pp"],
                    y=sel["_infra_pp"],
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

    if selected_country:
        row = plot_df[plot_df[COUNTRY_COL] == selected_country]
        if len(row) > 0:
            fig.add_trace(
                go.Scatter(
                    x=row["_gdp_pp"],
                    y=row["_infra_pp"],
                    mode="markers+text",
                    marker=dict(size=16, symbol="star"),
                    text=[selected_country],
                    textposition="top center",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.update_layout(coloraxis_showscale=False)
    fig = card_layout(fig, x_title="GDP per Capita (percentile points)", y_title="Infrastructure Score (percentile points)")
    return fig

def make_gap_ranked_bar(
    selected_countries: list[str],
    top_n: int = 10,
    data_df: Optional[pd.DataFrame] = None,
):
    data_df = df if data_df is None else data_df

    if "gap_infra_score_vs_gdp" not in data_df.columns:
        fig = go.Figure()
        fig.update_layout(template="infra_light")
        return fig

    selected_countries = selected_countries or []

    s = pd.to_numeric(col_as_series(data_df, "gap_infra_score_vs_gdp"), errors="coerce") * 100.0
    plot_df = pd.DataFrame({COUNTRY_COL: data_df[COUNTRY_COL], "_gap_pp": s}).dropna(subset=["_gap_pp"])
    if plot_df.empty:
        fig = go.Figure()
        fig.update_layout(template="infra_light")
        return fig

    plot_df = plot_df.sort_values("_gap_pp", ascending=False)
    top = plot_df.head(top_n)
    bottom = plot_df.tail(top_n)
    out = pd.concat([top, bottom], axis=0).drop_duplicates(subset=[COUNTRY_COL], keep="first")
    out = out.sort_values("_gap_pp", ascending=True)

    fig = px.bar(
        out,
        x="_gap_pp",
        y=COUNTRY_COL,
        orientation="h",
        color="_gap_pp",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        template="infra_light",
        labels={"_gap_pp": "Gap (Infra − GDP) (pp)", COUNTRY_COL: ""},
    )

    fig.update_layout(coloraxis_showscale=False)

    if selected_countries:
        sel = out[out[COUNTRY_COL].isin(selected_countries)]
        if len(sel) > 0:
            fig.add_trace(
                go.Bar(
                    x=sel["_gap_pp"],
                    y=sel[COUNTRY_COL],
                    orientation="h",
                    marker=dict(color="rgba(59,130,246,0.95)", line=dict(color="white", width=1.5)),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Gap (Infra − GDP): %{x:.1f} pp<extra></extra>",
        opacity=0.92,
    )

    fig = card_layout(fig, x_title="Gap (Infra − GDP) (percentile points)")
    return fig

def card_layout(fig, *, x_title=None, y_title=None):
    fig.update_layout(
        margin=dict(l=48, r=16, t=8, b=40),
        xaxis=dict(title=None, tickfont=dict(size=11)),
        yaxis=dict(title=None, tickfont=dict(size=11)),
    )

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
        dcc.Store(id="active-country", data=None),
        dcc.Store(id="selected-countries", data=[]),        # shared multi-country selection (empty = no subset)
        dcc.Store(id="selected-country-store", data=None),  # reserved for later (single country selection, not used yet)



        html.Div(
            className="peer-filter-bar",
            style={
                "display": "flex",
                "gap": "12px",
                "alignItems": "end",
                "padding": "10px 12px",
                "margin": "10px 0",
                "background": THEME["panel"],
                "border": f"1px solid {THEME['border']}",
                "borderRadius": "12px",
            },
            children=[
                html.Div(
                    style={"minWidth": "240px"},
                    children=[
                        html.Label("Income band (peer group)", className="control-label"),
                        dcc.Dropdown(
                            id="income-band-filter",
                            className="light-dropdown",
                            options=INCOME_BAND_OPTIONS,
                            value=[],
                            multi=True,
                            clearable=True,
                            placeholder="All income bands",
                        ),
                    ],
                ),
                html.Div(
                    style={"minWidth": "260px"},
                    children=[
                        html.Label("Population band (peer group)", className="control-label"),
                        dcc.Dropdown(
                            id="population-band-filter",
                            className="light-dropdown",
                            options=POP_BAND_OPTIONS,
                            value=[],
                            multi=True,
                            clearable=True,
                            placeholder="All population bands",
                        ),
                    ],
                ),
                html.Button(
                    "Clear peer filters",
                    id="peer-reset",
                    n_clicks=0,
                    className="btn",
                    style={"height": "36px"},
                ),
            ],
        ),

        html.Div(
            id="global-view",
            children=[
                html.Div(
                    className="top-row",
                    children=[
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

                        html.Div(
                            className="panel panel-tight",
                            children=[
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
                                                html.Label("Color variable:", className="control-label"),
                                                html.Div("*The variable selected for color is the variable set for the color scale in the plots below as well.", className="control-label"),
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
                                                html.Div(id="pcp-status", className="status-pill", children="Subset: none (showing all countries)"),
                                            ],
                                        ),
                                    ],
                                ),

                                dcc.Graph(
                                    id="pcp",
                                    className="panel-content",
                                    config={
                                        "displayModeBar": False,
                                        "responsive": True,
                                        "scrollZoom": False,
                                        "doubleClick": "reset",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),

                html.Div(
                    className="app-grid2",
                    children=[
                        html.Div(
                            className="panel panel-tall",
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
                                html.Div("Infrastructure vs GDP (Gap view)", className="panel-title"),
                                dcc.Graph(
                                    id="gap-scatter",
                                    className="panel-content",
                                    config={"displayModeBar": False, "responsive": True},
                                ),
                            ],
                        ),
                        html.Div(
                            className="panel",
                            children=[
                                html.Div("Gap leaders / laggards", className="panel-title"),
                                dcc.Graph(
                                    id="gap-ranked",
                                    className="panel-content",
                                    config={"displayModeBar": False, "responsive": True},
                                ),
                            ],
                        ),
                        html.Div(className="panel", children=[html.Div("View 6", className="panel-title"), html.Div("Placeholder", className="panel-placeholder")]),
                        html.Div(className="panel", children=[html.Div("View 7", className="panel-title"), html.Div("Placeholder", className="panel-placeholder")]),
                        html.Div(className="panel", children=[html.Div("View 8", className="panel-title"), html.Div("Placeholder", className="panel-placeholder")]),
                        html.Div(className="panel", children=[html.Div("View 9", className="panel-title"), html.Div("Placeholder", className="panel-placeholder")]),
                        html.Div(className="panel", children=[html.Div("View 10", className="panel-title"), html.Div("Placeholder", className="panel-placeholder")]),
                        html.Div(className="panel", children=[html.Div("View 11", className="panel-title"), html.Div("Placeholder", className="panel-placeholder")]),
                        html.Div(className="panel", children=[html.Div("View 12", className="panel-title"), html.Div("Placeholder", className="panel-placeholder")]),
                    ],
                ),
            ],
        ),

        html.Div(
            id="country-view",
            style={"display": "none"},
            children=[
                html.Div(
                    id="country-header",
                    className="selected-country",
                    style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "gap": "12px"},
                    children=[
                        html.Button("← Back to global overview", id="back-to-global", n_clicks=0, className="btn"),
                        html.Div(id="country-title", style={"fontWeight": "600"}),
                    ],
                ),
                html.Div(
                    className="app-grid2",
                    children=[
                        html.Div(
                            className="panel panel-tall",
                            style={"gridColumn": "span 2"},
                            children=[html.Div("Economic Snapshot", className="panel-title"), dcc.Graph(id="country-econ", className="panel-content")],
                        ),
                        html.Div(
                            className="panel panel-tall",
                            style={"gridColumn": "span 2"},
                            children=[html.Div("Labor & Demographics", className="panel-title"), dcc.Graph(id="country-labor", className="panel-content")],
                        ),
                        html.Div(
                            className="panel",
                            style={"gridColumn": "span 2"},
                            children=[html.Div("Infrastructure Snapshot", className="panel-title"), dcc.Graph(id="country-infra", className="panel-content")],
                        ),
                        html.Div(
                            className="panel",
                            style={"gridColumn": "span 2"},
                            children=[html.Div("Global Rank Context", className="panel-title"), dcc.Graph(id="country-rank", className="panel-content")],
                        ),
                    ],
                ),
            ],
        ),
    ],
)

# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
@app.callback(
    Output("income-band-filter", "value"),
    Output("population-band-filter", "value"),
    Input("peer-reset", "n_clicks"),
    prevent_initial_call=True,
)
def clear_peer_filters(_):
    return [], []

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
    Input("pcp-dims", "value"),
    Input("pcp-color", "value"),
    Input("pcp-scale", "value"),
    Input("pcp", "restyleData"),
    Input("pcp-reset", "n_clicks"),
    Input("income-band-filter", "value"),
    Input("population-band-filter", "value"),
    State("pcp-constraints", "data"),
)
def update_pcp_and_map(dims, color_metric, scale_mode, restyle, reset_clicks,
                       income_bands, population_bands, stored_constraints):
    trig = (callback_context.triggered[0]["prop_id"] if callback_context.triggered else "")
    dims = dims or []

    if "pcp-reset" in trig:
        constraints = {}
    elif "pcp.restyleData" in trig:
        incoming = parse_pcp_constraints(restyle, dims or [], existing=stored_constraints or {})
        merged = dict(stored_constraints or {})
        merged.update(incoming)
        constraints = merged
    else:
        constraints = stored_constraints or {}

    constraints = clean_pcp_constraints(constraints, dims)

    plot_df = apply_peer_filters(df, income_bands, population_bands)

    pcp_fig, selected = make_pcp(dims, color_metric, scale_mode, constraints, data_df=plot_df)

    if constraints:
        status = f"Peer group: {len(plot_df)} countries — Subset: {len(selected)} countries"
    else:
        status = f"Peer group: {len(plot_df)} countries — Subset: none (showing peer group)"

    return pcp_fig, selected, constraints, status

@app.callback(
    Output("world-map", "figure"),
    Input("map-metric", "value"),
    Input("selected-countries", "data"),
    Input("income-band-filter", "value"),
    Input("population-band-filter", "value"),
)
def update_world_map(map_metric, selected_countries, income_bands, population_bands):
    plot_df = apply_peer_filters(df, income_bands, population_bands)
    return make_map_with_selection(map_metric, selected_countries or [], data_df=plot_df)


@app.callback(
    Output("opp-risk-scatter", "figure"),
    Input("map-metric", "value"),
    Input("pcp-color", "value"),
    Input("income-band-filter", "value"),
    Input("population-band-filter", "value"),
)
def update_opp_risk_scatter(metric, pcp_color, income_bands, population_bands):
    opportunity_metric = metric
    risk_metric = "Public_Debt_percent_of_GDP" if "Public_Debt_percent_of_GDP" in df.columns else metric

    plot_df = apply_peer_filters(df, income_bands, population_bands)

    # IMPORTANT: do not pass selected-countries here (prevents selection reset flicker)
    return make_opp_risk_scatter(
        opportunity_metric,
        risk_metric,
        selected_countries=[],        # ✅ keep scatter independent of server-side selection
        color_metric=pcp_color,
        data_df=plot_df,
    )

@app.callback(
    Output("ranked-bar", "figure"),
    Output("metric-distribution", "figure"),
    Input("map-metric", "value"),
    Input("selected-countries", "data"),
    Input("world-map", "clickData"),
    Input("income-band-filter", "value"),
    Input("population-band-filter", "value"),
)
def update_bar_and_distribution(metric, selected_countries, map_click, income_bands, population_bands):
    selected_country = None
    if map_click:
        selected_country = map_click["points"][0]["location"]

    plot_df = apply_peer_filters(df, income_bands, population_bands)

    bars = make_ranked_bar(metric, selected_countries, data_df=plot_df)
    dist = make_distribution(metric, selected_country, data_df=plot_df)
    return bars, dist


@app.callback(
    Output("gap-scatter", "figure"),
    Output("gap-ranked", "figure"),
    Input("selected-countries", "data"),
    Input("world-map", "clickData"),
    Input("income-band-filter", "value"),
    Input("population-band-filter", "value"),
)
def update_gap_views(selected_countries, map_click, income_bands, population_bands):
    selected_country = None
    if map_click:
        selected_country = map_click["points"][0]["location"]

    plot_df = apply_peer_filters(df, income_bands, population_bands)

    gap_scatter = make_gap_scatter(selected_countries, selected_country=selected_country, data_df=plot_df)
    gap_ranked = make_gap_ranked_bar(selected_countries, top_n=10, data_df=plot_df)

    return gap_scatter, gap_ranked

@app.callback(
    Output("active-country", "data"),
    Input("world-map", "clickData"),
    Input("pcp-reset", "n_clicks"),
    Input("back-to-global", "n_clicks"),
)
def set_active_country(map_click, reset_clicks, back_clicks):
    ctx = (callback_context.triggered[0]["prop_id"] if callback_context.triggered else "")

    if "pcp-reset" in ctx or "back-to-global" in ctx:
        return None

    if "world-map" in ctx and map_click:
        return map_click["points"][0]["location"]

    return None

@app.callback(
    Output("global-view", "style"),
    Output("country-view", "style"),
    Output("country-title", "children"),
    Input("active-country", "data"),
)
def toggle_dashboard_views(country):
    if country:
        return (
            {"display": "none"},
            {"display": "block"},
            f"Country focus: {country}",
        )

    return (
        {"display": "block"},
        {"display": "none"},
        "",
    )

@app.callback(
    Output("country-econ", "figure"),
    Output("country-labor", "figure"),
    Output("country-infra", "figure"),
    Output("country-rank", "figure"),
    Input("active-country", "data"),
    Input("income-band-filter", "value"),
    Input("population-band-filter", "value"),
)
def update_country_dashboard(country, income_bands, population_bands):
    empty = go.Figure().update_layout(template="infra_light")
    if not country:
        return empty, empty, empty, empty

    row_df = df[df[COUNTRY_COL] == country]
    if row_df.empty:
        return empty, empty, empty, empty
    row = row_df.iloc[0]

    econ_fig = go.Figure(
        go.Bar(
            x=["GDP per Capita", "Public Debt"],
            y=[row.get("Real_GDP_per_Capita_USD"), row.get("Public_Debt_percent_of_GDP")],
        )
    ).update_layout(template="infra_light")

    labor_fig = go.Figure(
        go.Bar(
            x=["Unemployment", "Youth Unemployment"],
            y=[row.get("Unemployment_Rate_percent"), row.get("Youth_Unemployment_Rate_percent")],
        )
    ).update_layout(template="infra_light")

    infra_fig = go.Figure(
        go.Bar(
            x=["Electricity Access", "Roadways"],
            y=[row.get("electricity_access_percent"), row.get("roadways_km")],
        )
    ).update_layout(template="infra_light")

    peer_df = apply_peer_filters(df, income_bands, population_bands)
    rank_fig = make_ranked_bar("Real_GDP_per_Capita_USD", [country], data_df=peer_df)

    return econ_fig, labor_fig, infra_fig, rank_fig


@app.callback(
    Output("selected-countries", "data"),
    Input("pcp-selected-countries", "data"),
    Input("opp-risk-scatter", "selectedData"),
    Input("pcp-reset", "n_clicks"),
    Input("income-band-filter", "value"),
    Input("population-band-filter", "value"),
    State("selected-countries", "data"),
)
def sync_selected_countries(pcp_selected, scatter_selected, _reset_clicks,
                           income_bands, population_bands, current_selected):
    ctx = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""

    peer_df = apply_peer_filters(df, income_bands, population_bands)
    peer_set = set(peer_df[COUNTRY_COL].dropna().tolist())

    def _intersect(countries):
        countries = countries or []
        out = []
        seen = set()
        for c in countries:
            if c and c in peer_set and c not in seen:
                seen.add(c)
                out.append(c)
        return out

    if "pcp-reset" in ctx:
        return []

    if "opp-risk-scatter.selectedData" in ctx:
        if not scatter_selected or not scatter_selected.get("points"):
            return []
        countries = []
        for p in scatter_selected["points"]:
            cd = p.get("customdata")
            if isinstance(cd, (list, tuple)) and len(cd) > 0:
                countries.append(cd[0])
            else:
                ht = p.get("hovertext")
                if ht:
                    countries.append(ht)
        return _intersect(countries)

    if "pcp-selected-countries.data" in ctx:
        return _intersect(pcp_selected)

    # if peer filters changed (or anything else), keep selection but clip to peer set
    return _intersect(current_selected)


if __name__ == "__main__":
    app.run(debug=True)
