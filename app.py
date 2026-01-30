from dash import Dash, html, dcc, dash_table, Input, Output, State, callback_context, no_update
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
    "background": "#989898",   # page background (very light gray)
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
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            color=THEME["text"],
            size=12,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)"),

        # ✅ subtle grid so the gray plot area reads cleanly
        xaxis=dict(
            gridcolor="rgba(17,24,39,0.08)",
            zerolinecolor="rgba(17,24,39,0.10)",
        ),
        yaxis=dict(
            gridcolor="rgba(17,24,39,0.08)",
            zerolinecolor="rgba(17,24,39,0.10)",
        ),
    )
)



# -----------------------------------------------------------------------------
# Data (load + clean + derive)
# -----------------------------------------------------------------------------
COUNTRY_COL = "Country"
EXCLUDED_ENTITIES = {"world"}

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
DEFAULT_COUNTRY_METRIC = (
    "Real_GDP_per_Capita_USD"
    if "Real_GDP_per_Capita_USD" in df.columns
    else (numeric_cols[0] if numeric_cols else None)
)

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

# --------------------------------------------------------------------------
# Analyst score builder (weighted composite)
# --------------------------------------------------------------------------
SCORE_SLOTS = 4

DEFAULT_SCORE_METRICS = [
    "Real_GDP_per_Capita_USD",
    "electricity_access_percent",
    "fixed_broadband_per_100",
    "roadways_km_per_1k",
]
DEFAULT_SCORE_WEIGHTS = [3, 2, 2, 1]   # relative weights (0 disables a metric)
DEFAULT_SCORE_DIRS = ["high", "high", "high", "high"]  # "high" or "low"

COUNTRY_PROFILE_METRICS = [
    "Real_GDP_per_Capita_USD",
    "Public_Debt_percent_of_GDP",
    "Unemployment_Rate_percent",
    "electricity_access_percent",
    "internet_users_per_100",
    "fixed_broadband_per_100",
    "roadways_km_per_1k",
    "population_density_per_km2",
]

def _safe_pct(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series([np.nan] * len(series), index=series.index)
    return s.rank(pct=True)

def build_score_spec(metrics, weights, dirs):
    spec = []
    for m, w, d in zip(metrics, weights, dirs):
        if m and (w is not None) and float(w) > 0 and (m in df.columns):
            spec.append({"metric": m, "weight": float(w), "direction": (d or "high")})
    return spec

def compute_weighted_score(data_df: pd.DataFrame, spec: list[dict]) -> pd.DataFrame:
    """
    Returns df with columns:
      Country, score_0_100
    Score is weighted mean of per-metric percentiles (0..1), optionally inverted for "low".
    Rows with missing metrics re-normalize weights by available metrics.
    """
    out = pd.DataFrame({COUNTRY_COL: data_df[COUNTRY_COL].astype("string")})

    if not spec:
        out["score_0_100"] = np.nan
        return out

    pct_cols = []
    w_cols = []

    for i, item in enumerate(spec):
        m = item["metric"]
        w = float(item["weight"])
        direction = item["direction"]

        pct = _safe_pct(col_as_series(data_df, m))
        if direction == "low":
            pct = 1.0 - pct

        col_pct = f"_pct_{i}"
        col_w = f"_w_{i}"
        out[col_pct] = pct
        out[col_w] = w

        pct_cols.append(col_pct)
        w_cols.append(col_w)

    pct_mat = out[pct_cols].to_numpy(dtype=float)
    w_vecs = out[w_cols].to_numpy(dtype=float)

    # mask where pct is valid
    valid = ~np.isnan(pct_mat)

    # effective weights only where valid
    eff_w = np.where(valid, w_vecs, 0.0)

    denom = eff_w.sum(axis=1)
    numer = (pct_mat * eff_w).sum(axis=1)

    score01 = np.where(denom > 0, numer / denom, np.nan)
    out["score_0_100"] = score01 * 100.0
    return out

def make_score_ranked_bar(score_df: pd.DataFrame, selected_countries: list[str], selected_country: Optional[str]):
    s = pd.to_numeric(score_df["score_0_100"], errors="coerce")
    plot_df = pd.DataFrame({COUNTRY_COL: score_df[COUNTRY_COL], "_score": s}).dropna(subset=["_score"])
    if plot_df.empty:
        fig = go.Figure().update_layout(template="infra_light")
        return fig

    plot_df = plot_df.sort_values("_score", ascending=False)
    top = plot_df.head(12)
    bottom = plot_df.tail(12)
    out = pd.concat([top, bottom], axis=0).drop_duplicates(subset=[COUNTRY_COL], keep="first")
    out = out.sort_values("_score", ascending=True)

    fig = px.bar(
        out,
        x="_score",
        y=COUNTRY_COL,
        orientation="h",
        color="_score",
        color_continuous_scale=GLOBAL_COLORSCALE,
        template="infra_light",
        labels={"_score": "Composite score (0–100)", COUNTRY_COL: ""},
    )
    fig.update_layout(coloraxis_showscale=False)

    # highlight subset selection (from PCP/scatter)
    selected_countries = selected_countries or []
    if selected_countries:
        sel = out[out[COUNTRY_COL].isin(selected_countries)]
        if not sel.empty:
            fig.add_trace(
                go.Bar(
                    x=sel["_score"],
                    y=sel[COUNTRY_COL],
                    orientation="h",
                    marker=dict(color="rgba(59,130,246,0.95)", line=dict(color="white", width=1.5)),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # highlight clicked country
    if selected_country:
        row = out[out[COUNTRY_COL] == selected_country]
        if not row.empty:
            fig.add_trace(
                go.Scatter(
                    x=row["_score"],
                    y=row[COUNTRY_COL],
                    mode="markers",
                    marker=dict(size=14, symbol="star"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>",
        opacity=0.92,
    )
    fig = card_layout(fig, x_title="Composite score (0–100)")
    return fig

def make_score_distribution(score_df: pd.DataFrame, selected_country: Optional[str]):
    s = pd.to_numeric(score_df["score_0_100"], errors="coerce").dropna()
    if s.empty:
        fig = go.Figure().update_layout(template="infra_light")
        return fig

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=s,
            nbinsx=25,
            marker_color="rgba(17,24,39,0.35)",
            hovertemplate="Score range<br>Count: %{y}<extra></extra>",
        )
    )

    for q in [0.25, 0.5, 0.75]:
        fig.add_vline(x=s.quantile(q), line_width=1, line_dash="dot", line_color=THEME["muted_text"])

    if selected_country:
        row = score_df[score_df[COUNTRY_COL] == selected_country]
        if not row.empty:
            val = pd.to_numeric(row["score_0_100"].iloc[0], errors="coerce")
            if pd.notna(val):
                fig.add_vrect(x0=val, x1=val, fillcolor="rgba(59,130,246,0.15)", line_width=0)
                fig.add_vline(x=val, line_width=2, line_color="rgba(59,130,246,0.95)")

    fig.update_layout(template="infra_light", showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    fig = card_layout(fig, x_title="Composite score (0–100)", y_title="Countries")
    return fig


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
    plot_df[COUNTRY_COL] = plot_df[COUNTRY_COL].astype("string")
    plot_df = plot_df[~plot_df[COUNTRY_COL].str.strip().str.lower().isin(EXCLUDED_ENTITIES)]
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
    plot_df[COUNTRY_COL] = plot_df[COUNTRY_COL].astype("string")
    plot_df = plot_df[~plot_df[COUNTRY_COL].str.strip().str.lower().isin(EXCLUDED_ENTITIES)]
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
    plot_df[COUNTRY_COL] = plot_df[COUNTRY_COL].astype("string")
    plot_df = plot_df[~plot_df[COUNTRY_COL].str.strip().str.lower().isin(EXCLUDED_ENTITIES)]
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
    # Use real axis titles + automargins (prevents spillover into other panels)
    fig.update_layout(
        margin=dict(l=70, r=24, t=12, b=60),
    )

    fig.update_xaxes(
        title_text=(x_title or ""),
        automargin=True,
        title_font=dict(size=11, color=THEME["muted_text"]),
        tickfont=dict(size=11),
    )
    fig.update_yaxes(
        title_text=(y_title or ""),
        automargin=True,
        title_font=dict(size=11, color=THEME["muted_text"]),
        tickfont=dict(size=11),
    )
    return fig


def format_value(v):
    if v is None:
        return "-"
    try:
        if pd.isna(v):
            return "-"
    except Exception:
        pass
    try:
        num = float(v)
    except Exception:
        return str(v)
    if np.isnan(num):
        return "-"
    if float(num).is_integer():
        return f"{int(num):,}"
    return f"{num:,.2f}"


def peer_percentile(peer_df, country, metric):
    if metric not in peer_df.columns:
        return np.nan
    series = pd.to_numeric(col_as_series(peer_df, metric), errors="coerce").dropna()
    if series.empty:
        return np.nan

    value = np.nan
    if country:
        row = peer_df[peer_df[COUNTRY_COL] == country]
        if row.empty:
            row = df[df[COUNTRY_COL] == country]
        if not row.empty:
            value = pd.to_numeric(row[metric].iloc[0], errors="coerce")

    if pd.isna(value):
        return np.nan

    return (series <= value).sum() / len(series) * 100.0


def series_percentile(series: pd.Series, value) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty or pd.isna(value):
        return np.nan
    return (s <= value).sum() / len(s) * 100.0


def compute_opportunity_risk_scores(peer_df: pd.DataFrame):
    out = pd.DataFrame({COUNTRY_COL: peer_df[COUNTRY_COL].astype("string")})

    opp_cols = []
    opp_labels = []
    risk_cols = []
    risk_labels = []

    def _add_pct(col, out_col, label):
        s = pd.to_numeric(col_as_series(peer_df, col), errors="coerce")
        out[out_col] = _safe_pct(s)
        return label

    # Opportunity components (higher = better)
    if "Real_GDP_Growth_Rate_percent" in peer_df.columns:
        opp_cols.append("_opp_gdp_growth")
        opp_labels.append(_add_pct("Real_GDP_Growth_Rate_percent", "_opp_gdp_growth", "GDP growth"))
    if "Population_Growth_Rate" in peer_df.columns:
        opp_cols.append("_opp_pop_growth")
        opp_labels.append(_add_pct("Population_Growth_Rate", "_opp_pop_growth", "Population growth"))
    if {"gdp_pc_pct", "infra_score_pct_mean"}.issubset(peer_df.columns):
        gap = pd.to_numeric(peer_df["gdp_pc_pct"], errors="coerce") - pd.to_numeric(peer_df["infra_score_pct_mean"], errors="coerce")
        out["_opp_infra_gap"] = _safe_pct(gap)
        opp_cols.append("_opp_infra_gap")
        opp_labels.append("Infra gap vs GDP")

    # Fallback if no components are available
    if not opp_cols and "gdp_pc_pct" in peer_df.columns:
        out["_opp_gdp_pc"] = pd.to_numeric(peer_df["gdp_pc_pct"], errors="coerce")
        opp_cols.append("_opp_gdp_pc")
        opp_labels.append("GDP per capita (pct)")

    # Risk components (higher = riskier)
    if "Public_Debt_percent_of_GDP" in peer_df.columns:
        risk_cols.append("_risk_debt")
        risk_labels.append(_add_pct("Public_Debt_percent_of_GDP", "_risk_debt", "Public debt"))
    if "Unemployment_Rate_percent" in peer_df.columns:
        risk_cols.append("_risk_unemp")
        risk_labels.append(_add_pct("Unemployment_Rate_percent", "_risk_unemp", "Unemployment"))
    if "Population_Below_Poverty_Line_percent" in peer_df.columns:
        risk_cols.append("_risk_poverty")
        risk_labels.append(_add_pct("Population_Below_Poverty_Line_percent", "_risk_poverty", "Poverty rate"))

    if opp_cols:
        out["opportunity_score"] = out[opp_cols].mean(axis=1, skipna=True) * 100.0
    else:
        out["opportunity_score"] = pd.Series([np.nan] * len(out), index=out.index)

    if risk_cols:
        out["risk_score"] = out[risk_cols].mean(axis=1, skipna=True) * 100.0
    else:
        out["risk_score"] = pd.Series([np.nan] * len(out), index=out.index)

    return out, opp_labels, risk_labels


def make_country_kpis(country, peer_df):
    metrics = [m for m in COUNTRY_PROFILE_METRICS if m in peer_df.columns]
    if not metrics:
        metrics = [m for m in numeric_cols if m in peer_df.columns]
    metrics = metrics[:4]

    if not metrics:
        return []

    row = df[df[COUNTRY_COL] == country]
    cards = []
    for m in metrics:
        val = np.nan
        if not row.empty:
            val = pd.to_numeric(row[m].iloc[0], errors="coerce")
        pct = peer_percentile(peer_df, country, m)
        pct_txt = "-" if pd.isna(pct) else f"{pct:.0f}"
        cards.append(
            html.Div(
                className="country-kpi",
                children=[
                    html.Div(label_for(m), className="kpi-label"),
                    html.Div(format_value(val), className="kpi-value"),
                    html.Div(f"Peer percentile: {pct_txt}", className="kpi-sub"),
                ],
            )
        )
    return cards


def make_country_peer_bars(country, peer_df, metrics):
    metrics = [m for m in metrics if m in peer_df.columns]
    if not metrics:
        fig = go.Figure().update_layout(template="infra_light")
        return fig

    rows = []
    country_row = df[df[COUNTRY_COL] == country]
    for m in metrics:
        series = pd.to_numeric(col_as_series(peer_df, m), errors="coerce").dropna()
        if series.empty:
            continue
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        median = series.median()
        country_val = np.nan
        if not country_row.empty:
            country_val = pd.to_numeric(country_row[m].iloc[0], errors="coerce")
        rows.append((m, country_val, median, q25, q75))

    if not rows:
        fig = go.Figure().update_layout(template="infra_light")
        return fig

    metrics = [r[0] for r in rows]
    labels = [label_for(r[0]) for r in rows]
    country_vals = [r[1] for r in rows]
    medians = [r[2] for r in rows]
    q25s = [r[3] for r in rows]
    q75s = [r[4] for r in rows]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=q75s,
            mode="lines",
            line=dict(color="rgba(107,114,128,0.25)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=q25s,
            mode="lines",
            line=dict(color="rgba(107,114,128,0.25)"),
            fill="tonexty",
            fillcolor="rgba(107,114,128,0.15)",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=medians,
            mode="lines+markers",
            name="Peer median",
            line=dict(color="rgba(107,114,128,0.85)"),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=country_vals,
            name=country or "Country",
            marker=dict(color="rgba(59,130,246,0.85)"),
            customdata=metrics,
            hovertemplate="<b>%{x}</b><br>Country: %{y}<extra></extra>",
        )
    )

    fig.update_layout(template="infra_light", barmode="overlay")
    fig = card_layout(fig, y_title="Value")
    return fig


def make_country_radar(country, peer_df, metrics):
    metrics = [m for m in metrics if m in peer_df.columns]
    if not metrics:
        fig = go.Figure().update_layout(template="infra_light")
        return fig

    values = []
    for m in metrics:
        pct = peer_percentile(peer_df, country, m)
        if pd.notna(pct):
            values.append((m, pct))

    if not values:
        fig = go.Figure().update_layout(template="infra_light")
        return fig

    r = [v[1] for v in values]
    theta = [label_for(v[0]) for v in values]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=r,
            theta=theta,
            fill="toself",
            name=country or "Country",
            line=dict(color="rgba(59,130,246,0.85)"),
        )
    )
    fig.update_layout(
        template="infra_light",
        showlegend=False,
        polar=dict(
            radialaxis=dict(range=[0, 100], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=10)),
        ),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def make_country_metric_distribution(metric, country, peer_df):
    return make_distribution(metric, selected_country=country, data_df=peer_df)


def make_country_score_contrib(country, peer_df, spec):
    if not spec:
        fig = go.Figure().update_layout(template="infra_light")
        return fig

    rows = []
    for item in spec:
        metric = item["metric"]
        if metric not in peer_df.columns:
            continue
        pct = peer_percentile(peer_df, country, metric)
        if pd.isna(pct):
            continue
        if item.get("direction") == "low":
            pct = 100.0 - pct
        weight = float(item.get("weight", 0))
        contrib = (pct / 100.0) * weight
        rows.append((metric, pct, weight, contrib))

    if not rows:
        fig = go.Figure().update_layout(template="infra_light")
        return fig

    metrics = [r[0] for r in rows]
    labels = [label_for(r[0]) for r in rows]
    pcts = [r[1] for r in rows]
    weights = [r[2] for r in rows]
    contribs = [r[3] for r in rows]

    customdata = [[m, p, w] for m, p, w in zip(metrics, pcts, weights)]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=contribs,
            customdata=customdata,
            marker=dict(color="rgba(59,130,246,0.85)"),
            hovertemplate="<b>%{x}</b><br>Percentile: %{customdata[1]:.0f}<br>Weight: %{customdata[2]:.0f}<br>Contribution: %{y:.2f}<extra></extra>",
        )
    )

    score_df = compute_weighted_score(peer_df, spec)
    score_row = score_df[score_df[COUNTRY_COL] == country]
    if not score_row.empty:
        score_val = pd.to_numeric(score_row["score_0_100"].iloc[0], errors="coerce")
        if pd.notna(score_val):
            fig.add_annotation(
                x=1,
                y=1.1,
                xref="paper",
                yref="paper",
                text=f"Total score: {score_val:.1f}",
                showarrow=False,
                font=dict(color=THEME["muted_text"], size=12),
            )

    fig.update_layout(template="infra_light", showlegend=False)
    fig = card_layout(fig, y_title="Weighted contribution")
    return fig


def make_country_peer_scatter(country, peer_df):
    x_metric = "Real_GDP_per_Capita_USD" if "Real_GDP_per_Capita_USD" in peer_df.columns else (numeric_cols[0] if numeric_cols else None)
    if "infra_score_pct_mean" in peer_df.columns:
        y_metric = "infra_score_pct_mean"
    elif "electricity_access_percent" in peer_df.columns:
        y_metric = "electricity_access_percent"
    else:
        y_metric = numeric_cols[1] if len(numeric_cols) > 1 else (numeric_cols[0] if numeric_cols else None)

    if not x_metric or not y_metric:
        fig = go.Figure().update_layout(template="infra_light")
        return fig

    x = pd.to_numeric(col_as_series(peer_df, x_metric), errors="coerce")
    y = pd.to_numeric(col_as_series(peer_df, y_metric), errors="coerce")
    if y_metric == "infra_score_pct_mean":
        y = y * 100.0

    plot_df = pd.DataFrame({COUNTRY_COL: peer_df[COUNTRY_COL], "_x": x, "_y": y}).dropna(subset=["_x", "_y"])
    if plot_df.empty:
        fig = go.Figure().update_layout(template="infra_light")
        return fig

    fig = px.scatter(
        plot_df,
        x="_x",
        y="_y",
        hover_name=COUNTRY_COL,
        template="infra_light",
    )
    fig.update_traces(marker=dict(color="rgba(17,24,39,0.5)", size=7, line=dict(width=0)))

    row = plot_df[plot_df[COUNTRY_COL] == country]
    if not row.empty:
        fig.add_trace(
            go.Scatter(
                x=row["_x"],
                y=row["_y"],
                mode="markers+text",
                marker=dict(size=14, symbol="star", color="rgba(59,130,246,0.95)"),
                text=[country],
                textposition="top center",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    y_title = "Infrastructure score (percentile points)" if y_metric == "infra_score_pct_mean" else label_for(y_metric)
    fig = card_layout(fig, x_title=label_for(x_metric), y_title=y_title)
    return fig


def make_country_investment_outlook(country, peer_df):
    empty = go.Figure().update_layout(template="infra_light")

    scores_df, opp_labels, risk_labels = compute_opportunity_risk_scores(peer_df)
    plot_df = scores_df.dropna(subset=["opportunity_score", "risk_score"])
    if plot_df.empty:
        return empty, "Opportunity/risk indices are not available for this peer set."

    fig = px.scatter(
        plot_df,
        x="opportunity_score",
        y="risk_score",
        hover_name=COUNTRY_COL,
        template="infra_light",
    )
    fig.update_traces(
        marker=dict(color="rgba(17,24,39,0.5)", size=7, line=dict(width=0)),
        hovertemplate="<b>%{hovertext}</b><br>Opportunity: %{x:.1f}<br>Risk: %{y:.1f}<extra></extra>",
    )

    x_mid = plot_df["opportunity_score"].median()
    y_mid = plot_df["risk_score"].median()
    fig.add_shape(
        type="line",
        x0=x_mid, y0=plot_df["risk_score"].min(),
        x1=x_mid, y1=plot_df["risk_score"].max(),
        line=dict(width=1, dash="dot", color="rgba(107,114,128,0.8)"),
        layer="below",
    )
    fig.add_shape(
        type="line",
        x0=plot_df["opportunity_score"].min(), y0=y_mid,
        x1=plot_df["opportunity_score"].max(), y1=y_mid,
        line=dict(width=1, dash="dot", color="rgba(107,114,128,0.8)"),
        layer="below",
    )

    row = plot_df[plot_df[COUNTRY_COL] == country]
    if not row.empty:
        fig.add_trace(
            go.Scatter(
                x=row["opportunity_score"],
                y=row["risk_score"],
                mode="markers+text",
                marker=dict(size=16, symbol="star", color="rgba(59,130,246,0.95)"),
                text=[country],
                textposition="top center",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(showlegend=False)
    fig = card_layout(fig, x_title="Opportunity index (0-100)", y_title="Risk index (0-100)")

    opp_val = np.nan
    risk_val = np.nan
    if not row.empty:
        opp_val = pd.to_numeric(row["opportunity_score"].iloc[0], errors="coerce")
        risk_val = pd.to_numeric(row["risk_score"].iloc[0], errors="coerce")

    opp_pct = series_percentile(plot_df["opportunity_score"], opp_val)
    risk_pct = series_percentile(plot_df["risk_score"], risk_val)
    opp_txt = "-" if pd.isna(opp_val) else f"{opp_val:.1f}"
    risk_txt = "-" if pd.isna(risk_val) else f"{risk_val:.1f}"
    opp_pct_txt = "-" if pd.isna(opp_pct) else f"{opp_pct:.0f}"
    risk_pct_txt = "-" if pd.isna(risk_pct) else f"{risk_pct:.0f}"

    opp_components = ", ".join(opp_labels) if opp_labels else "n/a"
    risk_components = ", ".join(risk_labels) if risk_labels else "n/a"
    summary = (
        f"Opportunity index: {opp_txt} (peer pct {opp_pct_txt}) | "
        f"Risk index: {risk_txt} (peer pct {risk_pct_txt}). "
        f"Components: {opp_components} | {risk_components}"
    )

    return fig, summary


def build_country_metric_table(country, peer_df, metrics):
    base = [m for m in metrics if m in peer_df.columns]
    extras = [m for m in numeric_cols if m in peer_df.columns and m not in base]
    all_metrics = base + extras
    max_count = 20
    all_metrics = all_metrics[:max_count]

    if not all_metrics:
        return []

    country_row = df[df[COUNTRY_COL] == country]
    rows = []
    for m in all_metrics:
        series = pd.to_numeric(col_as_series(peer_df, m), errors="coerce").dropna()
        val = np.nan
        if not country_row.empty:
            val = pd.to_numeric(country_row[m].iloc[0], errors="coerce")

        pct = peer_percentile(peer_df, country, m)
        if pd.isna(pct) and pd.notna(val) and not series.empty:
            pct = (series <= val).sum() / len(series) * 100.0

        if pd.notna(val) and not series.empty:
            rank = int((series > val).sum() + 1)
        else:
            rank = None

        rows.append(
            {
                "metric": label_for(m),
                "metric_key": m,
                "value": format_value(val),
                "peer_pct": "-" if pd.isna(pct) else f"{pct:.0f}",
                "peer_rank": "-" if rank is None else str(rank),
            }
        )

    return rows


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
        dcc.Store(id="country-active-metric", data=DEFAULT_COUNTRY_METRIC),



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
                    className="btn filterbtn",
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
        # ------------------------------------------------------------
        # Row 1: big scatter + 2 small plots (same row)
        # ------------------------------------------------------------
        html.Div(
            className="panel panel-big",
            style={"gridColumn": "span 2"},
            children=[
                html.Div("Opportunity vs Risk (Global)", className="panel-title"),
                dcc.Graph(
                    id="opp-risk-scatter",
                    className="panel-content",
                    style={"height": "100%"},
                    config={"displayModeBar": False, "responsive": True},
                ),
            ],
        ),
        html.Div(
            className="panel panel-small",
            style={"gridColumn": "span 1"},
            children=[
                html.Div("Top / Bottom Countries", className="panel-title"),
                dcc.Graph(
                    id="ranked-bar",
                    className="panel-content",
                    style={"height": "100%"},
                    config={"displayModeBar": False, "responsive": True},
                ),
            ],
        ),
        html.Div(
            className="panel panel-small",
            style={"gridColumn": "span 1"},
            children=[
                html.Div("Gap leaders / laggards", className="panel-title"),
                dcc.Graph(
                    id="gap-ranked",
                    className="panel-content",
                    style={"height": "100%"},
                    config={"displayModeBar": False, "responsive": True},
                ),
            ],
        ),

        # ------------------------------------------------------------
        # Row 2: 2 panels per row
        # ------------------------------------------------------------
        html.Div(
            className="panel panel-tall",
            style={"gridColumn": "span 2"},
            children=[
                html.Div("Infrastructure vs GDP (Gap view)", className="panel-title"),
                dcc.Graph(
                    id="gap-scatter",
                    className="panel-content",
                    style={"height": "100%"},
                    config={"displayModeBar": False, "responsive": True},
                ),
            ],
        ),
        html.Div(
            className="panel panel-tall",
            style={"gridColumn": "span 2"},
            children=[
                html.Div("Composite Score Ranking (Top/Bottom)", className="panel-title"),
                dcc.Graph(
                    id="score-ranked",
                    className="panel-content",
                    style={"height": "100%"},
                    config={"displayModeBar": False, "responsive": True},
                ),
                html.Div(
                    style={"padding": "0 10px 10px 10px"},
                    children=[
                        html.Div("Top candidates (peer-filtered)", className="control-label"),
                        dash_table.DataTable(
                            id="score-table",
                            columns=[
                                {"name": "Rank", "id": "rank"},
                                {"name": "Country", "id": "country"},
                                {"name": "Score", "id": "score"},
                            ],
                            data=[],
                            page_size=8,
                            style_table={"overflowX": "auto"},
                            style_cell={
                                "fontFamily": "system-ui",
                                "fontSize": "12px",
                                "padding": "6px",
                                "backgroundColor": "transparent",
                                "color": THEME["text"],
                                "border": f"1px solid {THEME['border']}",
                            },
                            style_header={"fontWeight": "600", "backgroundColor": THEME["panel_alt"]},
                        ),
                    ],
                ),
            ],
        ),

        # ------------------------------------------------------------
        # Row 3: 2 panels per row
        # ------------------------------------------------------------
        html.Div(
            className="panel panel-tall",
            style={"gridColumn": "span 2"},
            children=[
                html.Div("Analyst Score Builder (weighted)", className="panel-title"),
                html.Div(
                    style={"padding": "0 8px 8px 8px", "display": "grid", "gap": "10px"},
                    children=[
                        html.Div(
                            "Pick up to 4 metrics, choose direction, and assign weights (0 disables a metric).",
                            className="control-label",
                        ),

                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "1fr 140px", "gap": "8px", "alignItems": "center"},
                            children=[
                                dcc.Dropdown(
                                    id="score-metric-1",
                                    className="light-dropdown",
                                    options=[{"label": label_for(c), "value": c} for c in numeric_cols],
                                    value=DEFAULT_SCORE_METRICS[0] if DEFAULT_SCORE_METRICS[0] in df.columns else None,
                                    clearable=True,
                                    placeholder="Metric 1",
                                ),
                                dcc.RadioItems(
                                    id="score-direction-1",
                                    options=[{"label": "High→good", "value": "high"}, {"label": "Low→good", "value": "low"}],
                                    value=DEFAULT_SCORE_DIRS[0],
                                    inline=True,
                                ),
                            ],
                        ),
                        dcc.Slider(id="score-weight-1", min=0, max=5, step=1, value=DEFAULT_SCORE_WEIGHTS[0],
                                   marks={0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}),

                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "1fr 140px", "gap": "8px", "alignItems": "center"},
                            children=[
                                dcc.Dropdown(
                                    id="score-metric-2",
                                    className="light-dropdown",
                                    options=[{"label": label_for(c), "value": c} for c in numeric_cols],
                                    value=DEFAULT_SCORE_METRICS[1] if DEFAULT_SCORE_METRICS[1] in df.columns else None,
                                    clearable=True,
                                    placeholder="Metric 2",
                                ),
                                dcc.RadioItems(
                                    id="score-direction-2",
                                    options=[{"label": "High→good", "value": "high"}, {"label": "Low→good", "value": "low"}],
                                    value=DEFAULT_SCORE_DIRS[1],
                                    inline=True,
                                ),
                            ],
                        ),
                        dcc.Slider(id="score-weight-2", min=0, max=5, step=1, value=DEFAULT_SCORE_WEIGHTS[1],
                                   marks={0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}),

                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "1fr 140px", "gap": "8px", "alignItems": "center"},
                            children=[
                                dcc.Dropdown(
                                    id="score-metric-3",
                                    className="light-dropdown",
                                    options=[{"label": label_for(c), "value": c} for c in numeric_cols],
                                    value=DEFAULT_SCORE_METRICS[2] if DEFAULT_SCORE_METRICS[2] in df.columns else None,
                                    clearable=True,
                                    placeholder="Metric 3",
                                ),
                                dcc.RadioItems(
                                    id="score-direction-3",
                                    options=[{"label": "High→good", "value": "high"}, {"label": "Low→good", "value": "low"}],
                                    value=DEFAULT_SCORE_DIRS[2],
                                    inline=True,
                                ),
                            ],
                        ),
                        dcc.Slider(id="score-weight-3", min=0, max=5, step=1, value=DEFAULT_SCORE_WEIGHTS[2],
                                   marks={0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}),

                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "1fr 140px", "gap": "8px", "alignItems": "center"},
                            children=[
                                dcc.Dropdown(
                                    id="score-metric-4",
                                    className="light-dropdown",
                                    options=[{"label": label_for(c), "value": c} for c in numeric_cols],
                                    value=DEFAULT_SCORE_METRICS[3] if DEFAULT_SCORE_METRICS[3] in df.columns else None,
                                    clearable=True,
                                    placeholder="Metric 4",
                                ),
                                dcc.RadioItems(
                                    id="score-direction-4",
                                    options=[{"label": "High→good", "value": "high"}, {"label": "Low→good", "value": "low"}],
                                    value=DEFAULT_SCORE_DIRS[3],
                                    inline=True,
                                ),
                            ],
                        ),
                        dcc.Slider(id="score-weight-4", min=0, max=5, step=1, value=DEFAULT_SCORE_WEIGHTS[3],
                                   marks={0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}),

                        html.Div(
                            style={"display": "flex", "gap": "10px", "alignItems": "center"},
                            children=[
                                html.Button("Reset score builder", id="score-reset", n_clicks=0, className="btn"),
                                html.Div(id="score-status", className="status-pill", children="Score: not computed yet"),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="panel panel-tall",
            style={"gridColumn": "span 2"},
            children=[
                html.Div("Composite Score Distribution", className="panel-title"),
                dcc.Graph(
                    id="score-dist",
                    className="panel-content",
                    style={"height": "100%"},
                    config={"displayModeBar": False, "responsive": True},
                ),
            ],
        ),
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
                        html.Button("Back to global overview", id="back-to-global", n_clicks=0, className="btn"),
                        html.Div(id="country-title", style={"fontWeight": "600"}),
                    ],
                ),
                html.Div(
                    className="app-grid2",
                    children=[
                        html.Div(
                            className="panel",
                            style={"gridColumn": "span 4"},
                            children=[
                                html.Div("Country KPIs", className="panel-title"),
                                html.Div(id="country-kpis", className="country-kpi-strip"),
                            ],
                        ),
                        html.Div(
                            className="panel panel-tall",
                            style={"gridColumn": "span 2"},
                            children=[
                                html.Div("Peer comparison (country vs peer median/IQR)", className="panel-title"),
                                dcc.Graph(
                                    id="country-peer-bars",
                                    className="panel-content",
                                    style={"height": "100%"},
                                    config={"displayModeBar": False, "responsive": True},
                                ),
                            ],
                        ),
                        html.Div(
                            className="panel panel-tall",
                            style={"gridColumn": "span 2"},
                            children=[
                                html.Div("Country percentile profile", className="panel-title"),
                                dcc.Graph(
                                    id="country-radar",
                                    className="panel-content",
                                    style={"height": "100%"},
                                    config={"displayModeBar": False, "responsive": True},
                                ),
                            ],
                        ),
                        html.Div(
                            className="panel panel-tall",
                            style={"gridColumn": "span 2"},
                            children=[
                                html.Div("Active metric distribution", className="panel-title"),
                                dcc.Graph(
                                    id="country-metric-dist",
                                    className="panel-content",
                                    style={"height": "100%"},
                                    config={"displayModeBar": False, "responsive": True},
                                ),
                            ],
                        ),
                        html.Div(
                            className="panel panel-tall",
                            style={"gridColumn": "span 2"},
                            children=[
                                html.Div("Peer scatter position", className="panel-title"),
                                dcc.Graph(
                                    id="country-peer-scatter",
                                    className="panel-content",
                                    style={"height": "100%"},
                                    config={"displayModeBar": False, "responsive": True},
                                ),
                            ],
                        ),
                        html.Div(
                            className="panel panel-tall",
                            style={"gridColumn": "span 4"},
                            children=[
                                html.Div("Investment outlook (opportunity vs risk)", className="panel-title"),
                                html.Div(
                                    id="country-investment-summary",
                                    className="control-label",
                                    style={"padding": "0 12px 6px 12px"},
                                ),
                                dcc.Graph(
                                    id="country-investment-outlook",
                                    className="panel-content",
                                    style={"height": "100%"},
                                    config={"displayModeBar": False, "responsive": True},
                                ),
                            ],
                        ),
                        html.Div(
                            className="panel panel-tall",
                            style={"gridColumn": "span 4"},
                            children=[
                                html.Div("Score contribution (current score spec)", className="panel-title"),
                                dcc.Graph(
                                    id="country-score-contrib",
                                    className="panel-content",
                                    style={"height": "100%"},
                                    config={"displayModeBar": False, "responsive": True},
                                ),
                            ],
                        ),
                        html.Div(
                            className="panel",
                            style={"gridColumn": "span 4"},
                            children=[
                                html.Div("Deep-dive metrics", className="panel-title"),
                                dash_table.DataTable(
                                    id="country-metric-table",
                                    columns=[
                                        {"name": "Metric", "id": "metric"},
                                        {"name": "Value", "id": "value"},
                                        {"name": "Peer percentile", "id": "peer_pct"},
                                        {"name": "Peer rank", "id": "peer_rank"},
                                        {"name": "metric_key", "id": "metric_key"},
                                    ],
                                    data=[],
                                    page_size=12,
                                    hidden_columns=["metric_key"],
                                    style_table={"overflowX": "auto"},
                                    style_cell={
                                        "fontFamily": "system-ui",
                                        "fontSize": "12px",
                                        "padding": "6px",
                                        "backgroundColor": "transparent",
                                        "color": THEME["text"],
                                        "border": f"1px solid {THEME['border']}",
                                    },
                                    style_header={"fontWeight": "600", "backgroundColor": THEME["panel_alt"]},
                                ),
                            ],
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
    Input("map-metric", "value"),
    Input("selected-countries", "data"),
    Input("income-band-filter", "value"),
    Input("population-band-filter", "value"),
)
def update_ranked_bar(metric, selected_countries, income_bands, population_bands):
    plot_df = apply_peer_filters(df, income_bands, population_bands)
    return make_ranked_bar(metric, selected_countries or [], data_df=plot_df)



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
    Input("score-ranked", "clickData"),
    Input("pcp-reset", "n_clicks"),
    Input("back-to-global", "n_clicks"),
)
def set_active_country(map_click, score_click, reset_clicks, back_clicks):
    ctx = (callback_context.triggered[0]["prop_id"] if callback_context.triggered else "")

    if "pcp-reset" in ctx or "back-to-global" in ctx:
        return None

    if "world-map" in ctx and map_click:
        return map_click["points"][0]["location"]

    if "score-ranked" in ctx and score_click:
        # horizontal bar: country is on y
        return score_click["points"][0].get("y")

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
    Output("country-active-metric", "data"),
    Input("country-metric-table", "active_cell"),
    Input("country-score-contrib", "clickData"),
    State("country-metric-table", "data"),
    prevent_initial_call=True,
)
def set_country_active_metric(active_cell, click_data, table_data):
    if active_cell and table_data:
        row_idx = active_cell.get("row")
        if row_idx is not None and 0 <= row_idx < len(table_data):
            metric_key = table_data[row_idx].get("metric_key")
            if metric_key:
                return metric_key

    if click_data and click_data.get("points"):
        cd = click_data["points"][0].get("customdata")
        if isinstance(cd, (list, tuple)):
            metric_key = cd[0] if cd else None
        else:
            metric_key = cd
        if metric_key:
            return metric_key

    return no_update


@app.callback(
    Output("country-kpis", "children"),
    Output("country-peer-bars", "figure"),
    Output("country-radar", "figure"),
    Output("country-metric-dist", "figure"),
    Output("country-peer-scatter", "figure"),
    Output("country-investment-outlook", "figure"),
    Output("country-investment-summary", "children"),
    Output("country-score-contrib", "figure"),
    Output("country-metric-table", "data"),
    Input("active-country", "data"),
    Input("country-active-metric", "data"),
    Input("income-band-filter", "value"),
    Input("population-band-filter", "value"),
    Input("score-metric-1", "value"),
    Input("score-weight-1", "value"),
    Input("score-direction-1", "value"),
    Input("score-metric-2", "value"),
    Input("score-weight-2", "value"),
    Input("score-direction-2", "value"),
    Input("score-metric-3", "value"),
    Input("score-weight-3", "value"),
    Input("score-direction-3", "value"),
    Input("score-metric-4", "value"),
    Input("score-weight-4", "value"),
    Input("score-direction-4", "value"),
)
def update_country_dashboard(country, active_metric, income_bands, population_bands,
                             m1, w1, d1, m2, w2, d2, m3, w3, d3, m4, w4, d4):
    empty = go.Figure().update_layout(template="infra_light")
    if not country:
        return [], empty, empty, empty, empty, empty, "", empty, []

    peer_df = apply_peer_filters(df, income_bands, population_bands)

    metrics = [m for m in COUNTRY_PROFILE_METRICS if m in peer_df.columns]
    if not metrics:
        metrics = [m for m in numeric_cols if m in peer_df.columns]

    kpis = make_country_kpis(country, peer_df)

    spec = build_score_spec(
        metrics=[m1, m2, m3, m4],
        weights=[w1, w2, w3, w4],
        dirs=[d1, d2, d3, d4],
    )

    if spec:
        score_df = compute_weighted_score(peer_df, spec)
        score_row = score_df[score_df[COUNTRY_COL] == country]
        if not score_row.empty:
            score_val = pd.to_numeric(score_row["score_0_100"].iloc[0], errors="coerce")
            series = pd.to_numeric(score_df["score_0_100"], errors="coerce").dropna()
            if pd.notna(score_val) and not series.empty:
                pct = (series <= score_val).sum() / len(series) * 100.0
            else:
                pct = np.nan
            pct_txt = "-" if pd.isna(pct) else f"{pct:.0f}"
            score_kpi = html.Div(
                className="country-kpi",
                children=[
                    html.Div("Composite score", className="kpi-label"),
                    html.Div("-" if pd.isna(score_val) else f"{score_val:.1f}", className="kpi-value"),
                    html.Div(f"Peer percentile: {pct_txt}", className="kpi-sub"),
                ],
            )
            kpis = [score_kpi] + (kpis[:3] if kpis else [])

    metrics_for_charts = metrics[:8] if metrics else []
    bars = make_country_peer_bars(country, peer_df, metrics_for_charts)
    radar = make_country_radar(country, peer_df, metrics_for_charts)

    metric_for_dist = active_metric if active_metric in peer_df.columns else (metrics_for_charts[0] if metrics_for_charts else None)
    dist = make_country_metric_distribution(metric_for_dist, country, peer_df)

    peer_scatter = make_country_peer_scatter(country, peer_df)
    investment_fig, investment_summary = make_country_investment_outlook(country, peer_df)
    score_fig = make_country_score_contrib(country, peer_df, spec) if spec else empty
    table_data = build_country_metric_table(country, peer_df, metrics)

    return kpis, bars, radar, dist, peer_scatter, investment_fig, investment_summary, score_fig, table_data


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


@app.callback(
    Output("score-metric-1", "value"),
    Output("score-weight-1", "value"),
    Output("score-direction-1", "value"),
    Output("score-metric-2", "value"),
    Output("score-weight-2", "value"),
    Output("score-direction-2", "value"),
    Output("score-metric-3", "value"),
    Output("score-weight-3", "value"),
    Output("score-direction-3", "value"),
    Output("score-metric-4", "value"),
    Output("score-weight-4", "value"),
    Output("score-direction-4", "value"),
    Input("score-reset", "n_clicks"),
    prevent_initial_call=True,
)
def reset_score_builder(_n):
    vals = []
    for i in range(SCORE_SLOTS):
        m = DEFAULT_SCORE_METRICS[i] if DEFAULT_SCORE_METRICS[i] in df.columns else None
        vals.extend([m, DEFAULT_SCORE_WEIGHTS[i], DEFAULT_SCORE_DIRS[i]])
    return vals


@app.callback(
    Output("score-ranked", "figure"),
    Output("score-dist", "figure"),
    Output("score-table", "data"),
    Output("score-status", "children"),
    Input("score-metric-1", "value"),
    Input("score-weight-1", "value"),
    Input("score-direction-1", "value"),
    Input("score-metric-2", "value"),
    Input("score-weight-2", "value"),
    Input("score-direction-2", "value"),
    Input("score-metric-3", "value"),
    Input("score-weight-3", "value"),
    Input("score-direction-3", "value"),
    Input("score-metric-4", "value"),
    Input("score-weight-4", "value"),
    Input("score-direction-4", "value"),
    Input("selected-countries", "data"),
    Input("world-map", "clickData"),
    Input("income-band-filter", "value"),
    Input("population-band-filter", "value"),
)
def update_score_views(m1, w1, d1, m2, w2, d2, m3, w3, d3, m4, w4, d4,
                       selected_countries, map_click, income_bands, population_bands):

    selected_country = None
    if map_click:
        selected_country = map_click["points"][0]["location"]

    plot_df = apply_peer_filters(df, income_bands, population_bands)

    spec = build_score_spec(
        metrics=[m1, m2, m3, m4],
        weights=[w1, w2, w3, w4],
        dirs=[d1, d2, d3, d4],
    )

    if not spec:
        empty = go.Figure().update_layout(template="infra_light")
        return empty, empty, [], "Score: select ≥1 metric with weight > 0"

    score_df = compute_weighted_score(plot_df, spec)
    bar = make_score_ranked_bar(score_df, selected_countries or [], selected_country)
    dist = make_score_distribution(score_df, selected_country)

    # top table
    t = score_df[[COUNTRY_COL, "score_0_100"]].copy()
    t["score_0_100"] = pd.to_numeric(t["score_0_100"], errors="coerce")
    t = t.dropna(subset=["score_0_100"]).sort_values("score_0_100", ascending=False).head(12)

    table_data = [
        {"rank": i + 1, "country": row[COUNTRY_COL], "score": f"{row['score_0_100']:.1f}"}
        for i, (_, row) in enumerate(t.iterrows())
    ]

    metric_txt = ", ".join([f"{label_for(x['metric'])}×{int(x['weight'])}" for x in spec])
    status = f"Score: {metric_txt} (peer-filtered n={len(plot_df)})"

    return bar, dist, table_data, status


if __name__ == "__main__":
    app.run(debug=True)

