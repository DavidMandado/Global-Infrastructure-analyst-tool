from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd

# -----------------------------------------------------------------------------
# App + data
# -----------------------------------------------------------------------------
app = Dash(__name__)

colors = {
    "background": "#000000",
    "panel": "#111111",
    "text": "#FFFFFF",
}

# Load your merged CIA data
df = pd.read_csv("data/CIA_DATA.csv")

# Try to guess some numeric columns for template plots
numeric_cols = df.select_dtypes(include="number").columns.tolist()

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

# 1) World map (choropleth)
#    Assumes you have a column called "Country" with country names that Plotly recognizes.
#    For color, we use the first numeric column if you don't want to hardcode something.
color_col = None
if numeric_cols:
    color_col = numeric_cols[0]

map_fig = px.choropleth(
    df,
    locations="Country",             # <-- adjust if your column is named differently
    locationmode="country names",
    color=color_col,
    color_continuous_scale="Viridis",
    title=f"World map colored by {color_col}" if color_col else "World map",
)
map_fig.update_layout(
    height=300,
    paper_bgcolor=colors["panel"],
    plot_bgcolor=colors["panel"],
    font_color=colors["text"],
)

# 2) Template scatter plot: first two numeric columns
if len(numeric_cols) >= 2:
    scatter_fig = px.scatter(
        df,
        x=numeric_cols[0],
        y=numeric_cols[1],
        hover_name="Country" if "Country" in df.columns else None,
        title=f"Scatter: {numeric_cols[0]} vs {numeric_cols[1]}",
    )
else:
    scatter_fig = px.scatter(title="Scatter (needs numeric columns)")
scatter_fig.update_layout(
    height=300,
    paper_bgcolor=colors["panel"],
    plot_bgcolor=colors["panel"],
    font_color=colors["text"],
)

# 3) Template histogram: first numeric column
if numeric_cols:
    hist_fig = px.histogram(
        df,
        x=numeric_cols[0],
        nbins=30,
        title=f"Distribution of {numeric_cols[0]}",
    )
else:
    hist_fig = px.histogram(title="Histogram (needs numeric columns)")
hist_fig.update_layout(
    height=300,
    paper_bgcolor=colors["panel"],
    plot_bgcolor=colors["panel"],
    font_color=colors["text"],
)

# 4) Another template plot (e.g., bar of top 10 countries on first numeric col)
if "Country" in df.columns and numeric_cols:
    df_top = df.nlargest(10, numeric_cols[0])
    bar_fig = px.bar(
        df_top,
        x="Country",
        y=numeric_cols[0],
        title=f"Top 10 countries by {numeric_cols[0]}",
    )
    bar_fig.update_layout(
        height=300,
        xaxis_tickangle=-45,
        paper_bgcolor=colors["panel"],
        plot_bgcolor=colors["panel"],
        font_color=colors["text"],
    )
else:
    bar_fig = px.bar(title="Bar (needs Country + numeric column)")
    bar_fig.update_layout(
        height=300,
        paper_bgcolor=colors["panel"],
        plot_bgcolor=colors["panel"],
        font_color=colors["text"],
    )

# -----------------------------------------------------------------------------
# Layout: 3 x 2 grid of blocks
# -----------------------------------------------------------------------------
app.layout = html.Div(
    style={
        "backgroundColor": colors["background"],
        "minHeight": "100vh",
        "padding": "1rem",
        "color": colors["text"],
    },
    children=[
        html.H1(
            "Global Infrastructure Investment Atlas (Demo Layout)",
            style={"textAlign": "center", "marginBottom": "1rem"},
        ),

        # Small text area that will show the selected country from the map
        html.Div(
            id="selected-country",
            style={
                "textAlign": "center",
                "marginBottom": "1rem",
                "fontSize": "1.1rem",
            },
            children="Click a country on the map to select it.",
        ),

        # Grid container
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(3, 1fr)",     # 3 columns
                "gridTemplateRows": "repeat(2, 360px)",      # 2 fixed-height rows
                "gap": "1rem",
            },
            children=[
                # Block 1: World map
                html.Div(
                    style={
                        "backgroundColor": colors["panel"],
                        "padding": "0.5rem",
                        "borderRadius": "8px",
                    },
                    children=[
                        html.H3("World Map", style={"margin": "0 0 0.5rem 0"}),
                        dcc.Graph(id="world-map", figure=map_fig),
                    ],
                ),

                # Block 2: Scatter
                html.Div(
                    style={
                        "backgroundColor": colors["panel"],
                        "padding": "0.5rem",
                        "borderRadius": "8px",
                    },
                    children=[
                        html.H3("Global Scatter", style={"margin": "0 0 0.5rem 0"}),
                        dcc.Graph(id="scatter-1", figure=scatter_fig),
                    ],
                ),

                # Block 3: Histogram
                html.Div(
                    style={
                        "backgroundColor": colors["panel"],
                        "padding": "0.5rem",
                        "borderRadius": "8px",
                    },
                    children=[
                        html.H3("Distribution", style={"margin": "0 0 0.5rem 0"}),
                        dcc.Graph(id="hist-1", figure=hist_fig),
                    ],
                ),

                # Block 4: Bar chart
                html.Div(
                    style={
                        "backgroundColor": colors["panel"],
                        "padding": "0.5rem",
                        "borderRadius": "8px",
                    },
                    children=[
                        html.H3("Top 10 (Template)", style={"margin": "0 0 0.5rem 0"}),
                        dcc.Graph(id="bar-1", figure=bar_fig),
                    ],
                ),

                # Block 5: empty placeholder for future view
                html.Div(
                    style={
                        "backgroundColor": colors["panel"],
                        "padding": "0.5rem",
                        "borderRadius": "8px",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                    },
                    children=[
                        html.Div("Placeholder for future view 1"),
                    ],
                ),

                # Block 6: empty placeholder for future view
                html.Div(
                    style={
                        "backgroundColor": colors["panel"],
                        "padding": "0.5rem",
                        "borderRadius": "8px",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                    },
                    children=[
                        html.Div("Placeholder for future view 2"),
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
