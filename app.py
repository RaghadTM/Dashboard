#!/usr/bin/env python
# coding: utf-8

# In[7]:


import re
import numpy as np
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
from pathlib import Path

# --- Load data
FILE_PATH = r"C:\Users\Ragha\Downloads\Master Data.xlsx"
df = pd.read_excel(FILE_PATH, sheet_name="Master Data")


SHEET_NAME = "Master Data"

df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)

# --- Clean headers & values
df.columns = df.columns.map(str).str.strip().str.replace(r"\s+", " ", regex=True)
BAD_VALUES = {"#REF!", "#####", "-", "—", "–", ""}
df = df.replace(list(BAD_VALUES), np.nan)

# Ensure required columns exist
for col, fallback in {
    "Market": "(Unknown Market)",
    "Supplier Name": "(Unknown Supplier)",
    "Family": "(Unknown Family)",
    "Item Description": "(Missing Item Description)",
}.items():
    if col not in df.columns:
        df[col] = fallback

# --- Detect time-like columns (WKxx, dates like 2025-06-12 or 12/06/2025)
def is_date_like(colname: str) -> bool:
    s = str(colname).strip()
    if s.upper().startswith("WK"):
        return True
    patterns = [r"^\d{4}-\d{1,2}-\d{1,2}$", r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$"]
    return any(re.match(p, s) for p in patterns)

time_cols = [c for c in df.columns if is_date_like(c)]

# Convert numeric-like columns (time columns + any total-ish columns)
maybe_total_cols = [c for c in df.columns if re.search(r"total|اجمالي|إجمالي|qty|quantity|units|value|قيمة", c, re.I)]
num_cols = set(list(time_cols) + list(maybe_total_cols))
for c in num_cols:
    df[c] = (
        df[c].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    df[c] = pd.to_numeric(df[c], errors="coerce")

# --- Sort periods and choose default latest period
def col_sort_key(c):
    s = str(c).strip()
    if s.upper().startswith("WK"):
        m = re.search(r"\d+", s)
        return (0, int(m.group()) if m else 10**6)
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="raise")
        return (1, dt.toordinal())
    except Exception:
        return (2, s)

time_cols_sorted = sorted(time_cols, key=col_sort_key) or []


def latest_nonempty_column(cols):
    for c in reversed(cols):
        if pd.to_numeric(df[c], errors="coerce").fillna(0).sum() > 0:
            return c
    return cols[-1] if cols else None


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").fillna(0)


def apply_filters(base_df, market, supplier, family):
    f = base_df
    if market != "(All)":
        f = f[f["Market"] == market]
    if supplier != "(All)":
        f = f[f["Supplier Name"] == supplier]
    if family != "(All)":
        f = f[f["Family"] == family]
    return f.copy()


default_time_col = latest_nonempty_column(time_cols_sorted)

# --- Dash app
app = dash.Dash(__name__)
server = app.server  # for deployment (Gunicorn, Render, etc.)

# Dropdowns
drop_time = dcc.Dropdown(
    options=[{"label": i, "value": i} for i in time_cols_sorted],
    value=default_time_col,
    id="time-dropdown",
    clearable=False,
)

drop_market = dcc.Dropdown(
    options=[{"label": "(All)", "value": "(All)"}]
            + [{"label": i, "value": i} for i in sorted(df["Market"].dropna().unique())],
    value="(All)", id="market-dropdown", clearable=False
)

drop_supplier = dcc.Dropdown(
    options=[{"label": "(All)", "value": "(All)"}]
            + [{"label": i, "value": i} for i in sorted(df["Supplier Name"].dropna().unique())],
    value="(All)", id="supplier-dropdown", clearable=False
)

drop_family = dcc.Dropdown(
    options=[{"label": "(All)", "value": "(All)"}]
            + [{"label": i, "value": i} for i in sorted(df["Family"].dropna().unique())],
    value="(All)", id="family-dropdown", clearable=False
)

# Controls
topn_slider = dcc.Slider(
    id="topn-slider", min=5, max=100, step=5, value=40,
    marks={5:"5",20:"20",40:"40",60:"60",80:"80",100:"100"},
    tooltip={"placement": "bottom", "always_visible": False}
)

log_toggle = dcc.Checklist(
    id="log-toggle",
    options=[{"label": " Log scale (Y)", "value": "log"}],
    value=[]
)

# Layout
app.layout = html.Div(
    [
        html.H1("Sales Mini-Dashboard"),

        # Filters
        html.Div(
            [
                html.Div([html.Label("Period"), drop_time], style={"flex": "1 1 220px", "minWidth": 220}),
                html.Div([html.Label("Market"), drop_market], style={"flex": "1 1 220px", "minWidth": 220}),
                html.Div([html.Label("Supplier"), drop_supplier], style={"flex": "1 1 220px", "minWidth": 220}),
                html.Div([html.Label("Family"), drop_family], style={"flex": "1 1 220px", "minWidth": 220}),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "10px"},
        ),

        # Controls row
        html.Div(
            [
                html.Div([html.Label("Top N items"), topn_slider], style={"flex": "2", "minWidth": 260}),
                html.Div([html.Label("Options"), log_toggle], style={"flex": "1", "minWidth": 180, "paddingTop": 18}),
            ],
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "6px"},
        ),

        # KPIs
        dcc.Loading(html.Div(id="kpi-block"), type="default"),

        # Tabs with plots
        dcc.Tabs(
            [
                dcc.Tab(label="Top Items", children=[
                    dcc.Loading(dcc.Graph(id="sales-chart", config={"displaylogo": False}), type="default")
                ]),
                dcc.Tab(label="Trend Over Time", children=[
                    dcc.Loading(dcc.Graph(id="trend-total", config={"displaylogo": False}), type="default")
                ]),
                dcc.Tab(label="Market Mix & Suppliers", children=[
                    html.Div(
                        [
                            dcc.Loading(dcc.Graph(id="market-share", config={"displaylogo": False}), type="default"),
                            dcc.Loading(dcc.Graph(id="supplier-top", config={"displaylogo": False}), type="default"),
                        ],
                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"}
                    )
                ]),
                dcc.Tab(label="Items × Time Heatmap", children=[
                    dcc.Loading(dcc.Graph(id="heatmap-items-time", config={"displaylogo": False}), type="default")
                ]),
            ]
        ),
    ],
    style={"padding": "12px"},
)

# --- Callbacks
@app.callback(
    [
        Output("kpi-block", "children"),
        Output("sales-chart", "figure"),
        Output("trend-total", "figure"),
        Output("market-share", "figure"),
        Output("supplier-top", "figure"),
        Output("heatmap-items-time", "figure"),
    ],
    [
        Input("time-dropdown", "value"),
        Input("market-dropdown", "value"),
        Input("supplier-dropdown", "value"),
        Input("family-dropdown", "value"),
        Input("topn-slider", "value"),
        Input("log-toggle", "value"),
    ]
)

def update_dashboard(selected_time, selected_market, selected_supplier, selected_family, top_n, log_opts):
    # Guards
    if not time_cols_sorted:
        empty = px.bar(title="")
        return html.Div("No period column found."), empty, empty, empty, empty, empty

    if not selected_time or selected_time not in df.columns:
        selected_time = default_time_col

    filtered = apply_filters(df, selected_market, selected_supplier, selected_family)

    # Numeric sales for the selected period
    y = safe_numeric(filtered[selected_time])
    filtered = filtered.assign(_y=y)

    # ---------- KPIs ----------
    total_sales = float(filtered["_y"].sum())
    items_with_sales = int((filtered["_y"] > 0).sum())

    if items_with_sales > 0:
        top_row = filtered.loc[filtered["_y"].idxmax()]
        top_item = str(top_row.get("Item Description", "(Missing Item Description)"))
        top_value = float(top_row["_y"])
    else:
        top_item, top_value = "(No sales)", 0.0

    kpis = html.Div(
        [
            html.Div(
                [html.Div(f"Total Sales ({selected_time})", style={"fontSize": 12, "color": "#666"}),
                 html.Div(f"{total_sales:,.2f}", style={"fontSize": 24, "fontWeight": 700})],
                style={"padding": "14px 18px", "border": "1px solid #ddd", "borderRadius": 12, "minWidth": 220}
            ),
            html.Div(
                [html.Div("Items with Sales", style={"fontSize": 12, "color": "#666"}),
                 html.Div(f"{items_with_sales:,}", style={"fontSize": 24, "fontWeight": 700})],
                style={"padding": "14px 18px", "border": "1px solid #ddd", "borderRadius": 12, "minWidth": 220}
            ),
            html.Div(
                [html.Div("Top Item", style={"fontSize": 12, "color": "#666"}),
                 html.Div(top_item, style={"fontSize": 16, "fontWeight": 600}),
                 html.Div(f"{top_value:,.2f}", style={"fontSize": 14, "color": "#666"})],
                style={"padding": "14px 18px", "border": "1px solid #ddd", "borderRadius": 12, "minWidth": 320}
            ),
        ],
        style={"display": "flex", "gap": 16, "flexWrap": "wrap", "marginBottom": "8px"},
    )

    # ---------- Chart 1: Top N items (selected period) ----------
    top_df = (
        filtered[["_y", "Item Description"]]
        .groupby("Item Description", dropna=False, as_index=False)["_y"].sum()
        .sort_values("_y", ascending=False)
        .head(int(top_n))
    )

    fig_items = px.bar(
        top_df,
        x="Item Description",
        y="_y",
        title=f"Top {min(int(top_n), len(top_df))} Items — {selected_time}",
    )
    fig_items.update_layout(
        xaxis_title=None,
        yaxis_title=str(selected_time),
        xaxis={"tickangle": -60, "categoryorder": "total descending"},
        margin=dict(l=40, r=20, t=60, b=160),
        height=520,
    )
    if "log" in (log_opts or []):
        fig_items.update_yaxes(type="log", dtick=1)

    # ---------- Long form across all time columns ----------
    long_df = filtered.melt(
        id_vars=["Market", "Supplier Name", "Family", "Item Description"],
        value_vars=time_cols_sorted,
        var_name="Period",
        value_name="Value",
    )
    long_df["Value"] = safe_numeric(long_df["Value"])  # coerce to numbers

    # Keep order of periods stable
    sorted_periods = time_cols_sorted
    period_order = {p: i for i, p in enumerate(sorted_periods)}

    # ---------- Chart 2: Trend over time (total) ----------
    trend_total = (
        long_df.groupby("Period", as_index=False)["Value"].sum()
        .assign(order=lambda d: d["Period"].map(period_order))
        .sort_values("order")
    )

    fig_trend = px.line(
        trend_total,
        x="Period",
        y="Value",
        markers=True,
        title="Total Sales Trend (All Periods)",
    )
    fig_trend.update_layout(
        xaxis_title=None,
        yaxis_title="Total",
        xaxis={"categoryorder": "array", "categoryarray": sorted_periods, "tickangle": -45},
        height=500,
        margin=dict(l=40, r=20, t=60, b=120),
    )
    if "log" in (log_opts or []):
        fig_trend.update_yaxes(type="log", dtick=1)

    # ---------- Chart 3: Market mix (selected period) ----------
    market_mix = (
        filtered.groupby("Market", as_index=False)["_y"].sum()
        .sort_values("_y", ascending=False)
    )
    fig_market = px.bar(
        market_mix,
        x="Market",
        y="_y",
        title=f"Market Mix — {selected_time}",
    )
    fig_market.update_layout(
        xaxis_title=None, yaxis_title="Sales",
        xaxis={"tickangle": -30}, height=450, margin=dict(l=40, r=20, t=60, b=100)
    )
    if "log" in (log_opts or []):
        fig_market.update_yaxes(type="log", dtick=1)

    # ---------- Chart 4: Top suppliers (selected period) ----------
    supplier_mix = (
        filtered.groupby("Supplier Name", as_index=False)["_y"].sum()
        .sort_values("_y", ascending=False)
        .head(20)
    )
    fig_supplier = px.bar(
        supplier_mix,
        x="Supplier Name",
        y="_y",
        title=f"Top 20 Suppliers — {selected_time}",
    )
    fig_supplier.update_layout(
        xaxis_title=None, yaxis_title="Sales",
        xaxis={"tickangle": -45}, height=450, margin=dict(l=40, r=20, t=60, b=140)
    )
    if "log" in (log_opts or []):
        fig_supplier.update_yaxes(type="log", dtick=1)

    # ---------- Chart 5: Heatmap Items × Time ----------
    sums_all_time = (
        long_df.groupby("Item Description", as_index=False)["Value"].sum()
        .sort_values("Value", ascending=False)
        .head(int(top_n))
    )
    top_items_list = sums_all_time["Item Description"].tolist()
    hm_df = long_df[long_df["Item Description"].isin(top_items_list)]

    hm_grouped = (
        hm_df.groupby(["Item Description", "Period"], as_index=False)["Value"].sum()
        .assign(order=lambda d: d["Period"].map(period_order))
        .sort_values(["Item Description", "order"])
    )

    hm_pivot = (
        hm_grouped.pivot(index="Item Description", columns="Period", values="Value")
        .reindex(columns=sorted_periods)
        .fillna(0.0)
    )

    hm_pivot = hm_pivot.head(min(30, len(hm_pivot)))  # readability

    fig_heatmap = px.imshow(
        hm_pivot.values,
        x=hm_pivot.columns.tolist(),
        y=hm_pivot.index.tolist(),
        aspect="auto",
        title=f"Heatmap — Top Items × Time (Top {min(int(top_n), len(hm_pivot))})",
        labels=dict(x="Period", y="Item", color="Sales"),
    )
    fig_heatmap.update_layout(
        height=650,
        xaxis={"tickangle": -45},
        margin=dict(l=120, r=20, t=60, b=120)
    )

    return kpis, fig_items, fig_trend, fig_market, fig_supplier, fig_heatmap


# --- Run (local)
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8051)


# In[ ]:




