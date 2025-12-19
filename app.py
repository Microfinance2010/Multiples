import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from main import Filing
import traceback

# ===========================
# Initialize Dash App
# ===========================

app = dash.Dash(__name__)
app.title = "AI Multiples Analyzer"

# ===========================
# App Layout
# ===========================

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("📊 AI Multiples Analyzer", style={"color": "#1f77b4", "marginBottom": "5px"}),
        html.P("Extract financial metrics from 10-K filings", style={"color": "#666", "fontSize": "14px"})
    ], style={
        "backgroundColor": "#f8f9fa",
        "padding": "20px",
        "borderBottom": "2px solid #e0e0e0",
        "marginBottom": "20px"
    }),
    
    html.Div([
        # Left Sidebar - Input
        html.Div([
            html.H3("📁 Input", style={"color": "#333", "borderBottom": "2px solid #1f77b4", "paddingBottom": "10px"}),
            
            # PDF File Input
            html.Div([
                html.Label("PDF File Path:", style={"fontWeight": "bold"}),
                dcc.Input(
                    id="pdf-path-input",
                    type="text",
                    placeholder="e.g., 10k_Meta.pdf",
                    value="10k_Meta.pdf",
                    style={
                        "width": "100%",
                        "padding": "10px",
                        "marginBottom": "10px",
                        "border": "1px solid #ccc",
                        "borderRadius": "4px",
                        "boxSizing": "border-box"
                    }
                )
            ], style={"marginBottom": "15px"}),
            
            # Company Name
            html.Div([
                html.Label("Company Name:", style={"fontWeight": "bold"}),
                dcc.Input(
                    id="company-name-input",
                    type="text",
                    placeholder="e.g., Meta",
                    value="Meta",
                    style={
                        "width": "100%",
                        "padding": "10px",
                        "marginBottom": "10px",
                        "border": "1px solid #ccc",
                        "borderRadius": "4px",
                        "boxSizing": "border-box"
                    }
                )
            ], style={"marginBottom": "15px"}),
            
            # Years to Extract
            html.Div([
                html.Label("Years (comma-separated):", style={"fontWeight": "bold"}),
                dcc.Input(
                    id="years-input",
                    type="text",
                    placeholder="e.g., 2024,2023,2022",
                    value="2024,2023,2022",
                    style={
                        "width": "100%",
                        "padding": "10px",
                        "marginBottom": "10px",
                        "border": "1px solid #ccc",
                        "borderRadius": "4px",
                        "boxSizing": "border-box"
                    }
                )
            ], style={"marginBottom": "15px"}),
            
            # Share Price
            html.Div([
                html.Label("Share Price (USD):", style={"fontWeight": "bold"}),
                dcc.Input(
                    id="share-price-input",
                    type="number",
                    placeholder="250.0",
                    value=250.0,
                    style={
                        "width": "100%",
                        "padding": "10px",
                        "marginBottom": "10px",
                        "border": "1px solid #ccc",
                        "borderRadius": "4px",
                        "boxSizing": "border-box"
                    }
                )
            ], style={"marginBottom": "15px"}),
            
            # FX Rate
            html.Div([
                html.Label("FX Rate:", style={"fontWeight": "bold"}),
                dcc.Input(
                    id="fx-rate-input",
                    type="number",
                    placeholder="1.0",
                    value=1.0,
                    step=0.01,
                    style={
                        "width": "100%",
                        "padding": "10px",
                        "marginBottom": "10px",
                        "border": "1px solid #ccc",
                        "borderRadius": "4px",
                        "boxSizing": "border-box"
                    }
                )
            ], style={"marginBottom": "15px"}),
            
            # Submit Button
            html.Button(
                "🚀 Extract Data",
                id="submit-button",
                n_clicks=0,
                style={
                    "width": "100%",
                    "padding": "12px",
                    "backgroundColor": "#1f77b4",
                    "color": "white",
                    "border": "none",
                    "borderRadius": "4px",
                    "fontSize": "14px",
                    "fontWeight": "bold",
                    "cursor": "pointer"
                }
            ),
            
            # Status Message
            html.Div(id="status-message", style={
                "marginTop": "15px",
                "padding": "10px",
                "borderRadius": "4px",
                "display": "none"
            })
            
        ], style={
            "width": "25%",
            "display": "inline-block",
            "verticalAlign": "top",
            "padding": "15px",
            "backgroundColor": "#f8f9fa",
            "borderRight": "1px solid #e0e0e0",
            "boxSizing": "border-box",
            "height": "100vh",
            "overflowY": "auto"
        }),
        
        # Right Content Area
        html.Div([
            # Tabs for different views
            dcc.Tabs(id="tabs", value="tab-1", children=[
                # Tab 1: Raw Figures
                dcc.Tab(label="📋 Raw Figures", value="tab-1", children=[
                    html.Div(id="raw-figures-content", style={"padding": "20px"})
                ]),
                
                # Tab 2: Metrics
                dcc.Tab(label="📈 Value Metrics", value="tab-2", children=[
                    html.Div(id="metrics-content", style={"padding": "20px"})
                ]),
                
                # Tab 3: Charts
                dcc.Tab(label="📊 Charts", value="tab-3", children=[
                    html.Div(id="charts-content", style={"padding": "20px"})
                ]),
                
                # Tab 4: Comparison
                dcc.Tab(label="🔍 Year Comparison", value="tab-4", children=[
                    html.Div(id="comparison-content", style={"padding": "20px"})
                ])
            ], style={"marginTop": "0"})
            
        ], style={
            "width": "75%",
            "display": "inline-block",
            "verticalAlign": "top",
            "boxSizing": "border-box",
            "paddingBottom": "20px"
        })
    ], style={
        "display": "flex",
        "width": "100%",
        "minHeight": "100vh"
    }),
    
    # Store for cached results
    dcc.Store(id="results-store", data={}),
    
], style={
    "fontFamily": "Arial, sans-serif",
    "backgroundColor": "#fff",
    "margin": "0",
    "padding": "0"
})


# ===========================
# Callbacks
# ===========================

@app.callback(
    [Output("results-store", "data"),
     Output("status-message", "children"),
     Output("status-message", "style")],
    Input("submit-button", "n_clicks"),
    [State("pdf-path-input", "value"),
     State("company-name-input", "value"),
     State("years-input", "value"),
     State("share-price-input", "value"),
     State("fx-rate-input", "value")],
    prevent_initial_call=True
)
def extract_data(n_clicks, pdf_path, company_name, years_str, share_price, fx_rate):
    """Extract financial data from PDF"""
    try:
        # Parse years
        years = [int(y.strip()) for y in years_str.split(",")]
        
        # Load and extract
        filing = Filing(path=pdf_path)
        filing.load()
        
        results = {}
        for year in years:
            try:
                filing.extract_raw_figures(
                    company_name=company_name,
                    filing_year=year,
                    currency="USD",
                    share_price=share_price,
                    fx_rate=fx_rate
                )
                filing.compute_metrics()
                result = filing.to_result()
                
                results[year] = {
                    "raw": {
                        "net_income": result.raw.net_income,
                        "equity": result.raw.equity,
                        "depreciation": result.raw.depreciation,
                        "dividends": result.raw.dividends,
                        "diluted_eps": result.raw.diluted_eps
                    },
                    "metrics": {
                        "ep": result.metrics.ep,
                        "cep": result.metrics.cep,
                        "bm": result.metrics.bm,
                        "dp": result.metrics.dp
                    }
                }
            except Exception as e:
                results[year] = {"error": str(e)}
        
        status_msg = f"✅ Successfully extracted {len([y for y in results if 'error' not in results[y]])} years"
        status_style = {
            "marginTop": "15px",
            "padding": "10px",
            "borderRadius": "4px",
            "display": "block",
            "backgroundColor": "#d4edda",
            "color": "#155724",
            "border": "1px solid #c3e6cb"
        }
        
        return results, status_msg, status_style
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        error_style = {
            "marginTop": "15px",
            "padding": "10px",
            "borderRadius": "4px",
            "display": "block",
            "backgroundColor": "#f8d7da",
            "color": "#721c24",
            "border": "1px solid #f5c6cb"
        }
        return {}, error_msg, error_style


@app.callback(
    Output("raw-figures-content", "children"),
    Input("results-store", "data")
)
def update_raw_figures(data):
    """Display raw financial figures"""
    if not data:
        return html.P("No data extracted yet. Fill in the form and click 'Extract Data'")
    
    tables = []
    for year in sorted(data.keys(), reverse=True):
        if "error" in data[year]:
            tables.append(html.Div([
                html.H4(f"❌ Year {year}", style={"color": "#721c24"}),
                html.P(data[year]["error"], style={"color": "#721c24"})
            ], style={"marginBottom": "20px", "padding": "10px", "backgroundColor": "#f8d7da", "borderRadius": "4px"}))
            continue
        
        raw = data[year]["raw"]
        tables.append(html.Div([
            html.H4(f"📊 Year {year}", style={"color": "#1f77b4", "borderBottom": "2px solid #1f77b4", "paddingBottom": "10px"}),
            html.Table([
                html.Tbody([
                    html.Tr([html.Td("Net Income:", style={"fontWeight": "bold", "padding": "8px"}), 
                             html.Td(f"${raw['net_income']:,.0f}M", style={"padding": "8px"})]),
                    html.Tr([html.Td("Equity:", style={"fontWeight": "bold", "padding": "8px", "backgroundColor": "#f9f9f9"}), 
                             html.Td(f"${raw['equity']:,.0f}M", style={"padding": "8px", "backgroundColor": "#f9f9f9"})]),
                    html.Tr([html.Td("Depreciation:", style={"fontWeight": "bold", "padding": "8px"}), 
                             html.Td(f"${raw['depreciation']:,.0f}M", style={"padding": "8px"})]),
                    html.Tr([html.Td("Dividends:", style={"fontWeight": "bold", "padding": "8px", "backgroundColor": "#f9f9f9"}), 
                             html.Td(f"${raw['dividends']:,.0f}M", style={"padding": "8px", "backgroundColor": "#f9f9f9"})]),
                    html.Tr([html.Td("Diluted EPS:", style={"fontWeight": "bold", "padding": "8px"}), 
                             html.Td(f"${raw['diluted_eps']:.2f}", style={"padding": "8px"})]),
                ])
            ], style={"width": "100%", "borderCollapse": "collapse", "border": "1px solid #ddd"})
        ], style={"marginBottom": "20px", "padding": "15px", "backgroundColor": "#f8f9fa", "borderRadius": "4px"}))
    
    return tables


@app.callback(
    Output("metrics-content", "children"),
    Input("results-store", "data")
)
def update_metrics(data):
    """Display calculated value metrics"""
    if not data:
        return html.P("No data extracted yet")
    
    tables = []
    for year in sorted(data.keys(), reverse=True):
        if "error" in data[year]:
            continue
        
        metrics = data[year]["metrics"]
        tables.append(html.Div([
            html.H4(f"📈 Year {year}", style={"color": "#1f77b4", "borderBottom": "2px solid #1f77b4", "paddingBottom": "10px"}),
            html.Table([
                html.Tbody([
                    html.Tr([html.Td("E/P (Earnings-to-Price):", style={"fontWeight": "bold", "padding": "8px"}), 
                             html.Td(f"{metrics['ep']:.4f}", style={"padding": "8px"})]),
                    html.Tr([html.Td("C/E/P (Cash-Earnings-to-Price):", style={"fontWeight": "bold", "padding": "8px", "backgroundColor": "#f9f9f9"}), 
                             html.Td(f"{metrics['cep']:.4f}", style={"padding": "8px", "backgroundColor": "#f9f9f9"})]),
                    html.Tr([html.Td("B/M (Book-to-Market):", style={"fontWeight": "bold", "padding": "8px"}), 
                             html.Td(f"{metrics['bm']:.4f}", style={"padding": "8px"})]),
                    html.Tr([html.Td("D/P (Dividend-to-Price):", style={"fontWeight": "bold", "padding": "8px", "backgroundColor": "#f9f9f9"}), 
                             html.Td(f"{metrics['dp']:.4f}", style={"padding": "8px", "backgroundColor": "#f9f9f9"})]),
                ])
            ], style={"width": "100%", "borderCollapse": "collapse", "border": "1px solid #ddd"})
        ], style={"marginBottom": "20px", "padding": "15px", "backgroundColor": "#f8f9fa", "borderRadius": "4px"}))
    
    return tables


@app.callback(
    Output("charts-content", "children"),
    Input("results-store", "data")
)
def update_charts(data):
    """Display charts for metrics over time"""
    if not data or len(data) < 2:
        return html.P("Need at least 2 years of data to show charts")
    
    # Prepare data for charts
    years = sorted([y for y in data.keys() if "error" not in data[y]], reverse=True)
    
    ep_values = [data[y]["metrics"]["ep"] for y in years]
    cep_values = [data[y]["metrics"]["cep"] for y in years]
    bm_values = [data[y]["metrics"]["bm"] for y in years]
    
    # Create charts
    fig_ep = go.Figure(data=go.Scatter(x=years, y=ep_values, mode='lines+markers', name='E/P'))
    fig_ep.update_layout(title="E/P Ratio Over Time", hovermode='x unified', height=400)
    
    fig_cep = go.Figure(data=go.Scatter(x=years, y=cep_values, mode='lines+markers', name='C/E/P', line=dict(color='orange')))
    fig_cep.update_layout(title="C/E/P Ratio Over Time", hovermode='x unified', height=400)
    
    fig_bm = go.Figure(data=go.Scatter(x=years, y=bm_values, mode='lines+markers', name='B/M', line=dict(color='green')))
    fig_bm.update_layout(title="B/M Ratio Over Time", hovermode='x unified', height=400)
    
    return html.Div([
        html.Div([dcc.Graph(figure=fig_ep)], style={"marginBottom": "30px"}),
        html.Div([dcc.Graph(figure=fig_cep)], style={"marginBottom": "30px"}),
        html.Div([dcc.Graph(figure=fig_bm)])
    ])


@app.callback(
    Output("comparison-content", "children"),
    Input("results-store", "data")
)
def update_comparison(data):
    """Display year-over-year comparison"""
    if not data or len(data) < 2:
        return html.P("Need at least 2 years of data for comparison")
    
    years = sorted([y for y in data.keys() if "error" not in data[y]], reverse=True)
    
    # Create comparison table
    rows = []
    metrics_keys = ["net_income", "equity", "depreciation", "dividends", "diluted_eps"]
    
    for metric in metrics_keys:
        metric_name = metric.replace("_", " ").title()
        row = [html.Td(metric_name, style={"fontWeight": "bold", "padding": "8px"})]
        for year in years:
            value = data[year]["raw"][metric]
            if metric == "diluted_eps":
                row.append(html.Td(f"${value:.2f}", style={"padding": "8px"}))
            else:
                row.append(html.Td(f"${value:,.0f}M", style={"padding": "8px"}))
        rows.append(html.Tr(row))
    
    header = html.Tr([html.Th("Metric", style={"fontWeight": "bold", "padding": "8px", "backgroundColor": "#1f77b4", "color": "white"})] + 
                     [html.Th(str(y), style={"fontWeight": "bold", "padding": "8px", "backgroundColor": "#1f77b4", "color": "white"}) for y in years])
    
    return html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={"width": "100%", "borderCollapse": "collapse", "border": "1px solid #ddd"}
    )


# ===========================
# Run App
# ===========================

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
