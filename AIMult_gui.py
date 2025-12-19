import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import json
import os
# Helper function to create 3-year line chart
def load_company_options():
    # Load company_tickers.json and convert to dropdown options
    json_path = os.path.join(os.path.dirname(__file__), 'assets', 'company_tickers.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    # data is a dict with numeric keys
    options = []
    for entry in data.values():
        label = f"{entry['title']} ({entry['ticker']})"
        value = str(entry['cik_str'])
        options.append({'label': label, 'value': value, 'title': entry['title']})
    return options

company_options = load_company_options()
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from main_api import Filing

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper function to create 3-year line chart
def create_3year_chart(ni_3year, oi_3year, ocf_3year, capex_3year):
    """
    Creates an interactive Plotly line chart for 3-year data (all 4 metrics in one plot).
    """
    years = ["fy-2", "fy-1", "fy"]
    ni_billions = [x / 1e9 for x in ni_3year]
    oi_billions = [x / 1e9 for x in oi_3year]
    ocf_billions = [x / 1e9 for x in ocf_3year]
    capex_billions = [x / 1e9 for x in capex_3year]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=ni_billions, mode='lines+markers', name='Net Income', line=dict(color='steelblue', width=3)))
    fig.add_trace(go.Scatter(x=years, y=oi_billions, mode='lines+markers', name='Operating Income', line=dict(color='darkgreen', width=3)))
    fig.add_trace(go.Scatter(x=years, y=ocf_billions, mode='lines+markers', name='Operating CF', line=dict(color='darkorange', width=3)))
    fig.add_trace(go.Scatter(x=years, y=capex_billions, mode='lines+markers', name='CAPEX', line=dict(color='crimson', width=3)))

    fig.update_layout(
        title="3-Year Historical Data (Line Chart)",
        xaxis_title="Fiscal Year",
        yaxis_title="$ Billions",
        legend_title="Metric",
        height=500,
        template="plotly_white",
        hovermode="x unified"
    )
    return fig


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container([
    # Kopfzeile mit Logo
    dbc.Row([
        dbc.Col(
            html.Img(src='assets/michelin.png', style={'height': '150px', 'margin': '10px', 'maxWidth': '100%'}),
            width=12,
            className="text-center"
        )
    ]),

    # Überschrift
    dbc.Row([
        dbc.Col(
            html.H1("Multiple Extractor (SEC API)", className="text-center text-primary mb-4"),
            width=12
        )
    ]),

    # Input-Bereich
    dbc.Row([ 
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H5("Enter SEC Filing Data", className="card-title"),
                    
                    dbc.Row([
                            dbc.Col([
#                                html.Label("Company (Suchfeld):", className="fw-bold"),
                                # dbc.Row([
                                #     dbc.Col([
                                        html.Label("Company (Suchfeld):", className="fw-bold"),
                                        dcc.Dropdown(
                                            id='company-dropdown',
                                            options=company_options,
                                            placeholder='Firma suchen...',
                                            searchable=True,
                                            style={'marginBottom': '16px'},
                                            persistence=True
                                        ),
                                    ], width=8),
                                    # dbc.Col([
                                    #     html.Label("CIK:", className="fw-bold"),
                                    #     dbc.Input(id='cik-input', placeholder='CIK', type='text', className="mb-3", readonly=True)
                                    # ], width=4),
                            dbc.Col([
                                        html.Label("Fiscal Year (z.B. 2024):", className="fw-bold"),
                                        dbc.Input(id='year-input', placeholder='2024', type='number', value=2024, className="mb-3")
                                    ], width=4),
                                ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("FX-Rate ($/€):", className="fw-bold"),
                            dbc.Input(id='fx-rate-input', placeholder='1.0', type='number', value=1.0, step=0.01, className="mb-3")
                        ], width=6),
                        dbc.Col([
                            html.Label("Share Price (€):", className="fw-bold"),
                            dbc.Input(id='share-price-input', placeholder='250.0', type='number', value=250.0, step=0.01, className="mb-3")
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Discount Rate (%):", className="fw-bold"),
                            dbc.Input(id='discount-rate-input', placeholder='10.0', type='number', value=10.0, step=0.1, className="mb-3")
                        ], width=6),
                        dbc.Col([
                            html.Label("Growth Rate (%):", className="fw-bold"),
                            dbc.Input(id='growth-rate-input', placeholder='2.0', type='number', value=2.0, step=0.1, className="mb-3")
                        ], width=6)
                    ]),
                    
                    dbc.Row([
                        dbc.Col(
                            dbc.Button(
                                'Extract Metrics',
                                id='extract-button',
                                color='primary',
                                className='w-100 mb-3',
                                size='lg'
                            ),
                            width=12
                        )
                    ])
                ])
            ]),
            width=8,
            className="offset-md-2 mb-4"
        )
    ]),

    # Status/Loading Indicator
    dbc.Row([
        dbc.Col(
            html.Div(id='status-message', style={'marginTop': '10px', 'textAlign': 'center', 'fontWeight': 'bold'}),
            width=12
        )
    ]),

    # Ergebnisse
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardBody(
                    html.Div(id='results-output')
                )
            ]),
            width=12,
            className="mb-4"
        )
    ]),

    # Storage für Ergebnisse
    dcc.Store(id='extraction-store', data={}),

], fluid=True)


# ==========================
# CALLBACKS
# ==========================

@app.callback(
    [Output('results-output', 'children'),
     Output('status-message', 'children'),
     Output('extraction-store', 'data')],
    Input('extract-button', 'n_clicks'),
    [State('company-dropdown', 'value'),
     State('year-input', 'value'),
     State('share-price-input', 'value'),
     State('discount-rate-input', 'value'),
     State('growth-rate-input', 'value'),
     State('fx-rate-input', 'value')],
    prevent_initial_call=True
)
def extract_metrics(n_clicks, cik, year, share_price, discount_rate, growth_rate, fx_rate):
    """Extrahiert Metriken von SEC API via CIK und Jahr."""
    
    company_name = next((opt['title'] for opt in company_options if opt['value'] == cik), None)

    if not cik or year is None:
        return (
            html.Div("Bitte alle Felder ausfüllen (CIK, Fiscal Year).", style={'color': 'red'}),
            "❌ Fehlende Eingaben",
            {}
        )

    try:
        # Status: Loading
        logger.info(f"Extracting for CIK={cik}, Year={year}")
        
        # Filing-Klasse aus main_api.py verwenden
        # discount_rate/growth_rate werden als Prozent in die GUI eingegeben -> wir übergeben Dezimalwerte
        try:
            dr = float(discount_rate) / 100.0 if discount_rate is not None else 0.10
        except Exception:
            dr = 0.10
        try:
            gr = float(growth_rate) / 100.0 if growth_rate is not None else 0.02
        except Exception:
            gr = 0.02

        filing = Filing(cik=int(cik), filing_year=int(year), company_name=company_name, 
                fx_rate=float(fx_rate) if fx_rate else 1.0, 
                share_price=float(share_price), discount_rate=dr, growth_rate=gr)
        # ältere Varianten nennen die Methode anders; versuche beide (kompatibel)
        if hasattr(filing, 'fetch_data'):
            filing.fetch_data()
        elif hasattr(filing, 'load'):
            filing.load()

        # process() macht extract_raw_figures() + compute_metrics()
        result = filing.process()

        # Konvertiere zu Dict für Storage (verwende neues FilingResult-Format)
        result_dict = {
            'company_name': result.raw.company_name,
            'filing_year': result.raw.filing_year,
            'currency': result.raw.currency,
            'raw_figures': {
                'net_income': result.raw.net_income,
                'depreciation': result.raw.depreciation,
                'equity': result.raw.equity,
                'dividends': result.raw.dividends,
                'diluted_eps': result.raw.diluted_eps,
                'operating_cf': result.raw.operating_cf,
                'capex_abs': result.raw.capex_abs,
                'free_cash_flow': result.raw.free_cash_flow,
            },
            'value_metrics': {
                'ep': result.metrics.ep,
                'cep': result.metrics.cep,
                'bm': result.metrics.bm,
                'dp': result.metrics.dp,
                'eq_return': result.metrics.eq_return,
                'umsatzrendite': result.metrics.umsatzrendite,
                'kapitalumschlag': result.metrics.kapitalumschlag,
                'equity_valuation': result.metrics.equity_valuation,
                'market_cap_euro': result.metrics.market_cap_euro,
                'implied_growth': result.metrics.implied_growth,
                'fcf_to_equity': result.metrics.fcf_to_equity,
                'ocf_to_equity': result.metrics.ocf_to_equity,
                'fcf_to_debt': result.metrics.fcf_to_debt,
                'capex_to_ocf': result.metrics.capex_to_ocf,
                'ocf_cagr_3year': result.metrics.ocf_cagr_3year,
                'capex_cagr_3year': result.metrics.capex_cagr_3year,
                'ni_cagr_3year': result.metrics.ni_cagr_3year,
                'oi_cagr_3year': result.metrics.oi_cagr_3year,
            },
            # 3-year raw data for charting
            '3year_data': {
                'ni_3year': result.raw.ni_3year_tuple,
                'oi_3year': result.raw.oi_3year_tuple,
                'ocf_3year': result.raw.ocf_3year_tuple,
                'capex_3year': result.raw.capex_3year_tuple,
            }
        }

        # Formatiere Ergebnisse (zeige Raw + Computed + Neue Kennzahlen)
        metrics = result.metrics
        raw = result.raw
        results_table = html.Div([
            html.H5(f"✓ Results for {raw.company_name or company} ({raw.filing_year})", className="text-success"),
            html.Hr(),

            # Raw Figures
            html.H6("Raw Figures (reported units)", className="mt-4 mb-3 fw-bold"),
            dbc.Table([
                html.Tbody([
                    html.Tr([html.Td("Net Income:"), html.Td(f"${raw.net_income:,.2f}")]),
                    html.Tr([html.Td("Depreciation:"), html.Td(f"${raw.depreciation:,.2f}")]),
                    html.Tr([html.Td("Equity:"), html.Td(f"${raw.equity:,.2f}")]),
                    html.Tr([html.Td("Dividends:"), html.Td(f"${raw.dividends:,.2f}")]),
                    html.Tr([html.Td("Diluted EPS:"), html.Td(f"{raw.diluted_eps:.2f}")]),
                    html.Tr([html.Td("Operating CF:"), html.Td(f"${raw.operating_cf:,.2f}")]),
                    html.Tr([html.Td("CAPEX (abs):"), html.Td(f"${raw.capex_abs:,.2f}")]),
                    html.Tr([html.Td("Free Cash Flow:"), html.Td(f"${raw.free_cash_flow:,.2f}")]),
                ])
            ], bordered=True, hover=True),

            # Computed Ratios (from ValueMetrics)
            html.H6("Computed Ratios", className="mt-4 mb-3 fw-bold"),
            dbc.Table([
                html.Tbody([
                    html.Tr([html.Td("E/P (Earnings/Price):"), html.Td(f"{metrics.ep:.6f}")]),
                    html.Tr([html.Td("C/E/P (Cash Earnings/Price):"), html.Td(f"{metrics.cep:.6f}")]),
                    html.Tr([html.Td("B/M (Book/Market):"), html.Td(f"{metrics.bm:.6f}")]),
                    html.Tr([html.Td("D/P (Dividend/Price):"), html.Td(f"{metrics.dp:.6f}")]),
                ])
            ], bordered=True, hover=True),

            # Additional Metrics
            html.H6("Additional Metrics", className="mt-4 mb-3 fw-bold"),
            dbc.Table([
                html.Tbody([
                    html.Tr([html.Td("Equity Valuation (DCF shortcut):"), html.Td(f"${metrics.equity_valuation:,.2f}")]),
                    html.Tr([html.Td("Market Cap (EUR):"), html.Td(f"€{metrics.market_cap_euro:,.2f}")]),
                    html.Tr([html.Td("Implied Growth (g):"), html.Td(f"{metrics.implied_growth*100:.2f}%")]),
                    html.Tr([html.Td("FCF / Equity:"), html.Td(f"{metrics.fcf_to_equity:.4f}")]),
                    html.Tr([html.Td("OCF / Equity:"), html.Td(f"{metrics.ocf_to_equity:.4f}")]),
                    html.Tr([html.Td("FCF / Debt:"), html.Td(f"{metrics.fcf_to_debt:.4f}")]),
                    html.Tr([html.Td("CAPEX / OCF:"), html.Td(f"{metrics.capex_to_ocf:.4f}")]),
                    html.Tr([html.Td("OCF CAGR (3-Year):"), html.Td(f"{metrics.ocf_cagr_3year*100:.2f}%")]),
                    html.Tr([html.Td("CAPEX CAGR (3-Year):"), html.Td(f"{metrics.capex_cagr_3year*100:.2f}%")]),
                    html.Tr([html.Td("Net Income CAGR (3-Year):"), html.Td(f"{metrics.ni_cagr_3year*100:.2f}%")]),
                    html.Tr([html.Td("Operating Income CAGR (3-Year):"), html.Td(f"{metrics.oi_cagr_3year*100:.2f}%")]),
                    html.Tr([html.Td("Return on Equity (DuPont):"), html.Td(f"{metrics.eq_return:.4f}")]),
                    html.Tr([html.Td("Umsatzrendite:"), html.Td(f"{metrics.umsatzrendite:.4f}")]),
                    html.Tr([html.Td("Kapitalumschlag:"), html.Td(f"{metrics.kapitalumschlag:.4f}")]),
                ])
            ], bordered=True, hover=True),


            # 3-Year Chart
            html.H6("Chart: 3-Year Comparison", className="mt-4 mb-3 fw-bold"),
            dcc.Graph(
                figure=create_3year_chart(
                    raw.ni_3year_tuple,
                    raw.oi_3year_tuple,
                    raw.ocf_3year_tuple,
                    raw.capex_3year_tuple
                )
            ),
        ])
        
        return (
            results_table,
            f"✓ Extraction successful for {company_name} ({year})",
            result_dict
        )
        
    except Exception as e:
        logger.error(f"Error during extraction: {str(e)}", exc_info=True)
        return (
            html.Div(f"❌ Error: {str(e)}", style={'color': 'red'}),
            f"❌ Extraction failed: {str(e)}",
            {}
        )




if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
