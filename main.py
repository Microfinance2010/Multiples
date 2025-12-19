from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Dict, List

import pdfplumber


# ==========================
# 1) Datenklassen
# ==========================

@dataclass
class RawFigures:
    company_name: str
    filing_year: int
    currency: str
    net_income: float
    depreciation: float
    equity: float
    dividends: float
    diluted_eps: float
    share_price: float
    fx_rate: float


@dataclass
class ValueMetrics:
    ep: float     # Earnings-to-Price
    cep: float    # Cash-Earnings-to-Price
    bm: float     # Book-to-Market
    dp: float     # Dividend Yield


@dataclass
class FilingResult:
    raw: RawFigures
    metrics: ValueMetrics


@dataclass
class Filing:
    """
    Repräsentiert ein 10-K Filing inkl. Tabellen, Rohzahlen und Kennzahlen.
    """
    path: str
    tables: Optional[List[Dict]] = None
    raw_figures: Optional[RawFigures] = None
    metrics: Optional[ValueMetrics] = None

    def load(self) -> None:
        """Lädt Tabellen aus der 10-K Datei."""
        self.tables = extract_tables_from_pdf(self.path)
        print(f"✓ Loaded {len(self.tables)} tables from PDF")

    def extract_raw_figures(
        self,
        company_name: str,
        filing_year: int,
        currency: str = "USD",
        share_price: float = 0.0,
        fx_rate: float = 1.0,
    ) -> None:
        """Extrahiert Rohzahlen aus den Tabellen."""
        if self.tables is None:
            raise RuntimeError("Tables not loaded. Call load() first.")
        self.raw_figures = extract_raw_figures_from_tables(
            tables=self.tables,
            company_name=company_name,
            filing_year=filing_year,
            currency=currency,
            share_price=share_price,
            fx_rate=fx_rate,
        )

    def compute_metrics(self) -> None:
        """Berechnet Kennzahlen aus den Rohzahlen."""
        if self.raw_figures is None:
            raise RuntimeError("Raw figures not set. Call extract_raw_figures() first.")
        self.metrics = compute_value_metrics(self.raw_figures)

    def to_result(self) -> FilingResult:
        """Gibt ein FilingResult-Objekt zurück."""
        if self.raw_figures is None or self.metrics is None:
            raise RuntimeError("Filing not fully processed yet.")
        return FilingResult(raw=self.raw_figures, metrics=self.metrics)
    
    def extract_all_years(
        self,
        company_name: str,
        currency: str = "USD",
        share_price: float = 0.0,
        fx_rate: float = 1.0,
    ) -> List[FilingResult]:
        """
        Extrahiert alle verfügbaren Jahre aus dem Filing.
        
        Returns:
            Liste von FilingResult-Objekten, eines pro Jahr.
        """
        if self.tables is None:
            raise RuntimeError("Tables not loaded. Call load() first.")
        
        # Finde alle verfügbaren Jahre
        available_years = self._detect_available_years()
        
        if not available_years:
            raise ValueError("No years detected in filing tables")
        
        print(f"✓ Detected {len(available_years)} years: {available_years}")
        
        # Extrahiere für jedes Jahr
        results = []
        for year in available_years:
            print(f"\n--- Extracting data for year {year} ---")
            self.extract_raw_figures(
                company_name=company_name,
                filing_year=year,
                currency=currency,
                share_price=share_price,
                fx_rate=fx_rate,
            )
            self.compute_metrics()
            results.append(self.to_result())
        
        return results
    
    def _detect_available_years(self) -> List[int]:
        """
        Erkennt alle verfügbaren Jahre in den Tabellen.
        
        Returns:
            Liste von Jahren, sortiert absteigend (neuestes zuerst).
        """
        years_set = set()
        
        for table_data in self.tables:
            page_text = table_data['page_text']
            lines = page_text.split('\n')
            
            # Pattern 1: "Year Ended December 31," dann Zeile mit Jahren
            for i, line in enumerate(lines):
                if 'year ended' in line.lower() and ('december 31' in line.lower() or 'june 30' in line.lower()):
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]
                        years = re.findall(r'\b(20\d{2})\b', next_line)
                        if years:
                            years_set.update(int(y) for y in years)
            
            # Pattern 2: Zeile mit mehreren Jahren nacheinander
            for line in lines:
                years = re.findall(r'\b(20\d{2})\b', line)
                if len(years) >= 2:
                    years_set.update(int(y) for y in years)
        
        return sorted(list(years_set), reverse=True)


# ==========================
# 2) Tabellen-Extraktion aus PDF
# ==========================

def extract_tables_from_pdf(path: str) -> List[Dict]:
    """
    Extrahiert nur relevante Financial Statement Tabellen aus dem PDF.
    Gezieltes Suchen nach Income Statement, Balance Sheet, Cash Flow.
    """
    all_tables = []
    
    # Welche Statements brauchen wir?
    target_statements = {
        'income': ['CONSOLIDATED STATEMENTS OF INCOME', 'CONSOLIDATED STATEMENTS OF OPERATIONS', 'CONSOLIDATED STATEMENTS OF EARNINGS'],
        'balance': ['CONSOLIDATED BALANCE SHEETS', 'CONSOLIDATED STATEMENTS OF FINANCIAL POSITION'],
        'cashflow': ['CONSOLIDATED STATEMENTS OF CASH FLOWS'],
    }
    
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            page_text_upper = page_text.upper()
            
            # Prüfe ob diese Seite ein relevantes Statement hat
            is_relevant = False
            for statement_type, markers in target_statements.items():
                if any(marker in page_text_upper for marker in markers):
                    is_relevant = True
                    print(f"✓ Found {statement_type.upper()} statement on page {page_num + 1}")
                    break
            
            if is_relevant:
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        all_tables.append({
                            'table': table,
                            'page_text': page_text,
                            'page_num': page_num + 1
                        })
    
    print(f"✓ Extracted {len(all_tables)} tables from {len(set(t['page_num'] for t in all_tables))} relevant pages")
    return all_tables


# ==========================
# 3) Helper-Funktionen
# ==========================

def parse_number(num_str: str) -> float:
    """
    Wandelt Strings wie '100,118' oder '(1,234)' in float.
    Klammern = negative Zahlen.
    """
    num_str = str(num_str).strip()
    is_negative = False

    if num_str.startswith("(") and num_str.endswith(")"):
        is_negative = True
        num_str = num_str[1:-1]

    num_str = num_str.replace(",", "").replace("$", "").strip()
    
    if not num_str or num_str == "—" or num_str == "-":
        return 0.0
    
    try:
        value = float(num_str)
        return -value if is_negative else value
    except:
        return 0.0


# ==========================
# 4) Keyword-Registry
# ==========================

METRIC_KEYWORDS: Dict[str, List[str]] = {
    "net_income": [
        "Net income attributable",
        "Net income",
        "Net earnings",
    ],
    "equity": [
        "Total stockholders' equity",
        "Total shareholders' equity",
        "Total equity",
    ],
    "dividends": [
        "Cash dividends paid",
        "Dividends paid",
        "Dividends declared",
    ],
    "depreciation": [
        "Depreciation and amortization",
        "Depreciation",
    ],
    "diluted_eps": [
        "Diluted earnings per share",
        "Diluted net income per share",
        "Diluted EPS",
        "Diluted",
    ],
}


# ==========================
# 5) Universelle Tabellen-Extraktion
# ==========================

def extract_metric_from_tables(
    tables: List[Dict], 
    filing_year: int,
    row_keywords: List[str],
    metric_name: str
) -> Optional[float]:
    """
    Extrahiert eine Kennzahl aus Tabellen basierend auf Zeilen-Keywords und Jahr.
    
    Workflow:
    1. Extrahiere Jahr-Reihenfolge aus page_text (z.B. "2024 2023 2022")
    2. Finde Zeile mit einem der row_keywords
    3. Mappe filing_year zu Spalten-Index
    4. Extrahiere Wert an [row][spalte]
    """
    year_str = str(filing_year)
    
    for table_data in tables:
        table = table_data['table']
        page_text = table_data['page_text']
        
        if not table or len(table) < 2:
            continue
        
        # Extrahiere Jahre aus Seiten-Text
        # Suche nach Patterns wie "Year Ended December 31," gefolgt von Jahren
        # oder einfach "2024 2023 2022" in einer Zeile
        years_in_order = []
        
        # Pattern 1: "Year Ended December 31," dann Zeile mit Jahren
        lines = page_text.split('\n')
        for i, line in enumerate(lines):
            if 'year ended' in line.lower() and ('december 31' in line.lower() or 'june 30' in line.lower()):
                # Nächste Zeile könnte die Jahre haben
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    years = re.findall(r'\b(20\d{2})\b', next_line)
                    if years:
                        years_in_order = [int(y) for y in years]
                        break
        
        # Pattern 2: Zeile mit mehreren Jahren nacheinander
        if not years_in_order:
            for line in lines:
                years = re.findall(r'\b(20\d{2})\b', line)
                if len(years) >= 2:  # Mindestens 2 Jahre in einer Zeile
                    years_in_order = [int(y) for y in years]
                    break
        
        if not years_in_order or filing_year not in years_in_order:
            continue
        
        # Finde Index des filing_year
        year_index = years_in_order.index(filing_year)
        
        # Suche nach Zeile mit einem der Keywords
        for row_num, row in enumerate(table):
            if not row:
                continue
            
            # Prüfe alle Zellen der Zeile auf Keywords
            row_text = " ".join([str(cell or "") for cell in row]).lower()
            
            if any(keyword.lower() in row_text for keyword in row_keywords):
                # Finde die Spalte mit dem Wert
                # Oft: Label | $ | Wert1 | $ | Wert2 | $ | Wert3
                # oder: Label | Wert1 | None | Wert2 | None | Wert3 | None
                
                # Sammle alle nicht-leeren Zahlen-Zellen
                value_cells = []
                for cell in row[1:]:  # Skip first cell (label)
                    if cell and cell != '$' and cell != '':
                        try:
                            val = parse_number(cell)
                            if val != 0:
                                value_cells.append(val)
                        except:
                            pass
                
                # Mappe year_index zu value_cells
                if year_index < len(value_cells):
                    return abs(value_cells[year_index])
    
    return None


# ==========================
# 6) Rohzahlen-Extraktion
# ==========================

def extract_raw_figures_from_tables(
    tables: List[Dict],
    company_name: str,
    filing_year: int,
    currency: str = "USD",
    share_price: float = 0.0,
    fx_rate: float = 1.0,
) -> RawFigures:
    """
    Extrahiert alle Kennzahlen aus Tabellen.
    100% tabellen-basiert - kein Text-Parsing mehr!
    """
    
    # Extrahiere alle Metriken
    net_income = extract_metric_from_tables(tables, filing_year, METRIC_KEYWORDS["net_income"], "net_income")
    depreciation = extract_metric_from_tables(tables, filing_year, METRIC_KEYWORDS["depreciation"], "depreciation")
    equity = extract_metric_from_tables(tables, filing_year, METRIC_KEYWORDS["equity"], "equity")
    dividends = extract_metric_from_tables(tables, filing_year, METRIC_KEYWORDS["dividends"], "dividends")
    diluted_eps = extract_metric_from_tables(tables, filing_year, METRIC_KEYWORDS["diluted_eps"], "diluted_eps")
    
    # Validierung
    if net_income is None:
        raise ValueError(f"net_income for year {filing_year} not found in tables")
    if depreciation is None:
        raise ValueError(f"depreciation for year {filing_year} not found in tables")
    if equity is None:
        raise ValueError(f"equity for year {filing_year} not found in tables")
    if dividends is None:
        print(f"⚠ Warning: dividends for year {filing_year} not found - using 0")
        dividends = 0.0
    if diluted_eps is None:
        raise ValueError(f"diluted_eps for year {filing_year} not found in tables")
    
    print(f"✓ Net Income: ${net_income:,.0f}M for year {filing_year}")
    print(f"✓ Depreciation: ${depreciation:,.0f}M for year {filing_year}")
    print(f"✓ Equity: ${equity:,.0f}M for year {filing_year}")
    print(f"✓ Dividends: ${dividends:,.0f}M for year {filing_year}")
    print(f"✓ Diluted EPS: ${diluted_eps:.2f} for year {filing_year}")

    return RawFigures(
        company_name=company_name,
        filing_year=filing_year,
        currency=currency,
        net_income=net_income,
        depreciation=depreciation,
        equity=equity,
        dividends=dividends,
        diluted_eps=diluted_eps,
        share_price=share_price,
        fx_rate=fx_rate,
    )



# ==========================
# 7) Kennzahlen-Berechnung
# ==========================

def compute_value_metrics(raw: RawFigures) -> ValueMetrics:
    """
    Berechnet Value-Kennzahlen auf Basis der Rohzahlen.
    WICHTIG: net_income, depreciation, equity, dividends sind in Millionen!
    """
    price_usd = raw.share_price * raw.fx_rate
    
    # Konvertiere Millionen zu absoluten Zahlen für Market Cap Berechnung
    net_income_abs = raw.net_income * 1e6
    depreciation_abs = raw.depreciation * 1e6
    equity_abs = raw.equity * 1e6
    dividends_abs = raw.dividends * 1e6
    
    # Berechne diluted shares aus net_income und diluted_eps
    diluted_shares = net_income_abs / raw.diluted_eps if raw.diluted_eps != 0 else 0
    market_cap = diluted_shares * price_usd if diluted_shares != 0 else 0

    ep = net_income_abs / market_cap if market_cap != 0 else 0
    cep = (net_income_abs + depreciation_abs) / market_cap if market_cap != 0 else 0
    bm = equity_abs / market_cap if market_cap != 0 else 0
    dp = dividends_abs / market_cap if market_cap != 0 else 0

    return ValueMetrics(
        ep=ep,
        cep=cep,
        bm=bm,
        dp=dp,
    )


# ==========================
# 8) Pipeline-Funktion
# ==========================

def process_filing(
    path: str,
    company_name: str,
    filing_year: int,
    currency: str,
    share_price: float,
    fx_rate: float,
) -> FilingResult:
    """
    End-to-End: PDF → FilingResult (Rohzahlen + Kennzahlen).
    """
    filing = Filing(path=path)
    filing.load()
    filing.extract_raw_figures(
        company_name=company_name,
        filing_year=filing_year,
        currency=currency,
        share_price=share_price,
        fx_rate=fx_rate,
    )
    filing.compute_metrics()
    return filing.to_result()


# ==========================
# 9) Test
# ==========================

if __name__ == "__main__":
    # Test 1: Single year extraction
    print("="*60)
    print("TEST 1: Single Year Extraction (2024)")
    print("="*60)
    
    filing = Filing(path="10k_Meta.pdf")
    filing.load()
    filing.extract_raw_figures(
        company_name="Meta",
        filing_year=2024,
        currency="USD",
        share_price=250.0,
        fx_rate=1.2
    )
    filing.compute_metrics()

    result = filing.to_result()

    print("\n" + "="*60)
    print("EXTRACTION RESULTS")
    print("="*60)
    print(f"Company: {result.raw.company_name}")
    print(f"Year: {result.raw.filing_year}")
    print(f"\n--- Raw Figures (in millions) ---")
    print(f"Net Income: ${result.raw.net_income:,.0f}M")
    print(f"Equity: ${result.raw.equity:,.0f}M")
    print(f"Dividends: ${result.raw.dividends:,.0f}M")
    print(f"Depreciation: ${result.raw.depreciation:,.0f}M")
    print(f"Diluted EPS: ${result.raw.diluted_eps:.2f} per share")
    print(f"Share Price: ${result.raw.share_price:.2f}")
    print(f"\n--- Computed Metrics ---")
    print(f"E/P Ratio: {result.metrics.ep:.4f}")
    print(f"C/E/P Ratio: {result.metrics.cep:.4f}")
    print(f"B/M Ratio: {result.metrics.bm:.4f}")
    print(f"D/P Ratio: {result.metrics.dp:.4f}")
    print("="*60)
    
    # Test 2: Multi-year extraction
    print("\n\n" + "="*60)
    print("TEST 2: Multi-Year Extraction (All Years)")
    print("="*60)
    
    filing2 = Filing(path="10k_Meta.pdf")
    filing2.load()
    all_results = filing2.extract_all_years(
        company_name="Meta",
        currency="USD",
        share_price=250.0,
        fx_rate=1.2
    )
    
    print("\n" + "="*60)
    print("MULTI-YEAR COMPARISON")
    print("="*60)
    
    for result in all_results:
        print(f"\n--- {result.raw.filing_year} ---")
        print(f"Net Income: ${result.raw.net_income:,.0f}M")
        print(f"Equity: ${result.raw.equity:,.0f}M")
        print(f"Depreciation: ${result.raw.depreciation:,.0f}M")
        print(f"Diluted EPS: ${result.raw.diluted_eps:.2f}")
        print(f"E/P: {result.metrics.ep:.4f} | C/E/P: {result.metrics.cep:.4f} | B/M: {result.metrics.bm:.4f}")
    
    print("\n" + "="*60)




# ┌─────────────────────────────────────────────────────────────────┐
# │ USER CODE                                                       │
# ├─────────────────────────────────────────────────────────────────┤
# │ filing = Filing(path="10k_Meta.pdf")                           │
# │ filing.load()                                                   │
# │ filing.extract_raw_figures(company="Meta", filing_year=2024)   │
# │ filing.compute_metrics()                                        │
# │ result = filing.to_result()                                     │
# └─────────────────────────────────────────────────────────────────┘
#                            │
#                            ▼
# ┌─────────────────────────────────────────────────────────────────┐
# │ SCHRITT 1: filing.load()                                        │
# ├─────────────────────────────────────────────────────────────────┤
# │                                                                  │
# │  Filing.load()                                                  │
# │    └─→ extract_tables_from_pdf("10k_Meta.pdf")                 │
# │                                                                  │
# │         ┌─ Öffne PDF mit pdfplumber                            │
# │         ├─ Scanne alle Seiten nach Keywords:                   │
# │         │   • "CONSOLIDATED STATEMENTS OF INCOME"              │
# │         │   • "CONSOLIDATED BALANCE SHEETS"                    │
# │         │   • "CONSOLIDATED STATEMENTS OF CASH FLOWS"          │
# │         │                                                       │
# │         ├─ Gefunden auf Seiten:                                │
# │         │   • 72, 82, 83, 86, 88: Income Statements            │
# │         │   • 77, 87, 93: Balance Sheets                       │
# │         │   • 91, 92: Cash Flow Statements                     │
# │         │                                                       │
# │         ├─ Für jede relevante Seite:                           │
# │         │   page.extract_tables() → [[rows]]                   │
# │         │   Speichere: {                                       │
# │         │     'table': [[...]],                                │
# │         │     'page_text': "...",  # Enthält "2024 2023 2022" │
# │         │     'page_num': 88                                   │
# │         │   }                                                   │
# │         │                                                       │
# │         └─ Returns: Liste mit 8 Tabellen                       │
# │                                                                  │
# │  ✓ filing.tables = [8 Tabellen mit Kontext]                    │
# │                                                                  │
# └─────────────────────────────────────────────────────────────────┘
#                            │
#                            ▼
# ┌─────────────────────────────────────────────────────────────────┐
# │ SCHRITT 2: filing.extract_raw_figures(...)                     │
# ├─────────────────────────────────────────────────────────────────┤
# │                                                                  │
# │  Filing.extract_raw_figures()                                  │
# │    └─→ extract_raw_figures_from_tables(tables, "Meta", 2024)  │
# │                                                                  │
# │         ┌──────────────────────────────────────────────┐       │
# │         │ FÜR JEDE METRIK (5x derselbe Ablauf):       │       │
# │         └──────────────────────────────────────────────┘       │
# │                                                                  │
# │         ┌─ METRIK 1: Net Income                               │
# │         │                                                       │
# │         │  extract_metric_from_tables(                         │
# │         │    tables,                                           │
# │         │    filing_year=2024,                                 │
# │         │    row_keywords=["Net income", "Net earnings"],      │
# │         │    metric_name="net_income"                          │
# │         │  )                                                    │
# │         │                                                       │
# │         │  Für jede Tabelle:                                   │
# │         │    ┌─ 1. Extrahiere Jahre aus page_text             │
# │         │    │    "Year Ended December 31,"                   │
# │         │    │    "2024 2023 2022"                            │
# │         │    │    → years_in_order = [2024, 2023, 2022]       │
# │         │    │                                                 │
# │         │    ├─ 2. Finde Index von filing_year (2024)         │
# │         │    │    → year_index = 0                            │
# │         │    │                                                 │
# │         │    ├─ 3. Suche Zeile mit Keywords                   │
# │         │    │    Row: ['Net income', '$', '62,360',          │
# │         │    │          '$', '39,098', '$', '23,200']         │
# │         │    │                                                 │
# │         │    ├─ 4. Extrahiere Zahlen (skip $ und None)        │
# │         │    │    → value_cells = [62360, 39098, 23200]       │
# │         │    │                                                 │
# │         │    └─ 5. Return value_cells[year_index]             │
# │         │         → 62360                                      │
# │         │                                                       │
# │         │  ✓ net_income = 62,360M                             │
# │         │                                                       │
# │         ├─ METRIK 2: Depreciation                             │
# │         │  extract_metric_from_tables(                         │
# │         │    ..., ["Depreciation and amortization"], ...      │
# │         │  )                                                    │
# │         │  ✓ depreciation = 15,498M                           │
# │         │                                                       │
# │         ├─ METRIK 3: Equity                                    │
# │         │  extract_metric_from_tables(                         │
# │         │    ..., ["Total stockholders' equity"], ...         │
# │         │  )                                                    │
# │         │  ✓ equity = 182,637M                                │
# │         │                                                       │
# │         ├─ METRIK 4: Dividends                                 │
# │         │  extract_metric_from_tables(                         │
# │         │    ..., ["Cash dividends paid"], ...                │
# │         │  )                                                    │
# │         │  ✗ dividends = None → 0 (nicht gefunden)            │
# │         │                                                       │
# │         └─ METRIK 5: Diluted EPS                              │
# │            extract_metric_from_tables(                         │
# │              ..., ["Diluted earnings per share"], ...         │
# │            )                                                    │
# │            Row: ['Diluted', '$', '23.86', '$', '14.87', ...]  │
# │            ✓ diluted_eps = 23.86                              │
# │                                                                  │
# │  ✓ filing.raw_figures = RawFigures(                            │
# │      company_name="Meta",                                      │
# │      filing_year=2024,                                         │
# │      net_income=62360,      # in Millionen                    │
# │      equity=182637,         # in Millionen                    │
# │      depreciation=15498,    # in Millionen                    │
# │      dividends=0,           # in Millionen                    │
# │      diluted_eps=23.86      # pro Aktie                       │
# │  )                                                              │
# │                                                                  │
# └─────────────────────────────────────────────────────────────────┘
#                            │
#                            ▼
# ┌─────────────────────────────────────────────────────────────────┐
# │ SCHRITT 3: filing.compute_metrics()                            │
# ├─────────────────────────────────────────────────────────────────┤
# │                                                                  │
# │  Filing.compute_metrics()                                      │
# │    └─→ compute_value_metrics(filing.raw_figures)              │
# │                                                                  │
# │         ┌─ Input: RawFigures                                   │
# │         │   share_price = 250.00 USD                          │
# │         │   fx_rate = 1.2                                      │
# │         │                                                       │
# │         ├─ Konvertiere zu absoluten Zahlen                     │
# │         │   net_income_abs  = 62,360 × 1e6 = 62,360,000,000  │
# │         │   depreciation_abs = 15,498 × 1e6 = 15,498,000,000 │
# │         │   equity_abs = 182,637 × 1e6 = 182,637,000,000     │
# │         │   dividends_abs = 0 × 1e6 = 0                       │
# │         │                                                       │
# │         ├─ Berechne Shares & Market Cap                        │
# │         │   price_usd = 250 × 1.2 = 300 USD                   │
# │         │   diluted_shares = 62,360,000,000 / 23.86           │
# │         │                  = 2,613,500,000 shares             │
# │         │   market_cap = 2,613,500,000 × 300                  │
# │         │              = 784,050,000,000 USD                  │
# │         │                                                       │
# │         └─ Berechne Value Multiples                            │
# │             E/P  = 62,360,000,000 / 784,050,000,000 = 0.0795  │
# │             C/E/P = (62,360M + 15,498M) / 784,050M = 0.0993   │
# │             B/M  = 182,637,000,000 / 784,050,000,000 = 0.2329 │
# │             D/P  = 0 / 784,050,000,000 = 0.0000               │
# │                                                                  │
# │  ✓ filing.metrics = ValueMetrics(                              │
# │      ep=0.0795,    # Earnings-to-Price                        │
# │      cep=0.0993,   # Cash-Earnings-to-Price                   │
# │      bm=0.2329,    # Book-to-Market                           │
# │      dp=0.0000     # Dividend-to-Price                        │
# │  )                                                              │
# │                                                                  │
# └─────────────────────────────────────────────────────────────────┘
#                            │
#                            ▼
# ┌─────────────────────────────────────────────────────────────────┐
# │ SCHRITT 4: result = filing.to_result()                         │
# ├─────────────────────────────────────────────────────────────────┤
# │                                                                  │
# │  Filing.to_result()                                            │
# │    └─→ Returns: FilingResult(                                  │
# │          raw=filing.raw_figures,                               │
# │          metrics=filing.metrics                                │
# │        )                                                        │
# │                                                                  │
# │  ✓ Finales Ergebnis mit allen Daten                            │
# │                                                                  │
# └─────────────────────────────────────────────────────────────────┘