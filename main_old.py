from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Dict, List

import pdfplumber  # ggf. vorher installieren: pip install pdfplumber


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
    diluted_eps: float  # Diluted Earnings Per Share
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
    Repräsentiert ein 10-K Filing inkl. Text, Rohzahlen und Kennzahlen.
    Orchestriert nur – die eigentliche Logik liegt in Funktionen unten.
    """
    path: str
    text: Optional[str] = None
    tables: Optional[List[List[List[str]]]] = None
    raw_figures: Optional[RawFigures] = None
    metrics: Optional[ValueMetrics] = None

    def load(self) -> None:
        """Lädt den Text und Tabellen aus der 10-K Datei."""
        self.text = load_10k_text(self.path)
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
        """Extrahiert Rohzahlen aus dem Text und speichert sie in self.raw_figures."""
        if self.text is None:
            raise RuntimeError("Filing text not loaded. Call load() first.")
        self.raw_figures = extract_raw_figures_from_text(
            text=self.text,
            company_name=company_name,
            filing_year=filing_year,
            currency=currency,
            share_price=share_price,
            fx_rate=fx_rate,
            tables=self.tables,
        )

    def compute_metrics(self) -> None:
        """Berechnet Kennzahlen aus den Rohzahlen und speichert sie in self.metrics."""
        if self.raw_figures is None:
            raise RuntimeError("Raw figures not set. Call extract_raw_figures() first.")
        self.metrics = compute_value_metrics(self.raw_figures)

    def to_result(self) -> FilingResult:
        """Gibt ein FilingResult-Objekt zurück (z. B. für LLM/RAG)."""
        if self.raw_figures is None or self.metrics is None:
            raise RuntimeError("Filing not fully processed yet.")
        return FilingResult(raw=self.raw_figures, metrics=self.metrics)


# ==========================
# 2) Loader
# ==========================

def load_10k_text(path: str) -> str:
    """
    Liest ein PDF-basiertes 10-K Filing ein und gibt
    den extrahierten Text als String zurück.
    Extrahiert nur PART II (Financial Information), stoppt bei PART III.
    """
    text = ""
    start_reading = False
    
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            
            # Starte Extraktion ab PART II (Financial Information)
            if not start_reading and "PART II" in page_text.upper():
                start_reading = True
                print(f"✓ Found PART II on page {page_num + 1}, starting extraction")
            
            # Stoppe bei PART III (Exhibits, Signatures)
            if start_reading and "PART III" in page_text.upper():
                print(f"✓ Found PART III on page {page_num + 1}, stopping extraction")
                break
            
            if start_reading:
                text += page_text + "\n"

    # Basic Cleanup: Mehrfach-Whitespace entfernen
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_tables_from_pdf(path: str) -> List[List[List[str]]]:
    """
    Extrahiert Tabellen aus PART II des PDFs.
    Financial Statements sind in PART II - Item 8, stoppt bei PART III.
    """
    all_tables = []
    start_reading = False
    
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            
            # Starte ab PART II
            if not start_reading and "PART II" in page_text.upper():
                start_reading = True
            
            # Stoppe bei PART III
            if start_reading and "PART III" in page_text.upper():
                break
            
            if start_reading:
                tables = page.extract_tables()
                if tables:
                    all_tables.extend(tables)
                    
    return all_tables


# ==========================
# 3) Parser-Helfer
# ==========================

def parse_number(num_str: str) -> float:
    """
    Wandelt Strings wie '100,118' oder '(1,234)' in float.
    """
    num_str = num_str.strip()
    is_negative = False

    if num_str.startswith("(") and num_str.endswith(")"):
        is_negative = True
        num_str = num_str[1:-1]

    num_str = num_str.replace(",", "")
    value = float(num_str)
    return -value if is_negative else value


def extract_metric_from_tables(
    tables: List[List[List[str]]], 
    filing_year: int,
    row_keywords: List[str],
    metric_name: str
) -> Optional[float]:
    """
    Extrahiert eine Kennzahl aus Tabellen basierend auf Zeilen-Keywords und Jahr.
    Sucht nach Zeilen die eines der Keywords enthalten und extrahiert den Wert für filing_year.
    
    Args:
        tables: Liste aller Tabellen aus dem PDF
        filing_year: Das Jahr für das der Wert extrahiert werden soll
        row_keywords: Liste von Keywords die in der Zeile vorkommen können (z.B. ["Net income", "Net earnings"])
        metric_name: Name der Metrik für Debugging
    
    Returns:
        Extrahierter Wert oder None wenn nicht gefunden
    """
    year_str = str(filing_year)
    
    for table_idx, table in enumerate(tables):
        if not table or len(table) < 2:
            continue
        
        # Finde Header-Zeile mit Jahren
        header_row_idx = None
        year_column_idx = None
        
        for row_idx, row in enumerate(table[:10]):
            if not row:
                continue
            for col_idx, cell in enumerate(row):
                cell_str = str(cell or "")
                if re.search(r'\b' + year_str + r'\b', cell_str):
                    header_row_idx = row_idx
                    year_column_idx = col_idx
                    break
            if year_column_idx is not None:
                break
        
        if year_column_idx is None:
            continue
        
        # Suche nach Zeile mit einem der Keywords
        for row_num, row in enumerate(table[header_row_idx+1:], header_row_idx+1):
            if not row:
                continue
            
            # Prüfe alle Zellen der Zeile auf Keywords
            row_text = " ".join([str(cell or "") for cell in row]).lower()
            
            if any(keyword.lower() in row_text for keyword in row_keywords):
                # SPECIAL CASE: Gesamte Zeile in einer Zelle (merged cells)
                # Extrahiere alle Zahlen und mappe sie zu Jahren
                if len(row) == 1 or (len(row) > 0 and len(str(row[0])) > 100):
                    cell_str = str(row[0])
                    
                    # Finde alle Zahlen (mit oder ohne $, mit Kommas, mit Klammern für negative)
                    number_patterns = [
                        r'\$?\s*([\d,]+(?:\.\d+)?)',
                        r'\(([\d,]+(?:\.\d+)?)\)'
                    ]
                    
                    values = []
                    for pattern in number_patterns:
                        matches = re.findall(pattern, cell_str)
                        for match in matches:
                            try:
                                values.append(parse_number(match))
                            except:
                                pass
                    
                    # Hole Jahre aus Header
                    if header_row_idx is not None:
                        header_cells = table[header_row_idx]
                        years_in_header = []
                        for h_cell in header_cells:
                            if h_cell:
                                year_matches = re.findall(r'\b(20\d{2})\b', str(h_cell))
                                years_in_header.extend([int(y) for y in year_matches])
                        
                        # Mappe Jahr zu Wert
                        if filing_year in years_in_header and len(values) >= len(years_in_header):
                            year_index = years_in_header.index(filing_year)
                            if year_index < len(values):
                                return abs(values[year_index])  # abs() für Klammer-Werte
                
                # NORMALE CASE: Multi-Column Tabelle
                if year_column_idx < len(row):
                    value_str = str(row[year_column_idx] or "")
                    # Entferne $, Kommas, etc.
                    value_str = value_str.replace("$", "").replace(",", "").strip()
                    
                    if value_str:
                        try:
                            value = parse_number(row[year_column_idx])
                            return abs(value)  # abs() für negative Werte in Klammern
                        except:
                            pass
    
    return None



# ==========================
# 4) Pattern-Registry für Kennzahlen
# ==========================

METRIC_PATTERNS: Dict[str, List[str]] = {
    "net_income": [
        r"Net income attributable to [^\$]+?\$?\s*([\d,()]+)",
        r"Net income\s+\$?\s*([\d,()]+)",
    ],
    "equity": [
        r"Total stockholders[’'`]? equity[^\$]*\$?\s*([\d,()]+)",
        r"Total shareholders[’'`]? equity[^\$]*\$?\s*([\d,()]+)",
    ],
    "dividends": [
        r"Cash dividends paid[^\$]*\$?\s*([\d,()]+)",
        r"Dividends (?:declared|paid)[^\$]*\$?\s*([\d,()]+)",
    ],
    "diluted_eps": [
        # Jahr-spezifische Patterns - suchen nach dem filing_year und dem zugehörigen Wert
        # Diese werden dynamisch mit filing_year gefüllt (siehe extract_diluted_eps)
    ],
    "depreciation": [
        r"Depreciation and amortization(?: expense)?[^\$]*\$?\s*([\d,()]+)",
        r"Depreciation[^\$]*\$?\s*([\d,()]+)",
    ],
}


# ==========================
# 4a) Jahr-spezifische EPS-Extraktion
# ==========================

def extract_diluted_eps(text: str, filing_year: int) -> float:
    """
    Extrahiert Diluted EPS für ein spezifisches Jahr.
    Sucht nach verschiedenen Formaten und verwendet das filing_year zur Validierung.
    """
    # Pattern 1: "Diluted [earnings per share|net income per share|EPS] ... 2024 ... $ X.XX"
    # Findet Zeilen mit dem Jahr und extrahiert den nachfolgenden $-Wert
    patterns = [
        # Vertikale Tabelle: "2024" in eigener Zeile, dann später "Diluted ... $ 8.04"
        rf"(?:Year Ended (?:December 31|June 30),?\s+)?{filing_year}[^\d]{{0,200}}Diluted (?:earnings per share|net income per share|EPS)[^\$]{{0,50}}\$\s*([\d.]+)",
        
        # Horizontale Tabelle: "Diluted ... 2024 ... $ 8.04" oder "Diluted ... $ ... $ 8.04" mit Jahr vorher
        rf"Diluted (?:earnings per share|net income per share|EPS)(?:\s+\([\w/]+\))?[^\d]{{0,100}}{filing_year}[^\d]{{0,100}}\$\s*([\d.]+)",
        
        # Income Statement Format: Header mit Jahren, dann später Zeile mit Werten
        # "2022    2023    2024" ... später ... "Diluted earnings per share $ 4.56 $ 5.80 $ 8.04"
        rf"Year Ended (?:December 31|June 30),?\s+\d{{4}}\s+\d{{4}}\s+{filing_year}[^D]{{0,500}}Diluted (?:earnings per share|net income per share|EPS)[^\$]{{0,50}}\$\s+[\d.]+\s+\$\s+[\d.]+\s+\$\s*([\d.]+)",
        
        # Umgekehrte Reihenfolge: "Diluted EPS" zuerst, dann Jahr-Spalten
        rf"Diluted (?:earnings per share|net income per share|EPS)(?:\s+\([\w/]+\))?[^\d]{{0,50}}\$\s+[\d.]+\s+\$\s+[\d.]+\s+\$\s*([\d.]+)[^\d]{{0,100}}{filing_year}",
        
        # Sehr einfach: Diluted EPS/earnings/net income per share gefolgt von $ und Wert
        rf"Diluted (?:EPS|earnings per share|net income per share)(?:\s+\([\w/]+\))?\s+\$\s*([\d.]+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            value = float(match.group(1))
            # Sanity check: EPS sollte zwischen 0.01 und 1000 liegen
            if 0.01 <= value <= 1000:
                return value
    
    raise ValueError(f"diluted_eps for year {filing_year} not found")


def extract_diluted_eps_from_tables(tables: List[List[List[str]]], filing_year: int) -> Optional[float]:
    """
    Versucht Diluted EPS aus strukturierten Tabellen zu extrahieren.
    Sucht nach Zeilen die 'Diluted' enthalten und Spalten mit dem filing_year.
    
    Returns: EPS-Wert oder None wenn nicht gefunden
    """
    year_str = str(filing_year)
    
    # Zuerst: Finde alle Tabellen mit "earnings per share" oder "diluted"
    eps_tables = []
    for table_idx, table in enumerate(tables):
        if not table:
            continue
        # Nur erste 10 Zeilen prüfen für Performance
        table_text = " ".join([" ".join([str(cell or "") for cell in row]) for row in table[:10]]).lower()
        if "earnings per share" in table_text or "diluted" in table_text:
            eps_tables.append((table_idx, table))
    
    tables_with_year = 0
    tables_with_diluted = 0
    
    # Durchsuche nur die EPS-relevanten Tabellen
    for table_idx, table in eps_tables:
        if not table or len(table) < 2:
            continue
            
        # Suche nach Header-Zeile mit Jahren - flexibler: "2024" oder "June 30, 2024" etc.
        header_row = None
        year_column_idx = None
        
        for row_idx, row in enumerate(table[:10]):  # Prüfe erste 10 Zeilen für Header (manche Tabellen haben mehrere Header-Zeilen)
            if not row:
                continue
            # Suche nach Jahr in dieser Zeile - auch in Teil-Strings wie "June 30, 2024"
            for col_idx, cell in enumerate(row):
                cell_str = str(cell or "")
                # Suche nach 4-stelligem Jahr
                if re.search(r'\b' + year_str + r'\b', cell_str):
                    header_row = row_idx
                    year_column_idx = col_idx
                    tables_with_year += 1
                    break
            if year_column_idx is not None:
                break
        
        if year_column_idx is None:
            continue
            
        # Jetzt suche nach "Diluted" Zeile
        for row_num, row in enumerate(table[header_row+1:], header_row+1):  # Starte nach Header
            if not row:
                continue
            
            # Check all cells for "diluted" since tables may be merged
            for cell_idx, cell in enumerate(row):
                cell_str = str(cell or "")
                cell_lower = cell_str.lower()
                # Erweiterte Suche: "diluted" kann auch alleine stehen (z.B. Microsoft: "Diluted (A/C)")
                # ODER in Kombination mit "earnings per share", "eps", "net income per share"
                if "diluted" in cell_lower:
                    tables_with_diluted += 1
                    
                    # SPECIAL CASE: Wenn die gesamte Zeile in EINER Zelle steht (pdfplumber parsing issue)
                    # z.B. "Diluted EPS(2) $ 5.80 $ 8.04 $ 2.24 39 %"
                    # Extrahiere alle Zahlen und nimm die richtige basierend auf Jahr-Position
                    cell_str = str(cell)
                    
                    # Finde alle $ X.XX Werte
                    dollar_values = re.findall(r'\$\s*([\d.]+)', cell_str)
                    
                    if dollar_values:
                        # Versuche herauszufinden, welcher Wert zu welchem Jahr gehört
                        # Strategie: Suche nach Jahr-Pattern in der Zelle oder im Header
                        # Wenn Header "2023", "2024" hat, nehme entsprechenden Index
                        
                        # Checke ob der Header Jahre in separaten Spalten hat
                        if header_row is not None and year_column_idx < len(table[header_row]):
                            header_cells = table[header_row]
                            # Finde alle Jahre im Header
                            years_in_header = []
                            for h_cell in header_cells:
                                if h_cell:
                                    year_matches = re.findall(r'\b(20\d{2})\b', str(h_cell))
                                    years_in_header.extend(year_matches)
                            
                            # Wenn wir Jahre im Header haben, mappe sie zu den Werten
                            if filing_year in [int(y) for y in years_in_header]:
                                year_index = [int(y) for y in years_in_header].index(filing_year)
                                if year_index < len(dollar_values):
                                    value = float(dollar_values[year_index])
                                    if 0.01 <= value <= 1000:
                                        return value
                        
                        # Fallback: Suche nach Jahr direkt in der Zelle mit dem Diluted Text
                        # Pattern: finde Position von filing_year, dann nimm nächsten $-Wert
                        if year_str in cell_str:
                            # Teile am Jahr und suche nach $-Wert danach
                            parts = cell_str.split(year_str)
                            if len(parts) > 1:
                                after_year = parts[1]
                                match = re.search(r'\$\s*([\d.]+)', after_year)
                                if match:
                                    value = float(match.group(1))
                                    if 0.01 <= value <= 1000:
                                        return value
                        
                        # Letzter Fallback: nimm letzten Wert wenn es nur eine EPS-Zeile ist
                        # (oft ist die neueste Spalte rechts)
                        value = float(dollar_values[-1])
                        if 0.01 <= value <= 1000:
                            return value
                    
                    # NORMALE CASE: Multi-Column Table
                    # Hole Wert aus der Jahr-Spalte
                    if year_column_idx < len(row):
                        value_str = str(row[year_column_idx] or "")
                        match = re.search(r'\$?\s*([\d.]+)', value_str.replace(",", ""))
                        if match:
                            value = float(match.group(1))
                            if 0.01 <= value <= 1000:
                                return value
    
    return None


# ==========================
# 5) Parser: Text -> RawFigures
# ==========================

def extract_raw_figures_from_text(
    text: str,
    company_name: str,
    filing_year: int,
    currency: str = "USD",
    share_price: float = 0.0,
    fx_rate: float = 1.0,
    tables: Optional[List[List[List[str]]]] = None,
) -> RawFigures:
    """
    Extrahiert zentrale Kennzahlen aus Tabellen eines 10-K-Filings.
    Nutzt nur noch Tabellen-Extraktion für maximale Robustheit und Jahr-Genauigkeit.
    """
    
    if not tables:
        raise RuntimeError("No tables provided. Tables are required for extraction.")
    
    # Alle Metriken aus Tabellen extrahieren
    net_income = extract_metric_from_tables(tables, filing_year, METRIC_KEYWORDS["net_income"], "net_income")
    depreciation = extract_metric_from_tables(tables, filing_year, METRIC_KEYWORDS["depreciation"], "depreciation")
    equity = extract_metric_from_tables(tables, filing_year, METRIC_KEYWORDS["equity"], "equity")
    dividends = extract_metric_from_tables(tables, filing_year, METRIC_KEYWORDS["dividends"], "dividends")
    diluted_eps = extract_diluted_eps_from_tables(tables, filing_year)
    
    # Validierung
    if net_income is None:
        raise ValueError(f"net_income for year {filing_year} not found in tables")
    if depreciation is None:
        raise ValueError(f"depreciation for year {filing_year} not found in tables")
    if equity is None:
        raise ValueError(f"equity for year {filing_year} not found in tables")
    if dividends is None:
        raise ValueError(f"dividends for year {filing_year} not found in tables")
    if diluted_eps is None:
        raise ValueError(f"diluted_eps for year {filing_year} not found in tables")
    
    print(f"✓ Net Income: ${net_income/1e6:.2f}M for year {filing_year}")
    print(f"✓ Depreciation: ${depreciation/1e6:.2f}M for year {filing_year}")
    print(f"✓ Equity: ${equity/1e6:.2f}M for year {filing_year}")
    print(f"✓ Dividends: ${dividends/1e6:.2f}M for year {filing_year}")
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
# 6) Calculator: RawFigures -> ValueMetrics
# ==========================

def compute_value_metrics(raw: RawFigures) -> ValueMetrics:
    """
    Berechnet Value-Kennzahlen auf Basis der Rohzahlen.
    Verwendet diluted_eps und share_price zur Berechnung der Multiples.
    share_price wird als EUR (o. ä.) angenommen,
    fx_rate z. B. als USD/EUR, sodass price_usd = share_price * fx_rate.
    """
    price_usd = raw.share_price * raw.fx_rate
    
    # Berechne diluted shares aus net_income und diluted_eps
    diluted_shares = raw.net_income / raw.diluted_eps if raw.diluted_eps != 0 else 0
    market_cap = diluted_shares * price_usd if diluted_shares != 0 else 0

    ep = raw.net_income / market_cap if market_cap != 0 else 0
    cep = (raw.net_income + raw.depreciation) / market_cap if market_cap != 0 else 0
    bm = raw.equity / market_cap if market_cap != 0 else 0
    dp = raw.dividends / market_cap if market_cap != 0 else 0

    return ValueMetrics(
        ep=ep,
        cep=cep,
        bm=bm,
        dp=dp,
    )



# ==========================
# 7) Pipeline-Funktion (Convenience)
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
    End-to-End: Datei → FilingResult (Rohzahlen + Kennzahlen).
    Praktisch für erste Tests oder Batch-Verarbeitung.
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

# Optional: Ergebnis abrufen
result = filing.to_result()

# Print the results
print("\n" + "="*60)
print("EXTRACTION RESULTS")
print("="*60)
print(f"Company: {result.raw.company_name}")
print(f"Year: {result.raw.filing_year}")
print(f"\n--- Raw Figures (in millions where applicable) ---")
print(f"Net Income: ${result.raw.net_income/1e6:,.2f}M")
print(f"Equity: ${result.raw.equity/1e6:,.2f}M")
print(f"Dividends: ${result.raw.dividends/1e6:,.2f}M")
print(f"Depreciation: ${result.raw.depreciation/1e6:,.2f}M")
print(f"Diluted EPS: ${result.raw.diluted_eps:.2f} per share")
print(f"Share Price: ${result.raw.share_price:.2f}")
print(f"\n--- Computed Metrics ---")
print(f"E/P Ratio: {result.metrics.ep:.4f}")
print(f"C/E/P Ratio: {result.metrics.cep:.4f}")
print(f"B/M Ratio: {result.metrics.bm:.4f}")
print(f"D/P Ratio: {result.metrics.dp:.4f}")
print("="*60)