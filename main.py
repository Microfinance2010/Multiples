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
    diluted_shares: float
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
    raw_figures: Optional[RawFigures] = None
    metrics: Optional[ValueMetrics] = None

    def load(self) -> None:
        """Lädt den Text aus der 10-K Datei."""
        self.text = load_10k_text(self.path)

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
    Für reine TXT-Files kann man das leicht anpassen.
    """
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

    # Basic Cleanup: Mehrfach-Whitespace entfernen
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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


def extract_metric(text: str, patterns: List[str], metric_name: str) -> float:
    """
    Versucht nacheinander mehrere Regex-Patterns für eine Kennzahl.
    Gibt beim ersten Treffer den Wert zurück, sonst Fehler.
    """
    for pat in patterns:
        match = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return parse_number(match.group(1))

    raise ValueError(f"{metric_name} not found with given patterns")


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
    "diluted_shares": [
        r"Weighted-average diluted shares(?: outstanding)?[^\d]*([\d,(),\.]+)",
        r"Diluted shares outstanding[^\d]*([\d,(),\.]+)",
    ],
    "depreciation": [
        r"Depreciation and amortization(?: expense)?[^\$]*\$?\s*([\d,()]+)",
        r"Depreciation[^\$]*\$?\s*([\d,()]+)",
    ],
}


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
) -> RawFigures:
    """
    Extrahiert zentrale Kennzahlen aus dem Text eines 10-K-Filings.
    Nutzt konfigurierbare Regex-Patterns aus METRIC_PATTERNS.
    """

    net_income = extract_metric(text, METRIC_PATTERNS["net_income"], "net_income")
    depreciation = extract_metric(text, METRIC_PATTERNS["depreciation"], "depreciation")
    equity = extract_metric(text, METRIC_PATTERNS["equity"], "equity")
    dividends = extract_metric(text, METRIC_PATTERNS["dividends"], "dividends")
    diluted_shares = extract_metric(
        text, METRIC_PATTERNS["diluted_shares"], "diluted_shares"
    )

    return RawFigures(
        company_name=company_name,
        filing_year=filing_year,
        currency=currency,
        net_income=net_income,
        depreciation=depreciation,
        equity=equity,
        dividends=dividends,
        diluted_shares=diluted_shares,
        share_price=share_price,
        fx_rate=fx_rate,
    )


# ==========================
# 6) Calculator: RawFigures -> ValueMetrics
# ==========================

def compute_value_metrics(raw: RawFigures) -> ValueMetrics:
    """
    Berechnet Value-Kennzahlen auf Basis der Rohzahlen.
    share_price wird als EUR (o. ä.) angenommen,
    fx_rate z. B. als USD/EUR, sodass price_usd = share_price * fx_rate.
    """
    price_usd = raw.share_price * raw.fx_rate
    market_cap = raw.diluted_shares * price_usd

    ep = raw.net_income / market_cap
    cep = (raw.net_income + raw.depreciation) / market_cap
    bm = raw.equity / market_cap
    dp = raw.dividends / market_cap

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
