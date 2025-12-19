

import time
import requests
from dataclasses import dataclass
from typing import Optional

SEC_HEADERS = {
    # bitte anpassen: Name + Email/URL
    "User-Agent": "Your Name your.email@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

def cik10(cik: int | str) -> str:
    return str(cik).lstrip("0").zfill(10)

def sec_get_json(url: str) -> dict:
    # einfache Fair-Access Bremse (konservativ)
    time.sleep(0.15)
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

def pick_fact(companyfacts: dict, tag: str, unit_hint: str | None, fy: int) -> float | None:
    facts = companyfacts.get("facts", {}).get("us-gaap", {}).get(tag, {})
    units = facts.get("units", {})
    if not units:
        return None

    # wenn unit_hint nicht passt/None ist, nimm "irgendeine" unit (typisch USD oder USD/shares)
    unit_keys = [unit_hint] if unit_hint in units else list(units.keys())

    candidates = []
    for unit in unit_keys:
        for item in units.get(unit, []):
            if item.get("fy") != fy:
                continue
            if item.get("fp") != "FY":
                continue
            if item.get("form") not in ("10-K", "10-K/A"):
                continue
            if "val" in item and item["val"] is not None:
                candidates.append(item)

    if not candidates:
        return None

    # nimm den zuletzt eingereichten
    candidates.sort(key=lambda x: x.get("filed", ""), reverse=True)
    return float(candidates[0]["val"])

def get_core_metrics_from_companyfacts(cik: int | str, fy: int) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10(cik)}.json"
    cf = sec_get_json(url)
    # Tag-Fallbacks (besonders für Dividenden/Equity/Depreciation sinnvoll)
    tag_map = {
        "net_income": [("NetIncomeLoss", "USD")],
        "depreciation": [
            ("DepreciationDepletionAndAmortization", "USD"),
            ("DepreciationAndAmortization", "USD"),
        ],
        "equity": [
            ("StockholdersEquity", "USD"),
            ("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "USD"),
        ],
        "dividends": [
            ("PaymentsOfDividends", "USD"),
            ("PaymentsOfDividendsCommonStock", "USD"),
            ("DividendsCommonStockCash", "USD"),
        ],
        "diluted_eps": [("EarningsPerShareDiluted", "USD/shares")],
        # additional items for cashflow / capex
        "operating_cf": [("NetCashProvidedByUsedInOperatingActivities", "USD")],
        "capex_candidates": [
            ("PurchasesOfPropertyPlantAndEquipment", "USD"),
            ("PaymentsToAcquirePropertyPlantAndEquipment", "USD"),
            ("AdditionsToPropertyPlantAndEquipment", "USD"),
            ("CapitalExpenditures", "USD"),
        ],
        "proceeds_candidates": [
            ("ProceedsFromSaleOfPropertyPlantAndEquipment", "USD"),
            ("ProceedsFromDispositionOfPropertyPlantAndEquipment", "USD"),
        ],
        "net_ppe": [("PropertyPlantAndEquipmentNet", "USD")],
        "gain_on_sale": [("GainOnSaleOfAsset", "USD"), ("GainLossOnDispositionOfAssets", "USD")],
    }

    out = {}

    # helper to pick first available tag in a list
    def pick_first(tag_list):
        for tag, unit in tag_list:
            v = pick_fact(cf, tag, unit, fy)
            if v is not None:
                return v, tag, unit
        return None, None, None

    # pick the basic tags
    for k in ("net_income", "depreciation", "equity", "dividends", "diluted_eps", "operating_cf", "net_ppe"):
        tag_list = tag_map.get(k, [])
        val, tag, unit = pick_first(tag_list)
        out[k] = val

    # capex: try direct cashflow tags first (may be negative if cash out)
    capex_val = None
    for tag, unit in tag_map["capex_candidates"]:
        capex_val = pick_fact(cf, tag, unit, fy)
        if capex_val is not None:
            out["capex_cashflow"] = capex_val
            out["capex_tag"] = tag
            break
    else:
        out["capex_cashflow"] = None

    # proceeds (sale of PPE)
    proceeds_val, ptag, punit = pick_first(tag_map["proceeds_candidates"])
    out["proceeds"] = proceeds_val or 0.0
    out["proceeds_tag"] = ptag

    # prev year net_ppe (for delta calculation)
    prev_net_ppe = None
    if out.get("net_ppe") is not None:
        prev_net_ppe = pick_fact(cf, tag_map["net_ppe"][0][0], tag_map["net_ppe"][0][1], fy - 1)
    out["prev_net_ppe"] = prev_net_ppe

    # Vorjahreswerte für Wachstum (wenn verfügbar)
    # Net Income prev
    net_income_prev = None
    ni_tag = tag_map["net_income"][0][0] if tag_map.get("net_income") else None
    ni_unit = tag_map["net_income"][0][1] if tag_map.get("net_income") else None
    if ni_tag:
        net_income_prev = pick_fact(cf, ni_tag, ni_unit, fy - 1)
    out["net_income_prev"] = net_income_prev

    # Operating CF prev
    oper_prev = None
    ocf_tag = tag_map.get("operating_cf", [(None, None)])[0][0]
    ocf_unit = tag_map.get("operating_cf", [(None, None)])[0][1]
    if ocf_tag:
        oper_prev = pick_fact(cf, ocf_tag, ocf_unit, fy - 1)
    out["operating_cf_prev"] = oper_prev

    # Prev capex: try same capex candidates for fy-1
    prev_capex_cf = None
    for tag, unit in tag_map["capex_candidates"]:
        prev_capex_cf = pick_fact(cf, tag, unit, fy - 1)
        if prev_capex_cf is not None:
            out["capex_cashflow_prev"] = prev_capex_cf
            out["capex_tag_prev"] = tag
            break
    else:
        out["capex_cashflow_prev"] = None

    # compute prev free cash flow if possible
    prev_capex_abs = None
    if out.get("capex_cashflow_prev") is not None:
        prev_capex_abs = -out["capex_cashflow_prev"] if out["capex_cashflow_prev"] < 0 else out["capex_cashflow_prev"]
    out["capex_abs_prev"] = prev_capex_abs

    if oper_prev is not None and prev_capex_abs is not None:
        out["free_cash_flow_prev"] = oper_prev - prev_capex_abs
    else:
        out["free_cash_flow_prev"] = None

    # 3-year data for CAGR: OCF and CAPEX for fy-2, fy-1, fy
    # Operating CF for fy-2
    ocf_fy_minus_2 = None
    if ocf_tag:
        ocf_fy_minus_2 = pick_fact(cf, ocf_tag, ocf_unit, fy - 2)
    out["operating_cf_fy_minus_2"] = ocf_fy_minus_2

    # CAPEX for fy-2
    capex_fy_minus_2 = None
    for tag, unit in tag_map["capex_candidates"]:
        capex_fy_minus_2 = pick_fact(cf, tag, unit, fy - 2)
        if capex_fy_minus_2 is not None:
            out["capex_fy_minus_2"] = capex_fy_minus_2
            break
    else:
        out["capex_fy_minus_2"] = None

    # CAPEX for fy-1
    capex_fy_minus_1 = None
    for tag, unit in tag_map["capex_candidates"]:
        capex_fy_minus_1 = pick_fact(cf, tag, unit, fy - 1)
        if capex_fy_minus_1 is not None:
            out["capex_fy_minus_1"] = capex_fy_minus_1
            break
    else:
        out["capex_fy_minus_1"] = None

    # CAPEX for fy (current year)
    capex_fy = None
    for tag, unit in tag_map["capex_candidates"]:
        capex_fy = pick_fact(cf, tag, unit, fy)
        if capex_fy is not None:
            out["capex_fy"] = capex_fy
            break
    else:
        out["capex_fy"] = None

    # Store 3-year OCF tuple (fy-2, fy-1, fy)
    oper = out.get("operating_cf")
    out["ocf_3year"] = (ocf_fy_minus_2, oper_prev, oper)
    # Store 3-year CAPEX tuple (fy-2, fy-1, fy)
    out["capex_3year"] = (capex_fy_minus_2, capex_fy_minus_1, capex_fy)

    # 3-year data for net_income (fy-2, fy-1, fy)
    ni_fy_minus_2 = None
    if ni_tag:
        ni_fy_minus_2 = pick_fact(cf, ni_tag, ni_unit, fy - 2)
    out["ni_3year"] = (ni_fy_minus_2, net_income_prev, out.get("net_income"))

    # 3-year data for operating_income - try to extract from SEC API
    oi_3year = (None, None, None)
    # operating_income tags (if available in XBRL)
    oi_candidates = [("OperatingIncomeLoss", "USD"), ("OperatingIncome", "USD")]
    oi_fy_minus_2 = None
    oi_fy_minus_1 = None
    oi_fy = None
    for tag, unit in oi_candidates:
        oi_fy_minus_2 = pick_fact(cf, tag, unit, fy - 2)
        if oi_fy_minus_2 is not None:
            break
    for tag, unit in oi_candidates:
        oi_fy_minus_1 = pick_fact(cf, tag, unit, fy - 1)
        if oi_fy_minus_1 is not None:
            break
    for tag, unit in oi_candidates:
        oi_fy = pick_fact(cf, tag, unit, fy)
        if oi_fy is not None:
            break
    out["oi_3year"] = (oi_fy_minus_2, oi_fy_minus_1, oi_fy)

    # compute capex if not directly available
    if out.get("capex_cashflow") is None:
        net_ppe = out.get("net_ppe")
        depr = out.get("depreciation") or 0.0
        proceeds = out.get("proceeds") or 0.0
        if net_ppe is not None and prev_net_ppe is not None:
            delta_netppe = net_ppe - prev_net_ppe
            # CAPEX ≈ ΔNetPPE + Depreciation - Proceeds
            approx_capex = delta_netppe + depr - proceeds
            out["capex_approx"] = approx_capex
            out["capex_cashflow"] = approx_capex
        else:
            out["capex_approx"] = None

    # normalize capex (interpret as positive cash spent)
    capex_cf = out.get("capex_cashflow")
    if capex_cf is not None:
        # if reported as negative cash outflow, make positive magnitude
        capex_abs = -capex_cf if capex_cf < 0 else capex_cf
    else:
        capex_abs = None
    out["capex_abs"] = capex_abs

    # compute free cash flow if possible
    oper = out.get("operating_cf")
    if oper is not None and capex_abs is not None:
        out["free_cash_flow"] = oper - capex_abs
    else:
        out["free_cash_flow"] = None

    return out


# ==========================
# Datenklassen
# ==========================

@dataclass
class RawFigures:
    """Rohzahlen aus dem SEC Filing"""
    company_name: str
    filing_year: int
    currency: str
    net_income: float  # in Millionen
    depreciation: float  # in Millionen
    equity: float  # in Millionen
    debt: float  # in Millionen
    cash: float
    dividends: float  # in Millionen
    diluted_eps: float  # per Share
    share_price: float
    fx_rate: float
    # optional / extended fields (defaults so existing callers still work)
    revenues: float = 0.0
    operating_income: float = 0.0
    operating_cf: float = 0.0
    capex: float = 0.0
    capex_abs: float = 0.0
    proceeds: float = 0.0
    net_ppe: float = 0.0
    prev_net_ppe: float = 0.0
    free_cash_flow: float = 0.0
    discount_rate: float = 0.1
    growth_rate: float = 0.02
    # Vorjahreswerte (falls vorhanden)
    prev_net_income: float = 0.0
    prev_operating_cf: float = 0.0
    prev_free_cash_flow: float = 0.0
    # 3-year raw data tuples (fy-2, fy-1, fy) for charting
    ni_3year_tuple: tuple = (0.0, 0.0, 0.0)  # Net Income 3-year
    oi_3year_tuple: tuple = (0.0, 0.0, 0.0)  # Operating Income 3-year
    ocf_3year_tuple: tuple = (0.0, 0.0, 0.0)  # Operating CF 3-year
    capex_3year_tuple: tuple = (0.0, 0.0, 0.0)  # CAPEX 3-year


@dataclass
class ValueMetrics:
    """Berechnete Kennzahlen"""
    ep: float     # Earnings-to-Price
    cep: float    # Cash-Earnings-to-Price
    bm: float     # Book-to-Market
    dp: float     # Dividend Yield
    eq_return: float  # Eigenkapitalrendite
    umsatzrendite: float  # Umsatzrendite
    kapitalumschlag: float  # Kapitalumschlag
    equity_valuation: float = 0.0  # Diskontierte EK-Bewertung (per DCF-Shortcut)
    market_cap_euro: float = 0.0    # Market Cap in EUR (oder lokaler Währung nach FX)
    implied_growth: float = 0.0     # implizite Wachstumsrate (g), so dass EQ-Valuation = Market Cap
    # Additional ratios
    fcf_to_equity: float = 0.0
    ocf_to_equity: float = 0.0
    fcf_to_debt: float = 0.0
    capex_to_ocf: float = 0.0
    # 3-year CAGR growth rates (decimal)
    ocf_cagr_3year: float = 0.0
    capex_cagr_3year: float = 0.0
    ni_cagr_3year: float = 0.0
    oi_cagr_3year: float = 0.0



@dataclass
class FilingResult:
    """Komplettes Ergebnis: Rohzahlen + Kennzahlen"""
    raw: RawFigures
    metrics: ValueMetrics


# ==========================
# Filing-Klasse für SEC API
# ==========================

class Filing:
    """
    Repräsentiert ein 10-K Filing über SEC API.
    Zieht Daten anhand von CIK und Fiskaljahrm berechnet Kennzahlen.
    """
    
    def __init__(self, cik: int | str, filing_year: int, company_name: str, fx_rate: float = 1.0, share_price: float = 0.0, discount_rate: float = 0.1, growth_rate: float = 0.02):
        """
        Initialisiert ein Filing mit CIK und Fiskaljahrm.
        
        Args:
            cik: SEC CIK Nummer (z.B. 1326801 für Meta)
            filing_year: Fiskaljahrm (z.B. 2024)
            company_name: Unternehmensname
            fx_rate: Umrechnungskurs USD->EUR (default: 1.0)
            share_price: Aktienkurs in USD
            discount_rate: Diskontierungssatz für DCF
            growth_rate: Wachstumsrate für DCF
        """
        self.cik = cik
        self.filing_year = filing_year
        self.company_name = company_name
        self.currency = "USD"  # Immer USD von SEC API
        self.fx_rate = fx_rate
        self.share_price = share_price
        # store discount/growth for later valuation
        self.discount_rate = discount_rate
        self.growth_rate = growth_rate
        self.metrics_dict: Optional[dict] = None
        self.raw_figures: Optional[RawFigures] = None
        self.metrics: Optional[ValueMetrics] = None
    
    def load(self) -> dict:
        """
        Zieht Rohmetdaten vom SEC API.
        Gibt Dictionary mit net_income, depreciation, equity, dividends, diluted_eps zurück.
        """
        print(f"Lade Daten für CIK {self.cik}, Jahr {self.filing_year}...")
        self.metrics_dict = get_core_metrics_from_companyfacts(self.cik, self.filing_year)
        print(f"✓ Daten geladen: {self.metrics_dict}")
        return self.metrics_dict
    
    def extract_raw_figures(self) -> RawFigures:
        """
        Erstellt RawFigures-Objekt aus den gezogenen SEC-Daten.
        
        Returns:
            RawFigures-Objekt
        """
        if self.metrics_dict is None:
            raise ValueError("Zuerst load() aufrufen!")
        
        # Werte aus SEC API (in den Einheiten, wie gemeldet)
        net_income = self.metrics_dict.get("net_income") or 0.0
        cash = self.metrics_dict.get("cash") or 0.0
        debt = self.metrics_dict.get("debt") or 0.0
        depreciation = self.metrics_dict.get("depreciation") or 0.0
        equity = self.metrics_dict.get("equity") or 0.0
        dividends = self.metrics_dict.get("dividends") or 0.0
        diluted_eps = self.metrics_dict.get("diluted_eps") or 0.0

        # optional: operating CF / capex / proceeds / net PPE
        operating_cf = self.metrics_dict.get("operating_cf") or 0.0
        capex = self.metrics_dict.get("capex_cashflow")
        capex_abs = self.metrics_dict.get("capex_abs") or 0.0
        proceeds = self.metrics_dict.get("proceeds") or 0.0
        net_ppe = self.metrics_dict.get("net_ppe") or 0.0
        prev_net_ppe = self.metrics_dict.get("prev_net_ppe") or 0.0
        free_cash_flow = self.metrics_dict.get("free_cash_flow")

        # revenues / operating_income may not be present in the pulled dict; keep defaults
        revenues = self.metrics_dict.get("revenues") or 0.0
        operating_income = self.metrics_dict.get("operating_income") or 0.0
        prev_net_income = self.metrics_dict.get("net_income_prev") or 0.0
        prev_operating_cf = self.metrics_dict.get("operating_cf_prev") or 0.0
        prev_free_cash_flow = self.metrics_dict.get("free_cash_flow_prev") or 0.0

        # 3-year tuples from metrics_dict
        ni_3year_tuple = self.metrics_dict.get("ni_3year", (None, None, None))
        ni_3year_tuple = tuple(x or 0.0 for x in ni_3year_tuple)  # Replace None with 0.0
        oi_3year_tuple = self.metrics_dict.get("oi_3year", (None, None, None))
        oi_3year_tuple = tuple(x or 0.0 for x in oi_3year_tuple)
        ocf_3year_tuple = self.metrics_dict.get("ocf_3year", (None, None, None))
        ocf_3year_tuple = tuple(x or 0.0 for x in ocf_3year_tuple)
        capex_3year_tuple = self.metrics_dict.get("capex_3year", (None, None, None))
        capex_3year_tuple = tuple(x or 0.0 for x in capex_3year_tuple)

        self.raw_figures = RawFigures(
            company_name=self.company_name,
            filing_year=self.filing_year,
            currency=self.currency,
            net_income=net_income,
            depreciation=depreciation,
            equity=equity,
            dividends=dividends,
            diluted_eps=diluted_eps,
            share_price=self.share_price,
            fx_rate=self.fx_rate,
            revenues=revenues,
            operating_income=operating_income,
            operating_cf=operating_cf,
            capex=capex or 0.0,
            capex_abs=capex_abs,
            proceeds=proceeds,
            net_ppe=net_ppe,
            prev_net_ppe=prev_net_ppe,
            free_cash_flow=free_cash_flow or 0.0,
            prev_net_income=prev_net_income,
            prev_operating_cf=prev_operating_cf,
            prev_free_cash_flow=prev_free_cash_flow,
            ni_3year_tuple=ni_3year_tuple,
            oi_3year_tuple=oi_3year_tuple,
            ocf_3year_tuple=ocf_3year_tuple,
            capex_3year_tuple=capex_3year_tuple,
            debt=debt,
            cash=cash,
            discount_rate=self.discount_rate,
            growth_rate=self.growth_rate
        )

        
        print(f"✓ RawFigures erstellt:")
        print(f"  - Net Income: ${net_income:,.0f}M")
        print(f"  - Depreciation: ${depreciation:,.0f}M")
        print(f"  - Equity: ${equity:,.0f}M")
        print(f"  - Diluted EPS: ${diluted_eps:.2f}")
        
        return self.raw_figures
    
    def process(self) -> FilingResult:
        """
        Convenience-Methode: Extrahiert RawFigures und berechnet Metrics in einem Schritt.
        Erwartet, dass load() bereits aufgerufen wurde.
        
        Returns:
            FilingResult mit RawFigures und ValueMetrics
        """
        self.extract_raw_figures()
        self.compute_metrics()
        return self.to_result()
    
    def compute_metrics(self) -> ValueMetrics:
        """
        Berechnet Kennzahlen aus RawFigures.
        
        Returns:
            ValueMetrics-Objekt
        """
        if self.raw_figures is None:
            raise ValueError("Zuerst extract_raw_figures() aufrufen!")
        
        # Helper function to calculate CAGR safely
        def safe_cagr(start_value, end_value):
            """
            Calculates CAGR: (End/Start)^(1/2) - 1
            Returns 0.0 if data is insufficient or invalid
            """
            if start_value and start_value > 0 and end_value is not None:
                try:
                    return (abs(end_value) / abs(start_value)) ** (1.0 / 2.0) - 1.0
                except (ValueError, ZeroDivisionError):
                    return 0.0
            return 0.0
        
        raw = self.raw_figures
        
        # Approximate Shares Outstanding using Net Income / Diluted EPS when possible
        net_income_abs = raw.net_income * 1_000_000  # convert reported (usually millions) to absolute
        if raw.diluted_eps and raw.diluted_eps > 0:
            shares_outstanding = net_income_abs / raw.diluted_eps
        else:
            shares_outstanding = 0

        market_cap_dollar = shares_outstanding * raw.share_price if shares_outstanding > 0 else 0

        # Earnings-to-Price: Net Income / Market Cap
        ep = (net_income_abs / market_cap_dollar) if market_cap_dollar > 0 else 0

        # Cash Earnings to Price: (Net Income + Depreciation) / Market Cap
        cash_earnings = (raw.net_income + raw.depreciation) * 1_000_000
        cep = (cash_earnings / market_cap_dollar) if market_cap_dollar > 0 else 0

        # Book-to-Market: Equity / Market Cap
        equity_abs = raw.equity * 1_000_000
        bm = (equity_abs / market_cap_dollar) if market_cap_dollar > 0 else 0

        # Dividend Yield: Dividends / Market Cap
        dividends_abs = raw.dividends * 1_000_000
        dp = (dividends_abs / market_cap_dollar) if market_cap_dollar > 0 else 0

        #Du-Pont
        eq_return = net_income_abs / equity_abs if equity_abs > 0 else 0
        umsatzrendite = raw.revenues / equity_abs if equity_abs > 0 else 0
        kapitalumschlag = raw.operating_income / raw.revenues if raw.revenues >0 else 0

        # compute absolute free cash flow in USD
        free_cash_flow_abs = (raw.free_cash_flow or 0.0) * 1_000_000

        # Equity valuation in USD and convert to EUR
        denom = (raw.discount_rate - raw.growth_rate)
        equity_valuation_dollar = free_cash_flow_abs / denom if denom != 0 else 0
        equity_valuation_euro = equity_valuation_dollar * raw.fx_rate if raw.fx_rate != 0 else 0

        # market cap in EUR
        market_cap_euro = market_cap_dollar * raw.fx_rate if raw.fx_rate != 0 else 0

        # implied growth g such that equity_valuation_euro == market_cap_euro
        # formula simplifies to: g = i - (free_cash_flow_abs / market_cap_dollar)
        if market_cap_dollar > 0:
            implied_growth = raw.discount_rate - (free_cash_flow_abs / market_cap_dollar)
        else:
            implied_growth = 0.0

        # Additional ratios: FCF/EQ, OCF/EQ, FCF/DEBT, CAPEX/OCF
        fcf_to_equity = free_cash_flow_abs / equity_abs if equity_abs > 0 else 0
        ocf_abs = (raw.operating_cf or 0.0) * 1_000_000
        ocf_to_equity = ocf_abs / equity_abs if equity_abs > 0 else 0
        debt_abs = (raw.debt or 0.0) * 1_000_000
        fcf_to_debt = free_cash_flow_abs / debt_abs if debt_abs > 0 else 0
        # CAPEX to OCF: shows what % of OCF goes to capex
        capex_to_ocf = (raw.capex_abs or 0.0) * 1_000_000 / ocf_abs if ocf_abs > 0 else 0

        # 3-year CAGR for OCF and CAPEX
        # Formula: CAGR = (Ending_Value / Beginning_Value) ^ (1/n) - 1, where n=2 (for 3 years)
        ocf_3year = self.metrics_dict.get("ocf_3year", (None, None, None))
        ocf_fy_minus_2, ocf_fy_minus_1, ocf_fy = ocf_3year
        ocf_cagr_3year = safe_cagr(ocf_fy_minus_2, ocf_fy)

        capex_3year = self.metrics_dict.get("capex_3year", (None, None, None))
        capex_fy_minus_2, capex_fy_minus_1, capex_fy = capex_3year
        capex_cagr_3year = safe_cagr(capex_fy_minus_2, capex_fy)

        # 3-year CAGR for Net Income
        ni_3year = self.metrics_dict.get("ni_3year", (None, None, None))
        ni_fy_minus_2, ni_fy_minus_1, ni_fy = ni_3year
        ni_cagr_3year = safe_cagr(ni_fy_minus_2, ni_fy)

        # 3-year CAGR for Operating Income
        oi_3year = self.metrics_dict.get("oi_3year", (None, None, None))
        oi_fy_minus_2, oi_fy_minus_1, oi_fy = oi_3year
        oi_cagr_3year = safe_cagr(oi_fy_minus_2, oi_fy)

        self.metrics = ValueMetrics(
            ep=ep,
            cep=cep,
            bm=bm,
            dp=dp,
            eq_return=eq_return,
            umsatzrendite=umsatzrendite,
            kapitalumschlag=kapitalumschlag,
            equity_valuation=equity_valuation_euro,
            market_cap_euro=market_cap_euro,
            implied_growth=implied_growth,
            fcf_to_equity=fcf_to_equity,
            ocf_to_equity=ocf_to_equity,
            fcf_to_debt=fcf_to_debt,
            capex_to_ocf=capex_to_ocf,
            ocf_cagr_3year=ocf_cagr_3year,
            capex_cagr_3year=capex_cagr_3year,
            ni_cagr_3year=ni_cagr_3year,
            oi_cagr_3year=oi_cagr_3year,
        )

        print(f"✓ Kennzahlen berechnet:")
        print(f"  - E/P Ratio: {ep:.6f}")
        print(f"  - C/E/P Ratio: {cep:.6f}")
        print(f"  - B/M Ratio: {bm:.6f}")
        print(f"  - D/P Ratio: {dp:.6f}")
        print(f"  - EQ-Valuation (EUR): {equity_valuation_euro:,.2f}")
        print(f"  - Market-Cap (EUR): {market_cap_euro:,.2f}")
        print(f"  - Implied growth g: {implied_growth:.6f} ({implied_growth*100:.2f}%)")
        print(f"  - EK-Rendite: {eq_return:.6f}")
        print(f"  - Umsatzrendite: {umsatzrendite:.6f}")
        print(f"  - Kapitalumschlag: {kapitalumschlag:.6f}")

        return self.metrics
    
    def to_result(self) -> FilingResult:
        """
        Gibt komplettes FilingResult-Objekt zurück.
        
        Returns:
            FilingResult mit RawFigures und ValueMetrics
        """
        if self.raw_figures is None or self.metrics is None:
            raise ValueError("Zuerst extract_raw_figures() und compute_metrics() aufrufen!")
        
        return FilingResult(raw=self.raw_figures, metrics=self.metrics)


# ==========================
# Test
# ==========================

if __name__ == "__main__":
    # Test: Meta 2024
    filing = Filing(cik=1326801, filing_year=2024, company_name="Meta", share_price=400, fx_rate=1.15)
    filing.load()
    result = filing.process()
    
    print("\n" + "="*60)
    print("ERGEBNIS")
    print("="*60)
    print(result)
