
```markdown
```mermaid
classDiagram
    class Filing {
        +path: str
        +text: str
        +raw_figures: RawFigures
        +metrics: ValueMetrics
        +load(): None
        +extract_raw_figures(company_name: str, filing_year: int,
                             currency: str, share_price: float, fx_rate: float): None
        +compute_metrics(): None
        +to_result(): FilingResult
    }

    class RawFigures {
        +company_name: str
        +filing_year: int
        +currency: str
        +net_income: float
        +depreciation: float
        +equity: float
        +dividends: float
        +diluted_shares: float
        +share_price: float
        +fx_rate: float
    }

    class ValueMetrics {
        +ep: float
        +cep: float
        +bm: float
        +dp: float
    }

    class FilingResult {
        +raw: RawFigures
        +metrics: ValueMetrics
    }

    class Loader {
        +load_10k_text(path: str) str
    }

    class Parser {
        <<static>>
        +parse_number(num_str: str) float
        +extract_metric(text: str, patterns: list[str], name: str) float
        +extract_raw_figures_from_text(text: str,
                                       company_name: str,
                                       filing_year: int,
                                       currency: str,
                                       share_price: float,
                                       fx_rate: float) RawFigures
    }

    class Calculator {
        +compute_value_metrics(raw: RawFigures) ValueMetrics
    }

    class Pipeline {
        <<static>>
        +process_filing(path: str, company_name: str,
                        filing_year: int, currency: str,
                        share_price: float,
                        fx_rate: float) FilingResult
    }

    Filing --> RawFigures : holds
    Filing --> ValueMetrics : computes
    FilingResult --> RawFigures : contains
    FilingResult --> ValueMetrics : contains
    Filing ..> Loader : uses
    Parser ..> RawFigures : produces
    Calculator ..> ValueMetrics : produces
    Pipeline ..> Filing : orchestrates