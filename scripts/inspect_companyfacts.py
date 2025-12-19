#!/usr/bin/env python3
"""
Inspect available XBRL tags in the SEC companyfacts JSON for a given CIK and fiscal year.
Usage:
    python3 scripts/inspect_companyfacts.py --cik 1326801 --fy 2024

This script reuses `sec_get_json` and `cik10` from `main_api.py`.
"""
import argparse
import json
from pprint import pprint
from typing import List, Tuple

from main_api import cik10, sec_get_json, pick_fact

# Tags to probe: friendly name -> list of (xbrl_tag, unit_hint)
PROBE_TAGS = {
    'operating_cf': [('NetCashProvidedByUsedInOperatingActivities', 'USD')],
    'capex_candidates': [
        ('PurchasesOfPropertyPlantAndEquipment', 'USD'),
        ('PaymentsToAcquirePropertyPlantAndEquipment', 'USD'),
        ('AdditionsToPropertyPlantAndEquipment', 'USD'),
        ('CapitalExpenditures', 'USD'),
    ],
    'net_ppe': [('PropertyPlantAndEquipmentNet', 'USD')],
    'depreciation': [('DepreciationDepletionAndAmortization', 'USD'), ('DepreciationAndAmortization', 'USD')],
    'proceeds_candidates': [
        ('ProceedsFromSaleOfPropertyPlantAndEquipment', 'USD'),
        ('ProceedsFromDispositionOfPropertyPlantAndEquipment', 'USD'),
    ],
    'gain_on_sale': [('GainOnSaleOfAsset', 'USD'), ('GainLossOnDispositionOfAssets', 'USD')],
    'net_income': [('NetIncomeLoss', 'USD')],
    'revenues': [('Revenues', 'USD'), ('SalesRevenueNet', 'USD')],
    'assets': [('Assets', 'USD')],
    'shares_outstanding': [
        ('WeightedAverageNumberOfDilutedSharesOutstanding', 'shares'),
        ('CommonStockSharesOutstanding', 'shares'),
        ('EntityCommonStockSharesOutstanding', 'shares')
    ],
    'diluted_eps': [('EarningsPerShareDiluted', 'USD/shares'), ('EarningsPerShareBasic', 'USD/shares')]
}


def pick_first(cf: dict, tag_list: List[Tuple[str, str]], fy: int):
    """Try pick_fact for each (tag, unit) in tag_list and return first non-None."""
    for tag, unit in tag_list:
        try:
            v = pick_fact(cf, tag, unit, fy)
        except Exception:
            v = None
        if v is not None:
            return (tag, unit, v)
    return None


def inspect_cik(cik: str, fy: int):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10(cik)}.json"
    print(f"Fetching {url} ...")
    cf = sec_get_json(url)

    usgaap = cf.get('facts', {}).get('us-gaap', {})
    print(f"Found {len(usgaap)} us-gaap tags in companyfacts")

    results = {}
    for name, tag_list in PROBE_TAGS.items():
        if name.endswith('_candidates'):
            res = pick_first(cf, tag_list, fy)
            results[name] = res
        else:
            res = pick_first(cf, tag_list, fy)
            results[name] = res

    # Also report which of the capex candidate tags exist at all in the facts (regardless of FY)
    capex_tag_presence = {}
    for tag, unit in PROBE_TAGS['capex_candidates']:
        capex_tag_presence[tag] = tag in usgaap

    print('\n=== Probe results for CIK', cik, 'FY', fy, '===')
    for k, v in results.items():
        if v is None:
            print(f"{k:20s}: MISSING for FY {fy}")
        else:
            tag, unit, val = v
            print(f"{k:20s}: tag={tag:40s} unit={unit:12s} val={val}")

    print('\nCapEx tag presence in us-gaap (any FY):')
    for tag, present in capex_tag_presence.items():
        print(f"  {tag:50s}: {'YES' if present else 'NO'}")

    # Optionally pretty-print parts of the JSON for debugging
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cik', type=str, required=False, default='1326801', help='CIK (e.g. 1326801 for Meta)')
    p.add_argument('--fy', type=int, required=False, default=2024, help='Fiscal year to probe')
    args = p.parse_args()

    inspect_cik(args.cik, args.fy)
