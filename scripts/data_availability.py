#!/usr/bin/env python3
"""
Show data availability timeline for each symbol.
Helps identify which stocks to drop for longer backtests.
"""

import sys
sys.path.insert(0, '.')

import os
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

# Configuration
SYMBOLS = ["NEE", "DUK", "SO", "AEP", "SRE", "D", "EXC", "XEL", "ED", "PEG", "WEC", "ES", "ETR", "PPL", "AEE", "DTE", "FE", "CMS", "CNP", "ATO", "EVRG", "LNT", "NI", "AWK", "NRG", "PNW", "OGE", "IDA", "HE", "ALE", "POR", "UGI", "SR", "BKH", "NWE"]
START_DATE = "2021-01-01"
END_DATE = "2026-01-01"

def fetch_single_symbol(api_key, symbol, start_date, end_date):
    """Fetch data for a single symbol using Polygon API."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("resultsCount", 0) == 0:
            return None

        results = data.get("results", [])
        if not results:
            return None

        df = pd.DataFrame(results)
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.set_index("date")
        return df[["c"]].rename(columns={"c": "close"})
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    print(f"Fetching RAW data for {len(SYMBOLS)} symbols from {START_DATE} to {END_DATE}...")
    print("(This fetches each symbol individually, before alignment)\n")

    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        print("ERROR: POLYGON_API_KEY not found in .env")
        return

    # Fetch each symbol individually
    availability = []

    for symbol in SYMBOLS:
        print(f"  Fetching {symbol}...", end=" ", flush=True)
        df = fetch_single_symbol(api_key, symbol, START_DATE, END_DATE)

        if df is not None and len(df) > 0:
            first_date = df.index.min()
            last_date = df.index.max()
            trading_days = len(df)
            print(f"{trading_days} days ({first_date.date()} to {last_date.date()})")
        else:
            first_date = None
            last_date = None
            trading_days = 0
            print("NO DATA")

        availability.append({
            'symbol': symbol,
            'first_date': first_date,
            'last_date': last_date,
            'trading_days': trading_days
        })

    # Calculate coverage based on longest available
    max_days = max(a['trading_days'] for a in availability) if availability else 1
    for a in availability:
        a['coverage'] = (a['trading_days'] / max_days * 100) if max_days > 0 else 0

    # Summary
    print("\n" + "=" * 70)
    print("DATA AVAILABILITY SUMMARY")
    print("=" * 70)

    # Sort by first date
    availability.sort(key=lambda x: x['first_date'] if x['first_date'] else pd.Timestamp.max)

    for a in availability:
        if a['first_date']:
            bar = "█" * int(a['coverage'] / 5) + "░" * (20 - int(a['coverage'] / 5))
            print(f"{a['symbol']:6} {a['first_date'].date()} → {a['last_date'].date()}  {bar} {a['trading_days']:4} days")
        else:
            print(f"{a['symbol']:6} NO DATA")

    # Find common date range
    print("\n" + "=" * 70)
    print("ALIGNMENT ANALYSIS")
    print("=" * 70)

    first_dates = [a['first_date'] for a in availability if a['first_date']]
    last_dates = [a['last_date'] for a in availability if a['last_date']]

    if first_dates and last_dates:
        earliest_start = min(first_dates)
        latest_start = max(first_dates)
        earliest_end = min(last_dates)
        latest_end = max(last_dates)

        aligned_days = (earliest_end - latest_start).days

        print(f"\nEarliest data starts: {earliest_start.date()}")
        print(f"Latest data starts:   {latest_start.date()} ← (alignment uses this)")
        print(f"Earliest data ends:   {earliest_end.date()} ← (alignment uses this)")
        print(f"Latest data ends:     {latest_end.date()}")
        print(f"\nAligned range: ~{aligned_days} days")

        # Symbols causing the limitation
        print(f"\n🔴 Symbols LIMITING the start date (start late):")
        for a in availability:
            if a['first_date'] and a['first_date'] == latest_start:
                print(f"   {a['symbol']}: starts {a['first_date'].date()}")

        print(f"\n🔴 Symbols LIMITING the end date (end early):")
        for a in availability:
            if a['last_date'] and a['last_date'] == earliest_end:
                print(f"   {a['symbol']}: ends {a['last_date'].date()}")

        # Calculate what happens if we drop limiting symbols
        print("\n" + "=" * 70)
        print("DROP ANALYSIS - What if we remove late starters?")
        print("=" * 70)

        # Find symbols that start significantly later
        late_starters = [(a['symbol'], a['first_date']) for a in availability
                         if a['first_date'] and (a['first_date'] - earliest_start).days > 30]
        late_starters.sort(key=lambda x: x[1], reverse=True)

        if late_starters:
            print("\nSymbols starting >30 days after earliest:")
            for sym, date in late_starters:
                days_lost = (date - earliest_start).days
                print(f"\n   {sym}: starts {date.date()} ({days_lost} days after earliest)")
                # What if we dropped this symbol?
                remaining = [a for a in availability if a['symbol'] != sym and a['first_date']]
                if remaining:
                    new_start = max(a['first_date'] for a in remaining)
                    new_end = min(a['last_date'] for a in remaining)
                    new_days = (new_end - new_start).days
                    gain = new_days - aligned_days
                    print(f"       → Drop {sym}: range becomes {new_start.date()} to {new_end.date()}")
                    print(f"          (~{new_days} days, +{gain} days gained)")
        else:
            print("\nNo symbols starting significantly late.")

    # Create timeline visualization
    print("\n" + "=" * 70)
    create_timeline_chart(availability)
    print("Timeline saved to: data_availability_timeline.html")

def create_timeline_chart(availability):
    """Create a Gantt-style timeline chart."""
    # Build data for timeline
    timeline_data = []
    for a in availability:
        if a['first_date'] and a['last_date']:
            coverage_cat = 'Full (≥90%)' if a['coverage'] >= 90 else 'Partial (70-90%)' if a['coverage'] >= 70 else 'Limited (<70%)'
            timeline_data.append({
                'Symbol': a['symbol'],
                'Start': a['first_date'],
                'End': a['last_date'],
                'Days': a['trading_days'],
                'Coverage': f"{a['coverage']:.0f}%",
                'Coverage Category': coverage_cat,
            })

    df = pd.DataFrame(timeline_data)

    # Create timeline using plotly express
    color_map = {
        'Full (≥90%)': '#2ecc71',
        'Partial (70-90%)': '#f39c12',
        'Limited (<70%)': '#e74c3c'
    }

    fig = px.timeline(
        df,
        x_start='Start',
        x_end='End',
        y='Symbol',
        color='Coverage Category',
        color_discrete_map=color_map,
        hover_data=['Days', 'Coverage'],
        title="Data Availability Timeline by Symbol"
    )

    # Add vertical lines for aligned range
    first_dates = [a['first_date'] for a in availability if a['first_date']]
    last_dates = [a['last_date'] for a in availability if a['last_date']]

    if first_dates and last_dates:
        common_start = max(first_dates)
        common_end = min(last_dates)

        # Add vertical lines using shapes (more reliable with dates)
        fig.add_shape(
            type="line",
            x0=common_start, x1=common_start,
            y0=0, y1=1, yref="paper",
            line=dict(color="red", width=2, dash="dash")
        )
        fig.add_shape(
            type="line",
            x0=common_end, x1=common_end,
            y0=0, y1=1, yref="paper",
            line=dict(color="blue", width=2, dash="dash")
        )

        # Add annotations separately
        fig.add_annotation(
            x=common_start, y=1.05, yref="paper",
            text=f"Aligned Start: {common_start.date()}",
            showarrow=False, font=dict(color="red", size=10)
        )
        fig.add_annotation(
            x=common_end, y=1.05, yref="paper",
            text=f"Aligned End: {common_end.date()}",
            showarrow=False, font=dict(color="blue", size=10)
        )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Symbol",
        height=100 + len(availability) * 40,
    )

    fig.write_html("data_availability_timeline.html", include_plotlyjs=True)
    fig.show()

if __name__ == "__main__":
    main()
