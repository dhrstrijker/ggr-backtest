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
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

# Configuration
SYMBOLS = ['DHT', 'FRO', 'ASC', 'ECO', 'NAT', 'TNK', 'INSW', 'TRMD', 'TOPS', 'TORO', 'PSHG']
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
    fig = go.Figure()

    # Sort by first date
    availability.sort(key=lambda x: x['first_date'] if x['first_date'] else pd.Timestamp.max)

    colors = ['#2ecc71' if a['coverage'] >= 90 else '#f39c12' if a['coverage'] >= 70 else '#e74c3c'
              for a in availability]

    for i, a in enumerate(availability):
        if a['first_date'] and a['last_date']:
            fig.add_trace(go.Bar(
                x=[(a['last_date'] - a['first_date']).days],
                y=[a['symbol']],
                base=[a['first_date']],
                orientation='h',
                marker_color=colors[i],
                text=f"{a['trading_days']} days ({a['coverage']:.0f}%)",
                textposition='inside',
                hovertemplate=f"<b>{a['symbol']}</b><br>" +
                              f"Start: {a['first_date'].date()}<br>" +
                              f"End: {a['last_date'].date()}<br>" +
                              f"Days: {a['trading_days']}<br>" +
                              f"Coverage: {a['coverage']:.1f}%<extra></extra>"
            ))

    # Add vertical lines for common range
    first_dates = [a['first_date'] for a in availability if a['first_date']]
    last_dates = [a['last_date'] for a in availability if a['last_date']]

    if first_dates and last_dates:
        common_start = max(first_dates)
        common_end = min(last_dates)

        fig.add_shape(type="line", x0=common_start, x1=common_start,
                      y0=-0.5, y1=len(availability) - 0.5,
                      line=dict(color="red", width=2, dash="dash"))
        fig.add_shape(type="line", x0=common_end, x1=common_end,
                      y0=-0.5, y1=len(availability) - 0.5,
                      line=dict(color="red", width=2, dash="dash"))

        fig.add_annotation(x=common_start, y=len(availability),
                           text=f"Aligned Start: {common_start.date()}", showarrow=False,
                           font=dict(color="red"))

    fig.update_layout(
        title="Data Availability Timeline by Symbol<br><sup>Red lines = aligned date range used for backtest</sup>",
        xaxis_title="Date",
        yaxis_title="Symbol",
        showlegend=False,
        height=100 + len(availability) * 40,
        xaxis=dict(type='date'),
        barmode='overlay'
    )

    fig.write_html("data_availability_timeline.html")
    fig.show()

if __name__ == "__main__":
    main()
