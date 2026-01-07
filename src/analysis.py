"""Performance analysis and visualization module."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .backtest import Trade


def calculate_metrics(
    trades: list[Trade],
    equity_curve: pd.Series,
    risk_free_rate: float = 0.02,
) -> dict[str, Any]:
    """
    Calculate performance metrics from backtest results.

    Args:
        trades: List of completed trades
        equity_curve: Equity curve series
        risk_free_rate: Annual risk-free rate for Sharpe calculation

    Returns:
        Dictionary with performance metrics
    """
    if not trades:
        return {
            "total_trades": 0,
            "total_return": 0,
            "total_return_pct": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "max_drawdown_pct": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "avg_holding_days": 0,
        }

    # Basic stats
    pnls = [t.pnl for t in trades]
    pnl_pcts = [t.pnl_pct for t in trades]
    total_pnl = sum(pnls)
    initial_capital = equity_curve.iloc[0]
    total_return_pct = (equity_curve.iloc[-1] - initial_capital) / initial_capital

    # Win rate
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls) if pnls else 0

    # Average win/loss
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0

    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float("inf")

    # Sharpe ratio (annualized)
    returns = equity_curve.pct_change().dropna()
    if len(returns) > 0 and returns.std() > 0:
        # Assume ~252 trading days per year
        excess_returns = returns - risk_free_rate / 252
        sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
    else:
        sharpe = 0

    # Max drawdown
    rolling_max = equity_curve.expanding().max()
    drawdown = equity_curve - rolling_max
    max_drawdown = drawdown.min()
    max_drawdown_pct = (max_drawdown / rolling_max[drawdown.idxmin()]) if max_drawdown < 0 else 0

    # Average holding period
    avg_holding_days = np.mean([t.holding_days for t in trades])

    # Trade breakdown by direction
    long_trades = [t for t in trades if t.direction == 1]
    short_trades = [t for t in trades if t.direction == -1]

    return {
        "total_trades": len(trades),
        "total_return": total_pnl,
        "total_return_pct": total_return_pct,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "avg_holding_days": avg_holding_days,
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "long_win_rate": len([t for t in long_trades if t.pnl > 0]) / len(long_trades) if long_trades else 0,
        "short_win_rate": len([t for t in short_trades if t.pnl > 0]) / len(short_trades) if short_trades else 0,
    }


def print_metrics(metrics: dict[str, Any]) -> None:
    """Print formatted performance metrics."""
    print("=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Total Trades:     {metrics['total_trades']}")
    print(f"Total Return:     ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:.2%})")
    print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:     ${metrics['max_drawdown']:,.2f} ({metrics['max_drawdown_pct']:.2%})")
    print("-" * 50)
    print(f"Win Rate:         {metrics['win_rate']:.2%}")
    print(f"Avg Win:          ${metrics['avg_win']:,.2f}")
    print(f"Avg Loss:         ${metrics['avg_loss']:,.2f}")
    print(f"Profit Factor:    {metrics['profit_factor']:.2f}")
    print(f"Avg Holding Days: {metrics['avg_holding_days']:.1f}")
    print("-" * 50)
    print(f"Long Trades:      {metrics['long_trades']} ({metrics['long_win_rate']:.2%} win rate)")
    print(f"Short Trades:     {metrics['short_trades']} ({metrics['short_win_rate']:.2%} win rate)")
    print("=" * 50)


def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    """Convert list of trades to DataFrame."""
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([t.to_dict() for t in trades])


def plot_equity_curve(
    equity_curve: pd.Series,
    title: str = "Equity Curve",
    show_drawdown: bool = True,
) -> go.Figure:
    """
    Plot equity curve with optional drawdown overlay.

    Args:
        equity_curve: Equity curve series
        title: Chart title
        show_drawdown: Whether to show drawdown in secondary axis

    Returns:
        Plotly Figure object
    """
    if show_drawdown:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
        )

        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode="lines",
                name="Equity",
                line=dict(color="blue", width=2),
            ),
            row=1, col=1,
        )

        # Drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max * 100

        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode="lines",
                name="Drawdown %",
                fill="tozeroy",
                line=dict(color="red", width=1),
                fillcolor="rgba(255, 0, 0, 0.2)",
            ),
            row=2, col=1,
        )

        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode="lines",
                name="Equity",
                line=dict(color="blue", width=2),
            )
        )
        fig.update_yaxes(title_text="Equity ($)")

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )

    return fig


def plot_trade(
    close_prices: pd.DataFrame,
    trade: Trade,
    distance: pd.Series,
    lookback_days: int = 60,
) -> go.Figure:
    """
    Plot a single trade with price series and distance (GGR methodology).

    Args:
        close_prices: DataFrame with close prices
        trade: Trade object to visualize
        distance: Distance series (spread / formation_std)
        lookback_days: Days before entry to show

    Returns:
        Plotly Figure object
    """
    sym_a, sym_b = trade.pair

    # Find date range
    entry_idx = close_prices.index.get_loc(trade.entry_date)
    start_idx = max(0, entry_idx - lookback_days)
    exit_idx = close_prices.index.get_loc(trade.exit_date)
    end_idx = min(len(close_prices), exit_idx + 10)

    date_range = close_prices.index[start_idx:end_idx]
    prices_a = close_prices[sym_a].loc[date_range]
    prices_b = close_prices[sym_b].loc[date_range]
    d = distance.loc[date_range]

    # Normalize for comparison
    norm_a = prices_a / prices_a.iloc[0] * 100
    norm_b = prices_b / prices_b.iloc[0] * 100

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.1,
    )

    # Price series (normalized)
    fig.add_trace(
        go.Scatter(
            x=norm_a.index, y=norm_a.values,
            mode="lines", name=sym_a,
            line=dict(color="blue", width=2),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=norm_b.index, y=norm_b.values,
            mode="lines", name=sym_b,
            line=dict(color="orange", width=2),
        ),
        row=1, col=1,
    )

    # Entry/exit markers
    fig.add_vline(x=trade.entry_date, line_dash="dash", line_color="green", row=1)
    fig.add_vline(x=trade.exit_date, line_dash="dash", line_color="red", row=1)
    fig.add_vline(x=trade.entry_date, line_dash="dash", line_color="green", row=2)
    fig.add_vline(x=trade.exit_date, line_dash="dash", line_color="red", row=2)

    # Distance (GGR: static σ from formation)
    fig.add_trace(
        go.Scatter(
            x=d.index, y=d.values,
            mode="lines", name="Distance (σ)",
            line=dict(color="purple", width=2),
        ),
        row=2, col=1,
    )

    # Threshold lines (GGR: ±2σ entry, 0 exit)
    fig.add_hline(y=2, line_dash="dot", line_color="red", row=2,
                  annotation_text="+2σ Entry")
    fig.add_hline(y=-2, line_dash="dot", line_color="green", row=2,
                  annotation_text="-2σ Entry")
    fig.add_hline(y=0, line_dash="solid", line_color="black", row=2,
                  annotation_text="Exit (crossing)")

    direction = "Long" if trade.direction == 1 else "Short"
    fig.update_layout(
        title=f"{sym_a}/{sym_b} - {direction} Trade (P&L: ${trade.pnl:.2f})",
        template="plotly_white",
        height=600,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Normalized Price", row=1, col=1)
    fig.update_yaxes(title_text="Distance (σ)", row=2, col=1)

    return fig


def plot_ssd_heatmap(
    ssd_matrix: pd.DataFrame,
    title: str = "SSD Heatmap (Lower = More Similar)",
) -> go.Figure:
    """
    Plot SSD matrix as a heatmap.

    Args:
        ssd_matrix: SSD matrix from calculate_ssd_matrix
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=ssd_matrix.values,
            x=ssd_matrix.columns,
            y=ssd_matrix.index,
            colorscale="RdYlGn_r",  # Red = high (different), Green = low (similar)
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=500,
        width=600,
    )

    return fig


def plot_pair_prices(
    close_prices: pd.DataFrame,
    pair: tuple[str, str],
    title: str | None = None,
) -> go.Figure:
    """
    Plot normalized prices for a pair.

    Args:
        close_prices: DataFrame with close prices
        pair: Tuple of (symbol_a, symbol_b)
        title: Optional chart title

    Returns:
        Plotly Figure object
    """
    sym_a, sym_b = pair
    prices_a = close_prices[sym_a]
    prices_b = close_prices[sym_b]

    # Normalize
    norm_a = prices_a / prices_a.iloc[0] * 100
    norm_b = prices_b / prices_b.iloc[0] * 100

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=norm_a.index, y=norm_a.values,
            mode="lines", name=sym_a,
            line=dict(width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=norm_b.index, y=norm_b.values,
            mode="lines", name=sym_b,
            line=dict(width=2),
        )
    )

    if title is None:
        title = f"{sym_a} vs {sym_b} (Normalized)"

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Normalized Price (base 100)",
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


def plot_pair_analysis(
    close_prices: pd.DataFrame,
    pair: tuple[str, str],
    trades: list[Trade],
    distance: pd.Series,
    entry_threshold: float = 2.0,
) -> go.Figure:
    """
    Plot comprehensive pair analysis with all trades marked.

    Shows:
    - Row 1: Normalized prices for both stocks with entry/exit markers
    - Row 2: Distance (in σ) with threshold lines and trade regions

    Per GGR methodology: Exit occurs when spread crosses zero, not at a threshold.

    Args:
        close_prices: DataFrame with close prices
        pair: Tuple of (symbol_a, symbol_b)
        trades: List of Trade objects for this pair
        distance: Distance series (spread / formation_std) for this pair
        entry_threshold: Entry threshold in σ for reference lines

    Returns:
        Plotly Figure object
    """
    sym_a, sym_b = pair

    # Normalize prices
    prices_a = close_prices[sym_a]
    prices_b = close_prices[sym_b]
    norm_a = prices_a / prices_a.iloc[0] * 100
    norm_b = prices_b / prices_b.iloc[0] * 100

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.08,
        subplot_titles=(f"{sym_a} vs {sym_b} (Normalized Prices)", "Distance (σ from formation)"),
    )

    # Row 1: Normalized prices
    fig.add_trace(
        go.Scatter(
            x=norm_a.index, y=norm_a.values,
            mode="lines", name=sym_a,
            line=dict(color="blue", width=2),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=norm_b.index, y=norm_b.values,
            mode="lines", name=sym_b,
            line=dict(color="orange", width=2),
        ),
        row=1, col=1,
    )

    # Row 2: Distance (in σ)
    fig.add_trace(
        go.Scatter(
            x=distance.index, y=distance.values,
            mode="lines", name="Distance (σ)",
            line=dict(color="purple", width=2),
        ),
        row=2, col=1,
    )

    # Add threshold lines (GGR: entry at ±2σ, exit on crossing zero)
    fig.add_hline(y=entry_threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Short Entry (+{entry_threshold}σ)", row=2, col=1)
    fig.add_hline(y=-entry_threshold, line_dash="dash", line_color="green",
                  annotation_text=f"Long Entry (-{entry_threshold}σ)", row=2, col=1)
    fig.add_hline(y=0, line_color="black", line_width=2,
                  annotation_text="Exit (crossing)", row=2, col=1)

    # Add trade markers and shaded regions
    colors = {"long": "green", "short": "red"}

    for i, trade in enumerate(trades):
        direction = "long" if trade.direction == 1 else "short"
        color = colors[direction]

        # Shaded region for trade duration (on both subplots)
        fig.add_vrect(
            x0=trade.entry_date, x1=trade.exit_date,
            fillcolor=color, opacity=0.15,
            line_width=0,
            row=1, col=1,
        )
        fig.add_vrect(
            x0=trade.entry_date, x1=trade.exit_date,
            fillcolor=color, opacity=0.15,
            line_width=0,
            row=2, col=1,
        )

        # Entry marker
        fig.add_trace(
            go.Scatter(
                x=[trade.entry_date],
                y=[distance.loc[trade.entry_date] if trade.entry_date in distance.index else trade.entry_distance],
                mode="markers",
                marker=dict(symbol="triangle-up" if direction == "long" else "triangle-down",
                           size=12, color=color),
                name=f"Entry #{i+1}" if i == 0 else None,
                showlegend=(i == 0),
                hovertemplate=f"Entry #{i+1}<br>Date: %{{x}}<br>Distance: {trade.entry_distance:.2f}σ<br>Direction: {direction}<extra></extra>",
            ),
            row=2, col=1,
        )

        # Exit marker
        fig.add_trace(
            go.Scatter(
                x=[trade.exit_date],
                y=[distance.loc[trade.exit_date] if trade.exit_date in distance.index else trade.exit_distance],
                mode="markers",
                marker=dict(symbol="x", size=10, color=color),
                name=f"Exit #{i+1}" if i == 0 else None,
                showlegend=False,
                hovertemplate=f"Exit #{i+1}<br>Date: %{{x}}<br>Distance: {trade.exit_distance:.2f}σ<br>P&L: ${trade.pnl:.2f}<br>Reason: {trade.exit_reason}<extra></extra>",
            ),
            row=2, col=1,
        )

    # Calculate summary stats
    total_trades = len(trades)
    total_pnl = sum(t.pnl for t in trades)
    win_rate = len([t for t in trades if t.pnl > 0]) / total_trades if total_trades > 0 else 0

    fig.update_layout(
        title=f"{sym_a}/{sym_b} Pair Analysis | Trades: {total_trades} | P&L: ${total_pnl:,.2f} | Win Rate: {win_rate:.1%}",
        template="plotly_white",
        height=700,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Normalized Price (base 100)", row=1, col=1)
    fig.update_yaxes(title_text="Distance (σ)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig


def generate_pair_report(
    close_prices: pd.DataFrame,
    pair: tuple[str, str],
    trades: list[Trade],
    distance: pd.Series,
    config: dict,
) -> dict:
    """
    Generate a detailed report for a single pair.

    Args:
        close_prices: DataFrame with close prices
        pair: Tuple of (symbol_a, symbol_b)
        trades: List of trades for this pair
        distance: Distance series (spread / formation_std)
        config: Backtest configuration dict

    Returns:
        Dictionary with pair statistics and figure
    """
    sym_a, sym_b = pair

    # Calculate pair-level stats
    total_trades = len(trades)
    if total_trades > 0:
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in trades)
        win_rate = len(winning_trades) / total_trades
        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
        avg_holding = sum(t.holding_days for t in trades) / total_trades

        long_trades = [t for t in trades if t.direction == 1]
        short_trades = [t for t in trades if t.direction == -1]
    else:
        total_pnl = win_rate = avg_win = avg_loss = avg_holding = 0
        long_trades = short_trades = []

    # Create the visualization
    fig = plot_pair_analysis(
        close_prices, pair, trades, distance,
        entry_threshold=config.get("entry_threshold", 2.0),
    )

    # Create trades table
    trades_data = []
    for i, t in enumerate(trades, 1):
        trades_data.append({
            "#": i,
            "Entry": t.entry_date.strftime("%Y-%m-%d"),
            "Exit": t.exit_date.strftime("%Y-%m-%d"),
            "Direction": "Long" if t.direction == 1 else "Short",
            "Days": t.holding_days,
            "Entry σ": f"{t.entry_distance:.2f}",
            "Exit σ": f"{t.exit_distance:.2f}",
            "P&L": f"${t.pnl:.2f}",
            "Return": f"{t.pnl_pct:.2%}",
            "Exit Reason": t.exit_reason,
        })

    return {
        "pair": f"{sym_a}/{sym_b}",
        "total_trades": total_trades,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_holding_days": avg_holding,
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "figure": fig,
        "trades_table": pd.DataFrame(trades_data) if trades_data else pd.DataFrame(),
    }


def print_pair_report(report: dict) -> None:
    """Print a formatted pair report."""
    print("=" * 70)
    print(f"PAIR ANALYSIS: {report['pair']}")
    print("=" * 70)
    print(f"\nTotal Trades:      {report['total_trades']}")
    print(f"Total P&L:         ${report['total_pnl']:,.2f}")
    print(f"Win Rate:          {report['win_rate']:.1%}")
    print(f"Avg Win:           ${report['avg_win']:,.2f}")
    print(f"Avg Loss:          ${report['avg_loss']:,.2f}")
    print(f"Avg Holding Days:  {report['avg_holding_days']:.1f}")
    print(f"Long Trades:       {report['long_trades']}")
    print(f"Short Trades:      {report['short_trades']}")
    print("\nTrades:")
    print("-" * 70)
    if not report['trades_table'].empty:
        print(report['trades_table'].to_string(index=False))
    else:
        print("No trades for this pair.")
    print("=" * 70)


def plot_zscore_series(
    zscore: pd.Series,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    title: str = "Z-Score Series",
) -> go.Figure:
    """
    Plot z-score series with threshold lines.

    Args:
        zscore: Z-score series
        entry_threshold: Entry threshold value
        exit_threshold: Exit threshold value
        title: Chart title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Z-score line
    fig.add_trace(
        go.Scatter(
            x=zscore.index, y=zscore.values,
            mode="lines", name="Z-Score",
            line=dict(color="purple", width=2),
        )
    )

    # Fill areas
    fig.add_hrect(
        y0=entry_threshold, y1=10,
        fillcolor="red", opacity=0.1,
        line_width=0,
        annotation_text="Short Entry Zone",
    )
    fig.add_hrect(
        y0=-10, y1=-entry_threshold,
        fillcolor="green", opacity=0.1,
        line_width=0,
        annotation_text="Long Entry Zone",
    )

    # Threshold lines
    fig.add_hline(y=entry_threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Entry ({entry_threshold}σ)")
    fig.add_hline(y=-entry_threshold, line_dash="dash", line_color="green",
                  annotation_text=f"Entry (-{entry_threshold}σ)")
    fig.add_hline(y=exit_threshold, line_dash="dot", line_color="gray",
                  annotation_text=f"Exit ({exit_threshold}σ)")
    fig.add_hline(y=-exit_threshold, line_dash="dot", line_color="gray")
    fig.add_hline(y=0, line_color="black")

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Z-Score",
        template="plotly_white",
        hovermode="x unified",
    )

    return fig
