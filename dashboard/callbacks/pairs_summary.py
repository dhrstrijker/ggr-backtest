"""Callbacks for Page 4: Pairs Summary."""

from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

from ..components.metrics_card import format_currency, format_percentage


def register_pairs_summary_callbacks(app, data_store):
    """Register Pairs Summary page callbacks."""

    @app.callback(
        Output("pairs-summary-table", "children"),
        Input("wait-mode-store", "data"),
    )
    def update_pairs_table(wait_mode):
        """Build the pairs summary table."""
        results = data_store.get_results(wait_mode)

        rows = []
        for i, pair in enumerate(data_store.pairs, 1):
            sym_a, sym_b = pair
            pair_key = f"{sym_a}_{sym_b}"

            # Get trades for this pair
            trades = results[pair].trades if pair in results else []

            # Calculate stats
            total_pnl = sum(t.pnl for t in trades)
            num_trades = len(trades)
            win_count = sum(1 for t in trades if t.pnl > 0)
            win_rate = (win_count / num_trades * 100) if num_trades > 0 else 0
            avg_pnl = total_pnl / num_trades if num_trades > 0 else 0

            # Get SSD and correlation
            ssd = data_store.ssd_matrix.loc[sym_a, sym_b]
            corr = data_store.pair_correlations.get(pair, 0)

            pnl_color = "text-success" if total_pnl >= 0 else "text-danger"
            avg_pnl_color = "text-success" if avg_pnl >= 0 else "text-danger"

            rows.append(
                html.Tr(
                    [
                        html.Td(str(i)),
                        html.Td(html.Strong(f"{sym_a} / {sym_b}")),
                        html.Td(f"{ssd:.6f}"),
                        html.Td(f"{corr:.4f}"),
                        html.Td(str(num_trades), className="text-center"),
                        html.Td(f"{win_rate:.1f}%", className="text-center"),
                        html.Td(
                            html.Span(format_currency(total_pnl), className=pnl_color),
                            className="text-end",
                        ),
                        html.Td(
                            html.Span(format_currency(avg_pnl), className=avg_pnl_color),
                            className="text-end",
                        ),
                        html.Td(
                            dcc.Link(
                                dbc.Button(
                                    "Inspect",
                                    color="primary",
                                    size="sm",
                                    className="py-0",
                                ),
                                href=f"/pairs?pair={pair_key}",
                            ),
                            className="text-center",
                        ),
                    ],
                    className="align-middle",
                )
            )

        return dbc.Table(
            [
                html.Thead(
                    html.Tr([
                        html.Th("#", style={"width": "40px"}),
                        html.Th("Pair"),
                        html.Th("SSD"),
                        html.Th("Correlation"),
                        html.Th("Trades", className="text-center"),
                        html.Th("Win Rate", className="text-center"),
                        html.Th("Total P&L", className="text-end"),
                        html.Th("Avg P&L", className="text-end"),
                        html.Th("Action", className="text-center", style={"width": "80px"}),
                    ])
                ),
                html.Tbody(rows),
            ],
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
        )
