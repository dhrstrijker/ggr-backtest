"""Reusable metric card component."""

from dash import html
import dash_bootstrap_components as dbc


def create_metric_card(title: str, value_id: str, color: str = "primary") -> dbc.Card:
    """
    Create a metric display card.

    Args:
        title: Card title/label
        value_id: ID for the value element (for callbacks)
        color: Bootstrap color class (primary, success, danger, etc.)

    Returns:
        dbc.Card component
    """
    return dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H6(title, className="card-subtitle mb-2 text-muted"),
                    html.H3(id=value_id, className="card-title mb-0"),
                ],
                className="text-center py-3",
            )
        ],
        className=f"border-{color}",
    )


def format_currency(value: float, decimals: int = 2) -> str:
    """Format a value as currency."""
    if value >= 0:
        return f"${value:,.{decimals}f}"
    return f"-${abs(value):,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a value as percentage."""
    return f"{value:+.{decimals}f}%"


def format_with_color(value: float, format_func=format_percentage) -> html.Span:
    """Format a value with color based on sign."""
    formatted = format_func(value)
    color = "success" if value >= 0 else "danger"
    return html.Span(formatted, className=f"text-{color}")
