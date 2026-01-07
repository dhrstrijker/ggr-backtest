"""GGR Pairs Trading Dashboard - Multi-page Dash application."""

from .app import create_app
from .data_store import DataStore

__all__ = ["create_app", "DataStore"]
