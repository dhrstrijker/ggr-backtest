#!/usr/bin/env python3
"""
GGR Pairs Trading Dashboard - Main Entry Point

A multi-page Dash application for visualizing and analyzing
GGR Distance Method pairs trading backtest results.

Usage:
    python dashboard.py [--port PORT] [--debug] [--config CONFIG_FILE]

Examples:
    python dashboard.py                    # Run on default port 8050
    python dashboard.py --port 8080        # Run on custom port
    python dashboard.py --debug            # Run in debug mode
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dashboard.app import create_app
from dashboard.data_store import DataStore, DEFAULT_CONFIG


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GGR Pairs Trading Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python dashboard.py                    # Run on default port 8050
    python dashboard.py --port 8080        # Run on custom port
    python dashboard.py --debug            # Run in debug mode
    python dashboard.py --config my_config.json  # Use custom config
        """,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run the dashboard on (default: 8050)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with auto-reload",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file (optional)",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )

    return parser.parse_args()


def load_config(config_path: str | None) -> dict:
    """Load configuration from file or use defaults."""
    if config_path is None:
        print("Using default configuration")
        return DEFAULT_CONFIG.copy()

    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Warning: Config file {config_path} not found, using defaults")
        return DEFAULT_CONFIG.copy()

    print(f"Loading configuration from {config_path}")
    with open(config_file) as f:
        config = json.load(f)

    # Merge with defaults for any missing keys
    merged = DEFAULT_CONFIG.copy()
    merged.update(config)
    return merged


def main():
    """Main entry point for the dashboard."""
    args = parse_args()

    print("=" * 60)
    print("GGR Pairs Trading Dashboard")
    print("=" * 60)
    print()

    # Load configuration
    config = load_config(args.config)

    # Initialize data store and load/compute data
    print("Initializing data store...")
    data_store = DataStore()
    data_store.load_or_compute(config)

    # Create Dash app
    print("\nCreating dashboard application...")
    app = create_app(data_store)

    # Print startup info
    print()
    print("=" * 60)
    print(f"Dashboard ready!")
    print(f"Open your browser to: http://{args.host}:{args.port}")
    print("=" * 60)
    print()

    if args.debug:
        print("Running in DEBUG mode - changes will auto-reload")
        print()

    # Run the server
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
