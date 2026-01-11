#!/usr/bin/env python3
"""
GGR Pairs Trading Dashboard - Main Entry Point

A multi-page Dash application for visualizing and analyzing
GGR Distance Method pairs trading backtest results.

Usage:
    python dashboard.py [--port PORT] [--debug] [--config CONFIG_FILE]
    python dashboard.py --us-market        # Run with S&P 500 universe
    python dashboard.py --tech             # Run with tech sector
    python dashboard.py --shipping         # Run with shipping sector
    python dashboard.py --utilities        # Run with utilities sector (default)

Examples:
    python dashboard.py                    # Run on default port 8050
    python dashboard.py --port 8080        # Run on custom port
    python dashboard.py --debug            # Run in debug mode
    python dashboard.py --us-market        # Use S&P 500 universe
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dashboard.app import create_app
from dashboard.data_store import DataStore


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GGR Pairs Trading Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sector Presets:
    --us-market    S&P 500 universe (~500 stocks)
    --tech         Technology sector (~50 stocks)
    --shipping     Shipping sector (~20 stocks)
    --utilities    Utilities sector (~34 stocks, default)

Examples:
    python dashboard.py                    # Run with utilities (default)
    python dashboard.py --us-market        # Run with S&P 500
    python dashboard.py --tech --port 8080 # Run tech sector on port 8080
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
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )

    # Sector selection (mutually exclusive)
    sector_group = parser.add_mutually_exclusive_group()
    sector_group.add_argument(
        "--us-market",
        action="store_true",
        help="Use S&P 500 universe (~500 stocks)",
    )
    sector_group.add_argument(
        "--tech",
        action="store_true",
        help="Use technology sector (~50 stocks)",
    )
    sector_group.add_argument(
        "--shipping",
        action="store_true",
        help="Use shipping sector (~20 stocks)",
    )
    sector_group.add_argument(
        "--utilities",
        action="store_true",
        help="Use utilities sector (~34 stocks, default)",
    )
    sector_group.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom JSON config file",
    )

    return parser.parse_args()


# Mapping of sector flags to config file paths
SECTOR_CONFIG_MAP = {
    "us_market": "configs/sectors/us_market.json",
    "tech": "configs/sectors/tech.json",
    "shipping": "configs/sectors/shipping.json",
    "utilities": "configs/sectors/utilities.json",
}


def load_config_file(config_path: str) -> dict:
    """Load configuration from a JSON file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file) as f:
        config = json.load(f)

    return config


def resolve_sector_config(args) -> dict:
    """
    Resolve sector flag or config path to a configuration dict.

    Priority:
    1. --config path (custom config file)
    2. Sector flags (--us-market, --tech, --shipping, --utilities)
    3. Default to utilities if no flag specified
    """
    # Check for custom config path first
    if args.config:
        print(f"Loading custom configuration from {args.config}")
        return load_config_file(args.config)

    # Check sector flags
    for sector, config_path in SECTOR_CONFIG_MAP.items():
        flag_name = sector.replace("_", "-")  # us_market -> us-market
        if getattr(args, sector.replace("-", "_"), False):
            print(f"Loading {sector} sector configuration")
            return load_config_file(config_path)

    # Default to utilities
    print("Using default configuration (utilities sector)")
    return load_config_file(SECTOR_CONFIG_MAP["utilities"])


def main():
    """Main entry point for the dashboard."""
    args = parse_args()

    print("=" * 60)
    print("GGR Pairs Trading Dashboard")
    print("=" * 60)
    print()

    # Load configuration based on sector flag or custom config
    config = resolve_sector_config(args)
    sector_name = config.get("name", "Unknown")
    print(f"Sector: {sector_name} ({len(config.get('symbols', []))} symbols)")
    print()

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
