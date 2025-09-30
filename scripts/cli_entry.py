#!/usr/bin/env python3
"""
Entry point for the Transformer Calculator CLI
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

def main():
    """Main entry point for the CLI"""
    from cli_calculator import main as cli_main
    cli_main()

if __name__ == "__main__":
    main()

