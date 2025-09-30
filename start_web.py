#!/usr/bin/env python3
"""
Transformer Calculator Web Interface Entry Point
"""

import sys
import os

# Add the scripts directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

if __name__ == "__main__":
    import start_web as web_script
    web_script.main()