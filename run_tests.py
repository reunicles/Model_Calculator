#!/usr/bin/env python3
"""
Transformer Calculator Test Runner Entry Point
"""

import sys
import os

# Add the scripts directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

if __name__ == "__main__":
    from run_tests import main
    main()