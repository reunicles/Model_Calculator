#!/usr/bin/env python3
"""
Start the Transformer Calculator Web Interface
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Change to src directory to run the web interface
os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

def main():
    """Main entry point for the web interface"""
    # Import and run the web interface
    import web_interface_enhanced
    web_interface_enhanced.main()

if __name__ == "__main__":
    main()