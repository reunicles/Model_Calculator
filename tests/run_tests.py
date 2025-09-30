#!/usr/bin/env python3

# Fix imports for project structure
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

"""
Test Runner for Transformer Calculator

This script runs all unit tests and provides comprehensive test reporting.
"""

import sys
import unittest
import time
from pathlib import Path

def run_all_tests():
    """Run all unit tests and return results"""
    print("🧪 Transformer Calculator Test Suite")
    print("=" * 50)
    
    # Import test modules
    try:
        from test_transformer_calculator import run_tests
        print("✅ Test modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import test modules: {e}")
        return False
    
    # Run tests
    start_time = time.time()
    success = run_tests()
    end_time = time.time()
    
    # Print results
    print(f"\n📊 Test Results:")
    print(f"  Execution time: {end_time - start_time:.2f} seconds")
    print(f"  Status: {'✅ PASSED' if success else '❌ FAILED'}")
    
    return success

def run_specific_test(test_name: str):
    """Run a specific test"""
    print(f"🧪 Running specific test: {test_name}")
    print("=" * 50)
    
    # This would be implemented to run specific tests
    # For now, just run all tests
    return run_all_tests()

def main():
    """Main test runner function"""
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        success = run_specific_test(test_name)
    else:
        success = run_all_tests()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
