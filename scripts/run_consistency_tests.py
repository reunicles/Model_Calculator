#!/usr/bin/env python3
"""
Script to run CLI-Web consistency tests.
This ensures both interfaces produce identical results.
"""

import subprocess
import sys
import os

def run_consistency_tests():
    """Run all consistency tests"""
    print("🚀 Running CLI-Web Consistency Tests")
    print("=" * 60)
    
    # Change to project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_dir)
    
    # Run comprehensive consistency tests
    print("\n📊 Running comprehensive consistency tests...")
    result1 = subprocess.run([
        sys.executable, "tests/test_cli_web_consistency.py"
    ], capture_output=True, text=True)
    
    if result1.returncode != 0:
        print("❌ Comprehensive consistency tests failed!")
        print("STDOUT:", result1.stdout)
        print("STDERR:", result1.stderr)
        return False
    
    # Run DeepSeek validation tests
    print("\n📊 Running DeepSeek validation tests...")
    result2 = subprocess.run([
        sys.executable, "tests/test_deepseek_validation.py"
    ], capture_output=True, text=True)
    
    if result2.returncode != 0:
        print("❌ DeepSeek validation tests failed!")
        print("STDOUT:", result2.stdout)
        print("STDERR:", result2.stderr)
        return False
    
    print("\n✅ All consistency tests passed!")
    print("🔒 CLI and Web interface are guaranteed to be consistent!")
    return True

if __name__ == "__main__":
    success = run_consistency_tests()
    sys.exit(0 if success else 1)
