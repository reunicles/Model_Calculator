#!/usr/bin/env python3

# Fix imports for project structure
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

"""
Basic Type Checking Script for Transformer Calculator

This script performs basic type checking and validation for the transformer calculator
without requiring mypy installation.
"""

import ast
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

def check_file_types(file_path: str) -> List[str]:
    """Check a Python file for basic type issues"""
    issues = []
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content)
        
        # Check for common type issues
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for missing return type annotations
                if node.returns is None and not node.name.startswith('_'):
                    issues.append(f"Function '{node.name}' missing return type annotation")
                
                # Check for missing parameter type annotations
                for arg in node.args.args:
                    if arg.annotation is None and not arg.arg.startswith('_'):
                        issues.append(f"Parameter '{arg.arg}' in '{node.name}' missing type annotation")
            
            elif isinstance(node, ast.ClassDef):
                # Check for missing class docstrings
                if not any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) 
                          for n in node.body):
                    issues.append(f"Class '{node.name}' missing docstring")
    
    except Exception as e:
        issues.append(f"Error parsing {file_path}: {e}")
    
    return issues

def main():
    """Main type checking function"""
    print("🔍 Basic Type Checking for Transformer Calculator")
    print("=" * 50)
    
    # Files to check
    files_to_check = [
        "transformer_calculator.py",
        "datatypes.py", 
        "context.py",
        "table_formatter.py",
        "validation_common.py",
        "validation_ranges.py"
    ]
    
    all_issues = []
    
    for file_path in files_to_check:
        if Path(file_path).exists():
            print(f"\n📁 Checking {file_path}...")
            issues = check_file_types(file_path)
            if issues:
                all_issues.extend([f"{file_path}: {issue}" for issue in issues])
                for issue in issues:
                    print(f"  ⚠️  {issue}")
            else:
                print(f"  ✅ No type issues found")
        else:
            print(f"  ❌ File not found: {file_path}")
    
    print(f"\n📊 Summary:")
    print(f"  Total issues: {len(all_issues)}")
    
    if all_issues:
        print(f"\n🔧 Recommendations:")
        print(f"  1. Add return type annotations to all public functions")
        print(f"  2. Add parameter type annotations to all public functions")
        print(f"  3. Add docstrings to all classes and public methods")
        print(f"  4. Use Optional[T] for parameters that can be None")
        print(f"  5. Use Union[T, U] for parameters that can be multiple types")
    
    return len(all_issues)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
