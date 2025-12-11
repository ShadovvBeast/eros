#!/usr/bin/env python3
"""
Safe E.R.O.S Test Runner - No Hanging Tests

This script runs only the safe unit tests that won't hang or launch GUIs.
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"❌ {description} failed")
        if result.stderr:
            print("Error output:")
            print(result.stderr)
        if result.stdout:
            print("Standard output:")
            print(result.stdout)
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Safe E.R.O.S Test Runner")
    parser.add_argument("--cov", action="store_true", help="Include coverage reporting")
    parser.add_argument("--html", action="store_true", help="Generate HTML coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("🎯 E.R.O.S Safe Test Suite Runner")
    print("=" * 50)
    print("🛡️ Running only safe tests that won't hang or launch GUIs")
    
    # Safe test directories (no GUI or real process tests)
    safe_test_paths = [
        "tests/unit/logos/",
        "tests/unit/pathos/", 
        "tests/unit/memory/",
        "tests/unit/ethos/",
        "tests/unit/agents/",
        "tests/unit/visualization/",
        "tests/unit/core/",
        "tests/unit/tools/",
        "tests/unit/monitoring/",
        "tests/unit/dashboard/"  # Now safe with mocked tests
    ]
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"] + safe_test_paths
    
    # Add coverage if requested
    if args.cov:
        cmd.extend([
            "--cov=src",
            "--cov-report=term-missing"
        ])
        
        if args.html:
            cmd.append("--cov-report=html:htmlcov")
            print("📊 HTML coverage report will be generated")
    
    # Output options
    if args.verbose:
        cmd.extend(["-v"])
    else:
        cmd.extend(["-q"])
    
    # Add timeout to prevent hanging
    cmd.extend(["--timeout=30"])  # 30 second timeout per test
    
    print(f"\n📋 Running tests from {len(safe_test_paths)} safe directories")
    
    # Run tests
    success = run_command(cmd, "Safe test execution")
    
    if success and args.cov:
        print("\n📈 Coverage Summary:")
        coverage_cmd = ["python", "-m", "coverage", "report", "--show-missing"]
        run_command(coverage_cmd, "Coverage report generation")
        
        if args.html:
            print(f"\n🌐 HTML coverage report available at: {Path('htmlcov/index.html').absolute()}")
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All safe tests completed successfully!")
        print("🛡️ No hanging or GUI tests were run")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())