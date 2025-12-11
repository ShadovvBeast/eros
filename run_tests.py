#!/usr/bin/env python3
"""
E.R.O.S Test Runner with Coverage Reporting

This script provides convenient test execution with coverage measurement.
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
    parser = argparse.ArgumentParser(description="E.R.O.S Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--no-cov", action="store_true", help="Run tests without coverage")
    parser.add_argument("--html", action="store_true", help="Generate HTML coverage report")
    parser.add_argument("--fast", action="store_true", help="Run tests with minimal output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("🎯 E.R.O.S Test Suite Runner")
    print("=" * 50)
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test path selection
    if args.unit:
        cmd.append("tests/unit")
        print("📋 Running unit tests only")
    elif args.integration:
        cmd.append("tests/integration")
        print("🔗 Running integration tests only")
    else:
        cmd.append("tests")
        print("🧪 Running all tests")
    
    # Coverage options
    if not args.no_cov:
        cmd.extend([
            "--cov=src",
            "--cov-report=term-missing"
        ])
        
        if args.html:
            cmd.append("--cov-report=html:htmlcov")
            print("📊 HTML coverage report will be generated")
    
    # Output options
    if args.fast:
        cmd.extend(["-q", "--tb=line"])
    elif args.verbose:
        cmd.extend(["-v", "-s"])
    
    # Run tests
    success = run_command(cmd, "Test execution")
    
    if success and not args.no_cov:
        print("\n📈 Coverage Summary:")
        coverage_cmd = ["python", "-m", "coverage", "report", "--show-missing"]
        run_command(coverage_cmd, "Coverage report generation")
        
        if args.html:
            print(f"\n🌐 HTML coverage report available at: {Path('htmlcov/index.html').absolute()}")
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests completed successfully!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())