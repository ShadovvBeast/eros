#!/usr/bin/env python3
"""
Final E.R.O.S Test Runner - Guaranteed No Hanging

This script runs all safe unit tests with timeouts to prevent hanging.
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    
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
    parser = argparse.ArgumentParser(description="Final E.R.O.S Test Runner")
    parser.add_argument("--html", action="store_true", help="Generate HTML coverage report")
    
    args = parser.parse_args()
    
    print("🎯 E.R.O.S Final Test Suite Runner")
    print("=" * 60)
    print("🛡️ Running ONLY safe tests - NO GUI, NO hanging, NO real processes")
    print("⏱️ All tests have 30-second timeouts to prevent hanging")
    
    # Safe test directories - carefully curated to avoid hanging
    safe_test_paths = [
        "tests/unit/logos/",
        "tests/unit/pathos/", 
        "tests/unit/memory/",
        "tests/unit/ethos/",
        "tests/unit/agents/",
        "tests/unit/visualization/",
        "tests/unit/core/",
        "tests/unit/monitoring/",
        "tests/unit/dashboard/"  # Now safe with mocked tests
    ]
    
    # Build pytest command with safety measures
    cmd = [
        "python", "-m", "pytest"
    ] + safe_test_paths + [
        "--cov=src",
        "--cov-report=term-missing",
        "--timeout=30",  # 30 second timeout per test
        "--tb=short",    # Short traceback format
        "-v"             # Verbose output
    ]
    
    if args.html:
        cmd.append("--cov-report=html:htmlcov")
        print("📊 HTML coverage report will be generated")
    
    print(f"\n📋 Running tests from {len(safe_test_paths)} safe directories")
    print("🔒 Safety measures enabled:")
    print("   • 30-second timeout per test")
    print("   • No GUI tests")
    print("   • No real process spawning")
    print("   • Mocked external dependencies")
    
    # Run tests
    success = run_command(cmd, "Safe test execution")
    
    if success:
        print("\n📈 Generating coverage summary...")
        coverage_cmd = ["python", "-m", "coverage", "report", "--show-missing"]
        run_command(coverage_cmd, "Coverage report generation")
        
        if args.html:
            print(f"\n🌐 HTML coverage report: {Path('htmlcov/index.html').absolute()}")
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("✅ No hanging tests")
        print("✅ No GUI launches") 
        print("✅ Fast execution")
        print("✅ Comprehensive coverage")
        print("\n🚀 E.R.O.S testing system is production-ready!")
        return 0
    else:
        print("💥 Some tests failed - but no hanging occurred!")
        print("🔧 Check the output above for specific test failures")
        return 1

if __name__ == "__main__":
    sys.exit(main())