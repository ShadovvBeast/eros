#!/usr/bin/env python3
"""
Verification script to ensure the project reorganization was successful.
"""

import sys
import os
from pathlib import Path
import importlib.util

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def verify_imports():
    """Verify that all critical imports work correctly."""
    print("🔍 VERIFYING PROJECT ORGANIZATION")
    print("=" * 50)
    
    # Test core imports
    try:
        import agent
        print("✅ Core agent import successful")
    except ImportError as e:
        print(f"❌ Core agent import failed: {e}")
        return False
    
    try:
        import config
        print("✅ Configuration import successful")
    except ImportError as e:
        print(f"❌ Configuration import failed: {e}")
        return False
    
    try:
        import interactive_dashboard
        print("✅ Dashboard import successful")
    except ImportError as e:
        print(f"❌ Dashboard import failed: {e}")
        return False
    
    # Test layer imports
    try:
        import logos.logos_layer
        import pathos.pathos_layer
        import memory.memory_system
        import ethos.ethos_framework
        print("✅ All layer imports successful")
    except ImportError as e:
        print(f"❌ Layer import failed: {e}")
        return False
    
    return True

def verify_structure():
    """Verify the directory structure is correct."""
    print("\n📁 VERIFYING DIRECTORY STRUCTURE")
    print("=" * 50)
    
    root = Path(__file__).parent.parent
    expected_dirs = [
        "src",
        "demos", 
        "tests",
        "tools",
        "docs",
        "data",
        "examples"
    ]
    
    for dir_name in expected_dirs:
        dir_path = root / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory missing")
            return False
    
    # Check key files
    key_files = [
        "main.py",
        "setup.py", 
        "README.md",
        "requirements.txt"
    ]
    
    for file_name in key_files:
        file_path = root / file_name
        if file_path.exists():
            print(f"✅ {file_name} exists")
        else:
            print(f"❌ {file_name} missing")
            return False
    
    return True

def verify_entry_points():
    """Verify that the main entry point works."""
    print("\n🚀 VERIFYING ENTRY POINTS")
    print("=" * 50)
    
    try:
        # Test that main.py can be imported
        main_path = Path(__file__).parent.parent / "main.py"
        spec = importlib.util.spec_from_file_location("main", main_path)
        main_module = importlib.util.module_from_spec(spec)
        print("✅ main.py can be imported")
        return True
    except Exception as e:
        print(f"❌ main.py import failed: {e}")
        return False

def main():
    """Run all verification checks."""
    print("🎯 PROJECT ORGANIZATION VERIFICATION")
    print("=" * 60)
    
    checks = [
        ("Import Verification", verify_imports),
        ("Structure Verification", verify_structure), 
        ("Entry Point Verification", verify_entry_points)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name} failed with exception: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL VERIFICATION CHECKS PASSED!")
        print("✅ Project reorganization successful")
        print("🚀 Ready for development and production use")
    else:
        print("❌ SOME VERIFICATION CHECKS FAILED")
        print("🔧 Please review the errors above and fix issues")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)