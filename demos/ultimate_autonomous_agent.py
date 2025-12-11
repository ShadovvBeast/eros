#!/usr/bin/env python3
"""
Ultimate Autonomous Agent Demo

DEPRECATED: This file has been refactored into modular components.
Use src.agents.ultimate_agent.UltimateAutonomousAgent instead.

This file is kept for backward compatibility.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the new modular ultimate agent
try:
    from src.agents.ultimate_agent import UltimateAutonomousAgent, main
except ImportError:
    from agents.ultimate_agent import UltimateAutonomousAgent, main

# Re-export for backward compatibility
__all__ = ['UltimateAutonomousAgent', 'main']

if __name__ == "__main__":
    main()