#!/usr/bin/env python3
"""
System Readiness Check for Autonomous Agent

Comprehensive verification that all systems are ready for true autonomous operation
with no simulation or mock code.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from src.monitoring.session_manager import SessionManager, SessionConfig
from src.core.agent import AutonomousAgent
from src.core.config import AgentConfig

def check_system_readiness():
    """Comprehensive system readiness check"""
    print("🎯 E.R.O.S SYSTEM READINESS CHECK")
    print("=" * 60)
    
    checks_passed = 0
    total_checks = 8
    
    # Check 1: Import all core modules
    print("1️⃣ Checking core module imports...")
    try:
        from src.logos.logos_layer import LogosLayer
        from src.pathos.pathos_layer import PathosLayer
        from src.memory.memory_system import ConcreteMemorySystem
        from src.ethos.ethos_framework import ConcreteEthosFramework
        from src.tools.tool_layer import ToolLayer
        from src.monitoring.hardware_monitor import get_hardware_monitor
        from src.dashboard.core import create_interactive_dashboard
        print("   ✅ All core modules imported successfully")
        checks_passed += 1
    except Exception as e:
        print(f"   ❌ Module import failed: {e}")
    
    # Check 2: Session Manager initialization
    print("\n2️⃣ Checking Session Manager...")
    try:
        session_manager = SessionManager()
        config = SessionConfig()
        config.duration_minutes = 1.0  # Short test
        session_manager.configure_session(config)
        print("   ✅ Session Manager configured successfully")
        checks_passed += 1
    except Exception as e:
        print(f"   ❌ Session Manager failed: {e}")
    
    # Check 3: Agent initialization with all layers
    print("\n3️⃣ Checking Agent layer initialization...")
    try:
        agent_config = AgentConfig()
        agent = AutonomousAgent(agent_config)
        
        # Initialize all layers
        from src.logos.logos_layer import LogosLayer
        from src.pathos.pathos_layer import PathosLayer
        from src.memory.memory_system import ConcreteMemorySystem
        from src.ethos.ethos_framework import ConcreteEthosFramework
        from src.tools.tool_layer import ToolLayer
        
        logos = LogosLayer(agent_config.logos)
        pathos = PathosLayer(agent_config.pathos)
        memory = ConcreteMemorySystem(agent_config.memory)
        ethos = ConcreteEthosFramework(agent_config.ethos, pathos)
        tools = ToolLayer(agent_config.tools, ethos)
        
        agent.initialize_layers(logos, pathos, memory, ethos, tools)
        print("   ✅ All agent layers initialized successfully")
        checks_passed += 1
    except Exception as e:
        print(f"   ❌ Agent initialization failed: {e}")
    
    # Check 4: Tool system verification
    print("\n4️⃣ Checking tool system...")
    try:
        available_tools = agent.tools.get_available_tools()
        print(f"   📋 Available tools: {len(available_tools)}")
        
        # Check unrestricted mode
        unrestricted_mode = getattr(agent.ethos.config, 'unrestricted_mode', False)
        print(f"   🔓 Unrestricted mode: {unrestricted_mode}")
        
        # Verify no simulation tools
        simulation_found = False
        for tool_name in available_tools:
            description = agent.tools.get_tool_description(tool_name)
            if description and ('simulated' in description.lower() or 'mock' in description.lower()):
                simulation_found = True
                print(f"   ⚠️  Simulation tool found: {tool_name}")
        
        if not simulation_found and unrestricted_mode:
            print("   ✅ No simulation tools detected - all real implementations")
            print("   ✅ Unrestricted mode enabled - no tool category restrictions")
            checks_passed += 1
        elif not simulation_found:
            print("   ✅ No simulation tools detected - all real implementations")
            print("   ⚠️  Unrestricted mode not enabled")
            checks_passed += 1
        else:
            print("   ❌ Simulation tools still present")
    except Exception as e:
        print(f"   ❌ Tool system check failed: {e}")
    
    # Check 5: Hardware monitoring
    print("\n5️⃣ Checking hardware monitoring...")
    try:
        hardware_monitor = get_hardware_monitor()
        hardware_monitor.start_monitoring(1.0)
        time.sleep(2)  # Let it collect some data
        summary = hardware_monitor.get_current_summary()
        hardware_monitor.stop_monitoring()
        
        if summary and ('cpu_percent' in summary or 'cpu' in summary):
            cpu_val = summary.get('cpu_percent', summary.get('cpu', 'N/A'))
            print(f"   ✅ Hardware monitoring active - CPU: {cpu_val}")
            checks_passed += 1
        elif summary:
            print(f"   ✅ Hardware monitoring active - Data: {list(summary.keys())}")
            checks_passed += 1
        else:
            print("   ❌ Hardware monitoring not working properly")
    except Exception as e:
        print(f"   ❌ Hardware monitoring failed: {e}")
    
    # Check 6: Memory system
    print("\n6️⃣ Checking memory system...")
    try:
        # Test memory operations
        from src.core.models import MemoryTrace, SemanticVector
        import numpy as np
        
        # Create proper semantic vector
        semantic_vec = SemanticVector(
            embedding=np.random.normal(0, 0.1, 512),
            intention_text="Test memory trace",
            semantic_category="test"
        )
        
        # Create proper memory trace
        test_trace = MemoryTrace(
            affect_state=np.random.normal(0, 0.1, 128),
            semantic_vector=semantic_vec,
            reward=1.0,
            timestamp=int(time.time()),
            salience=0.8
        )
        
        agent.memory.store_trace(test_trace)
        retrieved = agent.memory.get_recent_traces(1)
        
        if retrieved and len(retrieved) > 0:
            print("   ✅ Memory system operational")
            checks_passed += 1
        else:
            print("   ❌ Memory system not storing/retrieving properly")
    except Exception as e:
        print(f"   ❌ Memory system failed: {e}")
    
    # Check 7: Pathos system
    print("\n7️⃣ Checking Pathos emotional system...")
    try:
        # Test pathos state
        current_state = agent.pathos.get_current_state()
        if current_state is not None:
            if hasattr(current_state, 'vector'):
                state_vector = current_state.vector
                print(f"   ✅ Pathos system active - State dimension: {len(state_vector)}")
                checks_passed += 1
            elif hasattr(current_state, '__len__'):
                print(f"   ✅ Pathos system active - State dimension: {len(current_state)}")
                checks_passed += 1
            else:
                print(f"   ✅ Pathos system active - State type: {type(current_state)}")
                checks_passed += 1
        else:
            print("   ❌ Pathos system not generating states")
    except Exception as e:
        print(f"   ❌ Pathos system failed: {e}")
    
    # Check 8: GUI system
    print("\n8️⃣ Checking GUI system...")
    try:
        # Test dashboard creation (don't run it)
        from src.core.logging_config import InstrumentationCollector
        from src.dashboard.core import create_interactive_dashboard
        test_collector = InstrumentationCollector()
        dashboard = create_interactive_dashboard(test_collector, update_interval=1.0)
        if dashboard:
            print("   ✅ GUI dashboard ready for launch")
            checks_passed += 1
        else:
            print("   ❌ GUI dashboard creation failed")
    except Exception as e:
        print(f"   ❌ GUI system failed: {e}")
    
    # Final assessment
    print("\n" + "=" * 60)
    print("🎯 SYSTEM READINESS ASSESSMENT")
    print("=" * 60)
    
    success_rate = (checks_passed / total_checks) * 100
    print(f"📊 Checks Passed: {checks_passed}/{total_checks} ({success_rate:.1f}%)")
    
    if checks_passed == total_checks:
        print("🎉 SYSTEM FULLY READY FOR AUTONOMOUS OPERATION!")
        print("✅ All systems operational")
        print("✅ No simulation code detected")
        print("✅ Real implementations verified")
        print("🚀 Ready to launch autonomous agent!")
        
        print("\n💡 To start the system:")
        print("   python main.py gui")
        
    elif checks_passed >= total_checks * 0.8:
        print("⚠️  SYSTEM MOSTLY READY - Minor issues detected")
        print("🔧 Review failed checks above")
        
    else:
        print("❌ SYSTEM NOT READY - Critical issues detected")
        print("🛠️  Fix failed checks before autonomous operation")
    
    return checks_passed == total_checks

def main():
    """Main entry point for system readiness check."""
    check_system_readiness()

if __name__ == "__main__":
    main()