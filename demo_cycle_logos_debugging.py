#!/usr/bin/env python3
"""
Demo: Cycle Logos Debugging Feature

This demo shows the new "Cycle Logos Debugging" feature that requires user review
of every prompt sent to Gemini before it's actually transmitted.

Key Features Demonstrated:
1. ☑️ Cycle Logos Debugging checkbox in the dashboard
2. 🔍 Prompt interception and review dialog
3. 🟢/🔴 User approval/rejection of prompts
4. 📋 Full context display for debugging
5. 🤖 Enhanced LLM-based intention generation when debugging is enabled

Usage:
1. Run this demo to see console-based prompt review
2. Or run: python main.py gui and check "Cycle Logos Debugging" in Log & Audit tab
"""

import sys
import os
import time
import threading
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.prompt_debugger import prompt_debugger, intercept_gemini_prompt
from src.core.config import AgentConfig
from src.logos.logos_layer import LogosLayer
from src.pathos.pathos_layer import PathosLayer
from src.memory.memory_system import MemorySystem
from src.ethos.ethos_framework import EthosFramework
from src.tools.tool_layer import ToolLayer
from src.monitoring.session_manager import SessionManager, SessionConfig


class InteractivePromptReviewer:
    """Interactive prompt reviewer for console-based demo."""
    
    def __init__(self):
        self.review_count = 0
        self.approved_count = 0
        self.rejected_count = 0
    
    def review_prompt(self, prompt_data):
        """Interactive prompt review with full context display."""
        self.review_count += 1
        
        print("\n" + "🔍" + "="*78 + "🔍")
        print(f"           CYCLE LOGOS DEBUGGING - PROMPT REVIEW #{self.review_count}")
        print("🔍" + "="*78 + "🔍")
        
        # Display context
        print(f"📊 Context Information:")
        print(f"   • Cycle: {prompt_data.get('cycle', 'Unknown')}")
        print(f"   • Category: {prompt_data.get('semantic_category', 'Unknown')}")
        print(f"   • Service: {prompt_data.get('service', 'Unknown')} ({prompt_data.get('model', 'Unknown')})")
        print(f"   • Priority: {prompt_data.get('priority', 'Unknown')}")
        print(f"   • Timestamp: {prompt_data.get('timestamp', 'Unknown')}")
        
        # Display intention
        intention = prompt_data.get('intention', 'Unknown')
        print(f"\n💭 Current Intention:")
        print(f"   {intention}")
        
        # Display memory context
        memory_context = prompt_data.get('memory_context', 'Unknown')
        print(f"\n🧠 Memory Context:")
        print(f"   {memory_context}")
        
        # Display tool candidates
        tool_candidates = prompt_data.get('tool_candidates', [])
        if tool_candidates:
            print(f"\n🔧 Tool Candidates:")
            for i, tool in enumerate(tool_candidates[:3], 1):
                print(f"   {i}. {tool}")
        
        # Display the actual prompt
        prompt_text = prompt_data.get('prompt_text', 'No prompt text available')
        print(f"\n📝 Prompt to be sent to Gemini:")
        print("┌" + "─" * 76 + "┐")
        
        # Word wrap the prompt for better display
        words = prompt_text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= 74:
                current_line += (" " if current_line else "") + word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        for line in lines[:10]:  # Show first 10 lines
            print(f"│ {line:<74} │")
        
        if len(lines) > 10:
            print(f"│ ... ({len(lines) - 10} more lines) {'':<50} │")
        
        print("└" + "─" * 76 + "┘")
        
        # Show statistics
        print(f"\n📈 Review Statistics:")
        print(f"   • Total Reviews: {self.review_count}")
        print(f"   • Approved: {self.approved_count}")
        print(f"   • Rejected: {self.rejected_count}")
        
        # Get user decision
        print(f"\n🤔 What would you like to do?")
        print(f"   [a] Approve - Send this prompt to Gemini")
        print(f"   [r] Reject - Block this prompt")
        print(f"   [v] View full prompt text")
        print(f"   [c] View full context")
        print(f"   [auto] Auto-approve remaining prompts")
        print(f"   [stop] Stop the session")
        
        while True:
            try:
                choice = input("\n👉 Your choice: ").lower().strip()
                
                if choice in ['a', 'approve']:
                    self.approved_count += 1
                    print("✅ APPROVED - Prompt will be sent to Gemini")
                    return True
                
                elif choice in ['r', 'reject']:
                    self.rejected_count += 1
                    print("❌ REJECTED - Prompt will NOT be sent")
                    return False
                
                elif choice in ['v', 'view']:
                    print(f"\n📄 Full Prompt Text:")
                    print("─" * 80)
                    print(prompt_text)
                    print("─" * 80)
                    continue
                
                elif choice in ['c', 'context']:
                    print(f"\n📋 Full Context:")
                    print("─" * 80)
                    for key, value in prompt_data.items():
                        if key != 'prompt_text':
                            print(f"{key}: {value}")
                    print("─" * 80)
                    continue
                
                elif choice in ['auto']:
                    self.approved_count += 1
                    print("🤖 AUTO-APPROVED - Will auto-approve remaining prompts")
                    # Switch to auto-approve mode
                    prompt_debugger.set_review_callback(lambda pd: True)
                    return True
                
                elif choice in ['stop']:
                    print("🛑 STOPPING - Session will be terminated")
                    # This will cause the prompt to be rejected and session to stop
                    return False
                
                else:
                    print("❓ Invalid choice. Please enter 'a', 'r', 'v', 'c', 'auto', or 'stop'")
                    continue
                    
            except (KeyboardInterrupt, EOFError):
                print("\n🛑 INTERRUPTED - Rejecting prompt")
                self.rejected_count += 1
                return False


def run_interactive_demo():
    """Run the interactive demo with console-based prompt review."""
    
    print("🔍 CYCLE LOGOS DEBUGGING - INTERACTIVE DEMO")
    print("="*60)
    print("This demo shows how the new debugging feature works.")
    print("You'll be asked to review every prompt before it's sent to Gemini.")
    print("="*60)
    
    # Create the interactive reviewer
    reviewer = InteractivePromptReviewer()
    
    # Set up prompt debugging
    prompt_debugger.set_review_callback(reviewer.review_prompt)
    prompt_debugger.enable_debugging()
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Create agent configuration from environment
    config = AgentConfig.from_env()
    
    # Ensure API key is loaded
    if not config.logos.gemini_api_key:
        print("⚠️  GEMINI_API_KEY not found in .env file")
        print("   Add GEMINI_API_KEY=your-key-here to .env file")
        config.logos.gemini_api_key = "demo_key"  # Fallback for demo
    
    # Create session configuration
    session_config = SessionConfig()
    session_config.duration_minutes = 2.0  # Short demo
    session_config.max_cycles = 5  # Just a few cycles
    
    # Create session manager
    session_manager = SessionManager()
    session_manager.configure_session(session_config)
    
    # Enable prompt debugging on the session
    session_manager.set_prompt_debugging(True)
    
    print(f"\n🚀 Starting demo session...")
    print(f"Duration: {session_config.duration_minutes} minutes")
    print(f"Max cycles: {session_config.max_cycles}")
    print(f"Prompt debugging: {'✅ ENABLED' if session_manager.is_prompt_debugging_enabled() else '❌ DISABLED'}")
    
    # Create and initialize agent
    agent = AutonomousAgent(config)
    
    # Create layers
    logos = LogosLayer(config.logos)
    pathos = PathosLayer(config.pathos)
    memory = MemorySystem(config.memory)
    ethos = EthosFramework(config.ethos)
    tools = ToolLayer(config.tools, ethos)
    
    agent.initialize_layers(logos, pathos, memory, ethos, tools)
    
    print(f"\n🤖 Agent initialized. Starting cycles...")
    print(f"Each cycle will generate a prompt that requires your review.")
    
    try:
        # Run a few cycles manually to demonstrate
        for cycle in range(session_config.max_cycles):
            print(f"\n🔄 Starting Cycle {cycle + 1}/{session_config.max_cycles}")
            
            try:
                result = agent.run_cycle()
                
                print(f"✅ Cycle {cycle + 1} completed:")
                print(f"   • Intention: {result['intention'][:80]}...")
                print(f"   • Duration: {result['duration']:.3f}s")
                print(f"   • Tool used: {result.get('tool_used', 'None')}")
                
                # Small delay between cycles
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Cycle {cycle + 1} failed: {e}")
                break
    
    except KeyboardInterrupt:
        print(f"\n🛑 Demo interrupted by user")
    
    finally:
        # Clean up
        prompt_debugger.disable_debugging()
        
        print(f"\n📊 Demo Summary:")
        print(f"   • Total prompt reviews: {reviewer.review_count}")
        print(f"   • Approved: {reviewer.approved_count}")
        print(f"   • Rejected: {reviewer.rejected_count}")
        
        if reviewer.review_count > 0:
            approval_rate = (reviewer.approved_count / reviewer.review_count) * 100
            print(f"   • Approval rate: {approval_rate:.1f}%")
        
        print(f"\n🎉 Demo completed!")


def show_dashboard_instructions():
    """Show instructions for using the feature in the dashboard."""
    
    print("\n📋 DASHBOARD USAGE INSTRUCTIONS")
    print("="*50)
    print("To use Cycle Logos Debugging in the full dashboard:")
    print()
    print("1. 🚀 Start the dashboard:")
    print("   python main.py gui")
    print()
    print("2. 📋 Navigate to the 'Log & Audit' tab")
    print()
    print("3. ☑️  Check the 'Cycle Logos Debugging' checkbox")
    print("   (It's next to 'Detailed Cycle Logging')")
    print()
    print("4. ▶️  Start an agent session from any tab")
    print()
    print("5. 🔍 Every Gemini prompt will now show a review dialog:")
    print("   • See the full prompt text")
    print("   • View context and intention")
    print("   • Approve or reject each prompt")
    print("   • View full context if needed")
    print()
    print("6. 📊 Monitor the logs to see approval/rejection messages")
    print()
    print("🎯 This is perfect for:")
    print("   • Understanding what prompts your agent generates")
    print("   • Debugging prompt quality and content")
    print("   • Learning how the agent reasons")
    print("   • Ensuring prompts meet your standards")
    print("   • Educational purposes and transparency")


def main():
    """Main demo function."""
    
    print("🔍 CYCLE LOGOS DEBUGGING FEATURE DEMO")
    print("="*50)
    print()
    print("This feature adds a new debugging option that intercepts")
    print("every prompt sent to Gemini and requires user review.")
    print()
    print("Choose an option:")
    print("1. Run interactive console demo")
    print("2. Show dashboard usage instructions")
    print("3. Run quick test")
    print()
    
    try:
        choice = input("Your choice (1/2/3): ").strip()
        
        if choice == "1":
            run_interactive_demo()
        elif choice == "2":
            show_dashboard_instructions()
        elif choice == "3":
            # Quick test
            print("\n🧪 Running quick test...")
            from test_cycle_logos_debugging import test_basic_prompt_interception
            
            # Set up auto-approve for quick test
            prompt_debugger.set_review_callback(lambda pd: True)
            result = test_basic_prompt_interception()
            print(f"Quick test result: {'✅ PASSED' if result else '❌ FAILED'}")
        else:
            print("Invalid choice. Please run the script again.")
    
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()