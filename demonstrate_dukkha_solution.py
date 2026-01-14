#!/usr/bin/env python3
"""
Demonstrate how the dukkha (dissatisfaction) system solves the "stuck" agent problem.

This script shows the difference between the old system (perpetually content)
and the new system with dukkha (healthy dissatisfaction that drives growth).
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def simulate_old_system():
    """Simulate the old system behavior - perpetually content."""
    print("🤖 OLD SYSTEM (Without Dukkha)")
    print("=" * 40)
    
    # Simulate the repetitive pattern from dashboard export
    actions = ['tool_finder', 'analysis_data_processor', 'tool_developer'] * 10
    rewards = [8.0, 7.8, 8.2] * 10  # Consistently high rewards
    emotional_state = "Feeling content and motivated"
    
    print("Simulating 30 cycles of repetitive behavior...")
    for i in range(0, 30, 5):
        cycle_actions = actions[i:i+5]
        cycle_rewards = rewards[i:i+5]
        avg_reward = np.mean(cycle_rewards)
        
        print(f"Cycles {i+1}-{i+5}:")
        print(f"  Actions: {cycle_actions}")
        print(f"  Avg Reward: {avg_reward:.1f}")
        print(f"  Emotional State: {emotional_state}")
        print(f"  Growth Motivation: None - system is 'satisfied'")
        print()
    
    print("❌ RESULT: Agent stuck in repetitive loop with no drive to change")
    print("   - Same actions repeated endlessly")
    print("   - High rewards mask lack of progress")
    print("   - No dissatisfaction to motivate growth")
    print()


def simulate_new_system_with_dukkha():
    """Simulate the new system with dukkha - healthy dissatisfaction."""
    print("🔥 NEW SYSTEM (With Dukkha)")
    print("=" * 40)
    
    # Import dukkha engine
    import importlib.util
    spec = importlib.util.spec_from_file_location('dukkha_engine', 'src/pathos/dukkha_engine.py')
    dukkha_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dukkha_module)
    
    engine = dukkha_module.DukkhaEngine()
    
    # Simulate the same repetitive pattern but with dukkha awareness
    actions = ['tool_finder', 'analysis_data_processor', 'tool_developer'] * 10
    rewards = [8.0, 7.8, 8.2] * 10
    
    print("Simulating 30 cycles with dukkha awareness...")
    
    cumulative_actions = []
    cumulative_rewards = []
    
    for i in range(0, 30, 5):
        cycle_actions = actions[i:i+5]
        cycle_rewards = rewards[i:i+5]
        
        # Add to cumulative history
        cumulative_actions.extend(cycle_actions)
        cumulative_rewards.extend(cycle_rewards)
        
        # Compute dukkha for this cycle
        state = np.random.normal(0, 1, 64)  # Simulated state
        dukkha_influences = engine.compute_dukkha_influence(
            state, cumulative_actions, cumulative_rewards, {}
        )
        
        total_dissatisfaction = sum(dukkha_influences.values()) / len(dukkha_influences)
        
        # Determine emotional state based on dukkha
        if total_dissatisfaction > 0.6:
            emotional_state = "Feeling restless and eager for change"
        elif total_dissatisfaction > 0.4:
            emotional_state = "Feeling mildly dissatisfied but purposeful"
        elif total_dissatisfaction > 0.2:
            emotional_state = "Feeling content but sensing room for growth"
        else:
            emotional_state = "Feeling calm and balanced"
        
        # Get growth suggestions
        suggestions = engine.suggest_growth_actions(dukkha_influences)
        top_suggestion = suggestions[0] if suggestions else "Continue current approach"
        
        print(f"Cycles {i+1}-{i+5}:")
        print(f"  Actions: {cycle_actions}")
        print(f"  Avg Reward: {np.mean(cycle_rewards):.1f}")
        print(f"  Emotional State: {emotional_state}")
        print(f"  Total Dissatisfaction: {total_dissatisfaction:.3f}")
        print(f"  Growth Motivation: {top_suggestion}")
        
        # Show specific dissatisfaction sources
        top_dukkha = sorted(dukkha_influences.items(), key=lambda x: x[1], reverse=True)[:2]
        print(f"  Top Dissatisfaction Sources:")
        for dukkha_type, level in top_dukkha:
            if level > 0.1:
                print(f"    - {dukkha_type.replace('_', ' ').title()}: {level:.3f}")
        print()
    
    print("✅ RESULT: Agent develops healthy dissatisfaction that drives growth")
    print("   - Detects repetitive patterns and becomes restless")
    print("   - Generates specific suggestions for improvement")
    print("   - Creates motivation to explore and change")
    print("   - Prevents stagnation through constructive dissatisfaction")
    print()


def show_dukkha_benefits():
    """Show the key benefits of the dukkha system."""
    print("🌟 KEY BENEFITS OF DUKKHA SYSTEM")
    print("=" * 40)
    
    benefits = [
        "🔄 Prevents Stagnation: Detects repetitive patterns and creates restlessness",
        "🎯 Goal-Directed Growth: Generates specific suggestions for improvement",
        "🤔 Existential Depth: Introduces deeper questioning about purpose",
        "🧠 Curiosity Drive: Creates tension from unresolved questions",
        "🏆 Mastery Motivation: Pressure to improve skills and capabilities",
        "🌈 Emotional Variety: Rich emotional states beyond 'content and motivated'",
        "⚡ Dynamic Adaptation: System responds to its own behavioral patterns",
        "🔍 Self-Awareness: Agent becomes aware of its own stagnation"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")
    
    print()
    print("💡 PHILOSOPHICAL INSIGHT:")
    print("   Dukkha (dissatisfaction) is not suffering - it's the healthy tension")
    print("   that drives all genuine growth and learning. Without it, systems")
    print("   become complacent and stuck in local optima.")
    print()


def main():
    """Demonstrate the dukkha solution."""
    print("🔥 DUKKHA SOLUTION: Addressing the 'Stuck' Agent Problem")
    print("=" * 60)
    print()
    
    # Show old system behavior
    simulate_old_system()
    
    # Show new system with dukkha
    simulate_new_system_with_dukkha()
    
    # Show benefits
    show_dukkha_benefits()
    
    print("🎯 CONCLUSION:")
    print("   The dukkha engine transforms the agent from a complacent system")
    print("   stuck in repetitive patterns into a dynamic, growth-oriented")
    print("   entity that actively seeks improvement and new challenges.")
    print()
    print("   This addresses the core issue observed in the dashboard exports")
    print("   where the agent was 'successful' but fundamentally stuck.")


if __name__ == "__main__":
    main()