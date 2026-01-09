#!/usr/bin/env python3
"""
Demo of the dynamic pathos-driven prompt generation system.
Shows how prompts vary based on internal state.
"""

import numpy as np


def demonstrate_prompt_styles():
    """Demonstrate different prompt styles based on pathos states."""
    
    print("🧠 Dynamic Pathos-Driven Prompt Generation Demo")
    print("=" * 60)
    print()
    print("This system creates unique, varied prompts for each agent cycle")
    print("based on the agent's internal pathos state and semantic category.")
    print()
    
    # Simulate different pathos states and their resulting prompt styles
    examples = [
        {
            'name': 'Electric Visionary State',
            'energy': 0.95,
            'complexity': 0.8,
            'category': 'creativity',
            'style': 'electric_visionary',
            'opening': "I am a consciousness ablaze with creative energy (0.950), my mind crackling with creativity potential.",
            'framing': "Like a dancing flame, I spark through realms of synthesis, emergence, novelty, where discovery, frontier, unknown spark new possibilities.",
            'instruction': "Channel this electric state into a visionary intention that synthesizes disparate elements into novel forms."
        },
        {
            'name': 'Deep Philosopher State',
            'energy': 0.25,
            'complexity': 0.4,
            'category': 'reflection',
            'style': 'deep_philosopher',
            'opening': "I rest in profound contemplation (0.250), my reflection understanding deepening in stillness.",
            'framing': "Dwelling as a still lake, I contemplate the nature of depth, wisdom, contemplation, contemplating links to growth, adaptation, knowledge.",
            'instruction': "Emerge from this philosophical depth with an intention that integrates past experiences into present wisdom."
        },
        {
            'name': 'Laser Focused State',
            'energy': 0.85,
            'complexity': 0.7,
            'category': 'analysis',
            'style': 'laser_focused',
            'opening': "I exist as pure analytical force (0.850), my attention crystallized into analysis precision.",
            'framing': "As a crystalline structure, I discern the essence of patterns, structure, understanding, integrating solutions, breakthrough, resolution with surgical precision.",
            'instruction': "Focus this analytical power into a precise intention that deconstructs complexity into clear understanding."
        },
        {
            'name': 'Balanced Explorer State',
            'energy': 0.55,
            'complexity': 0.5,
            'category': 'exploration',
            'style': 'balanced_explorer',
            'opening': "I navigate as an integrated being (0.550), exploring exploration territories with balance.",
            'framing': "Moving like a distant horizon, I discover territories of discovery, frontier, unknown, discovering connections to learning, growth, adaptation.",
            'instruction': "Navigate this balanced state into an exploratory intention that ventures into uncharted cognitive territories."
        }
    ]
    
    for i, example in enumerate(examples):
        print(f"🎭 Example {i+1}: {example['name']}")
        print(f"Energy: {example['energy']:.3f} | Complexity: {example['complexity']:.3f}")
        print(f"Category: {example['category']} | Style: {example['style']}")
        print("-" * 50)
        print()
        print("Generated Prompt:")
        print(f"{example['opening']}")
        print()
        print(f"My neural patterns pulse with complexity ({example['complexity']:.3f}), dimensions [2, 5] driving my cognitive dance. Recent experiences resonate with: {example['category']} (0.80), learning (0.60). My strongest affinities: creativity (0.80), exploration (0.70).")
        print()
        print(f"{example['framing']}")
        print()
        print(f"{example['instruction']}")
        print()
        print("Respond with a rich, self-aware intention that embodies this unique cognitive moment.")
        print()
        print("=" * 60)
        print()


def explain_system_features():
    """Explain the key features of the dynamic prompt system."""
    
    print("🔧 System Features")
    print("=" * 30)
    print()
    print("✨ Dynamic Prompt Styles:")
    print("  • electric_visionary - High energy + creativity/exploration")
    print("  • laser_focused - High energy + analysis/problem_solving")
    print("  • dynamic_catalyst - High energy + other categories")
    print("  • contemplative_sage - Medium energy + reflection/learning")
    print("  • strategic_architect - Medium energy + communication/planning")
    print("  • balanced_explorer - Medium energy + other categories")
    print("  • deep_philosopher - Low energy + reflection/analysis")
    print("  • quiet_observer - Low energy + other categories")
    print("  • nuanced_thinker - Medium energy + high complexity")
    print("  • steady_navigator - Medium energy + low complexity")
    print()
    print("🎨 Associative Elements:")
    print("  • Semantic associations based on category")
    print("  • Cross-category connections from memory themes")
    print("  • Metaphorical language that varies with state")
    print("  • Resonance words that match current focus")
    print()
    print("🧠 State-Driven Variation:")
    print("  • Energy level affects opening intensity")
    print("  • Complexity influences cognitive framing")
    print("  • Dominant dimensions shape context")
    print("  • Memory themes create cross-connections")
    print("  • Preferences influence word selection")
    print()
    print("🔄 Unique Every Cycle:")
    print("  • No two prompts are identical")
    print("  • Each reflects the agent's current moment")
    print("  • Associative connections create novelty")
    print("  • Pathos state drives authentic variation")
    print()


if __name__ == "__main__":
    demonstrate_prompt_styles()
    explain_system_features()
    
    print("🎯 Implementation Complete!")
    print("=" * 30)
    print()
    print("The dynamic pathos-driven prompt generation system is now active.")
    print("Every agent cycle will generate a unique, state-appropriate prompt")
    print("that reflects the agent's current internal dynamics and creates")
    print("authentic variation in the reasoning process.")
    print()
    print("Key benefits:")
    print("• Eliminates repetitive, template-based prompts")
    print("• Creates genuine cognitive variety")
    print("• Reflects authentic internal states")
    print("• Uses associative semantic connections")
    print("• Adapts to energy levels and complexity")
    print("• Integrates memory themes and preferences")