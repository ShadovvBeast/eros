#!/usr/bin/env python3
"""
Create a visual mockup of the dukkha dashboard to show what it looks like.

This generates static images showing the dukkha dashboard layout and features.
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def create_dukkha_dashboard_mockup():
    """Create a visual mockup of the dukkha dashboard."""
    print("🔥 Creating Dukkha Dashboard Mockup")
    print("=" * 40)
    
    # Create figure with subplots matching the dashboard layout
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    fig.suptitle('🔥 Dukkha (Dissatisfaction) Dashboard - Real-Time Monitoring', 
                 fontsize=16, fontweight='bold')
    
    # Create subplots in 3x2 grid
    ax_total = fig.add_subplot(3, 2, 1)
    ax_types = fig.add_subplot(3, 2, 2)
    ax_radar = fig.add_subplot(3, 2, 3, projection='polar')
    ax_emotional = fig.add_subplot(3, 2, 4)
    ax_suggestions = fig.add_subplot(3, 2, 5)
    ax_insights = fig.add_subplot(3, 2, 6)
    
    # Generate sample data
    time_steps = np.arange(0, 50)
    
    # 1. Total Dissatisfaction Over Time
    total_dissatisfaction = 0.3 + 0.4 * np.sin(time_steps * 0.2) + 0.1 * np.random.normal(0, 1, len(time_steps))
    total_dissatisfaction = np.clip(total_dissatisfaction, 0, 1)
    
    ax_total.plot(time_steps, total_dissatisfaction, 'r-', linewidth=2, label='Total Dissatisfaction')
    ax_total.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Mild Threshold')
    ax_total.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='High Threshold')
    
    # Color-code points
    for i, val in enumerate(total_dissatisfaction):
        color = 'green' if val < 0.3 else 'orange' if val < 0.6 else 'red'
        ax_total.scatter(time_steps[i], val, c=color, alpha=0.6, s=20)
    
    ax_total.set_title('Total Dissatisfaction Over Time')
    ax_total.set_xlabel('Time Steps')
    ax_total.set_ylabel('Total Dissatisfaction')
    ax_total.legend()
    ax_total.grid(True, alpha=0.3)
    ax_total.set_ylim(0, 1)
    
    # 2. Individual Dukkha Types
    dukkha_types = {
        'Stagnation': 0.7 + 0.2 * np.sin(time_steps * 0.3),
        'Curiosity': 0.4 + 0.3 * np.sin(time_steps * 0.4 + 1),
        'Mastery': 0.3 + 0.2 * np.sin(time_steps * 0.2 + 2),
        'Existential': 0.6 + 0.3 * np.sin(time_steps * 0.15 + 3),
        'Goals': 0.2 + 0.25 * np.sin(time_steps * 0.5 + 4),
        'Novelty': 0.5 + 0.4 * np.sin(time_steps * 0.6 + 5)
    }
    
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
    
    for i, (name, values) in enumerate(dukkha_types.items()):
        values = np.clip(values, 0, 1)
        ax_types.plot(time_steps, values, color=colors[i], linewidth=1.5, 
                     label=name, alpha=0.8)
    
    ax_types.set_title('Dukkha Types Over Time')
    ax_types.set_xlabel('Time Steps')
    ax_types.set_ylabel('Dissatisfaction Level')
    ax_types.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_types.grid(True, alpha=0.3)
    ax_types.set_ylim(0, 1)
    
    # 3. Radar Chart - Current Dukkha Profile
    current_values = [values[-1] for values in dukkha_types.values()]
    labels = list(dukkha_types.keys())
    
    angles = np.linspace(0, 2 * np.pi, len(current_values), endpoint=False)
    current_values += current_values[:1]  # Complete the circle
    angles = np.concatenate((angles, [angles[0]]))
    
    ax_radar.plot(angles, current_values, 'r-', linewidth=2, alpha=0.8)
    ax_radar.fill(angles, current_values, 'red', alpha=0.25)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(labels)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax_radar.grid(True)
    ax_radar.set_title('Current Dukkha Profile', pad=20)
    
    # 4. Emotional State Evolution (Pie Chart)
    emotional_states = ['Restless & Eager', 'Curious & Driven', 'Contemplative', 'Content but Aware']
    state_counts = [25, 35, 20, 20]
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(emotional_states)))
    
    wedges, texts, autotexts = ax_emotional.pie(state_counts, labels=emotional_states, 
                                               autopct='%1.1f%%', colors=colors_pie, 
                                               startangle=90)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax_emotional.set_title('Emotional State Evolution')
    
    # Current state indicator
    ax_emotional.text(0.5, -1.3, 'Current: Restless & Eager for Change', 
                     ha='center', va='center', transform=ax_emotional.transAxes,
                     fontsize=12, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    # 5. Growth Suggestions
    suggestions = [
        "• Try a completely different tool or approach",
        "• Explore a new domain or capability", 
        "• Set a challenging new goal",
        "• Investigate unresolved questions",
        "• Practice skills at higher difficulty",
        "• Reflect on deeper purpose and meaning"
    ]
    
    ax_suggestions.set_title('Recent Growth Suggestions')
    y_pos = 0.9
    for suggestion in suggestions:
        ax_suggestions.text(0.05, y_pos, suggestion, 
                           transform=ax_suggestions.transAxes, fontsize=10,
                           verticalalignment='top')
        y_pos -= 0.15
    
    ax_suggestions.set_xlim(0, 1)
    ax_suggestions.set_ylim(0, 1)
    ax_suggestions.axis('off')
    
    # 6. Dukkha Insights
    insights = [
        "Total Dissatisfaction: 0.652",
        "Trend: Increasing",
        "Dominant Type: Stagnation (0.700)",
        "Growth Status: 🔥 High Growth Potential",
        "",
        "💡 Dukkha Philosophy:",
        "Dissatisfaction is not suffering—",
        "it's the healthy tension that",
        "drives all genuine growth."
    ]
    
    ax_insights.set_title('Dukkha Insights')
    y_pos = 0.95
    for insight in insights:
        color = 'red' if "Growth Status" in insight else 'black'
        fontweight = 'bold' if any(x in insight for x in ["Total", "Trend", "Dominant", "Growth Status"]) else 'normal'
        
        ax_insights.text(0.05, y_pos, insight, 
                        transform=ax_insights.transAxes, fontsize=10,
                        verticalalignment='top', color=color, fontweight=fontweight)
        y_pos -= 0.1
    
    ax_insights.set_xlim(0, 1)
    ax_insights.set_ylim(0, 1)
    ax_insights.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the mockup
    filename = 'dukkha_dashboard_mockup.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Dukkha dashboard mockup saved as: {filename}")
    
    # Show the plot
    plt.show()
    
    return filename

def create_comparison_mockup():
    """Create a before/after comparison showing the value of dukkha monitoring."""
    print("\n📊 Creating Before/After Comparison")
    print("=" * 40)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Agent Monitoring: Before vs After Dukkha Dashboard', fontsize=16, fontweight='bold')
    
    # Before: Traditional metrics only
    ax1.set_title('❌ BEFORE: Traditional Metrics Only', fontsize=14, color='red')
    
    # Simulate "successful" traditional metrics
    time_steps = np.arange(0, 30)
    rewards = 8.0 + 0.5 * np.random.normal(0, 1, len(time_steps))
    rewards = np.clip(rewards, 7.0, 9.0)
    
    ax1.plot(time_steps, rewards, 'g-', linewidth=3, label='Rewards (7.5-9.0)')
    ax1.axhline(y=8.0, color='green', linestyle='--', alpha=0.7, label='Target')
    ax1.fill_between(time_steps, 7.0, rewards, alpha=0.3, color='green')
    
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Reward Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(6, 10)
    
    # Add "successful" indicators
    ax1.text(0.5, 0.95, '✅ High Rewards', transform=ax1.transAxes, 
             ha='center', fontsize=12, color='green', fontweight='bold')
    ax1.text(0.5, 0.85, '✅ Zero Errors', transform=ax1.transAxes, 
             ha='center', fontsize=12, color='green', fontweight='bold')
    ax1.text(0.5, 0.75, '✅ "Successful" Metrics', transform=ax1.transAxes, 
             ha='center', fontsize=12, color='green', fontweight='bold')
    ax1.text(0.5, 0.15, '❌ BUT: Agent is STUCK', transform=ax1.transAxes, 
             ha='center', fontsize=14, color='red', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7))
    ax1.text(0.5, 0.05, 'No way to detect stagnation!', transform=ax1.transAxes, 
             ha='center', fontsize=10, color='red')
    
    # After: With dukkha monitoring
    ax2.set_title('✅ AFTER: With Dukkha Monitoring', fontsize=14, color='green')
    
    # Show both rewards and dissatisfaction
    ax2_twin = ax2.twinx()
    
    # Same rewards
    ax2.plot(time_steps, rewards, 'g-', linewidth=2, label='Rewards', alpha=0.7)
    ax2.set_ylabel('Reward Value', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Add dissatisfaction data
    dissatisfaction = 0.2 + 0.5 * np.sin(time_steps * 0.3) + 0.1 * np.random.normal(0, 1, len(time_steps))
    dissatisfaction = np.clip(dissatisfaction, 0, 1)
    
    ax2_twin.plot(time_steps, dissatisfaction, 'r-', linewidth=3, label='Dissatisfaction')
    ax2_twin.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='High Threshold')
    ax2_twin.set_ylabel('Dissatisfaction Level', color='red')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    ax2_twin.set_ylim(0, 1)
    
    ax2.set_xlabel('Time Steps')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(6, 10)
    
    # Add insights
    ax2.text(0.5, 0.95, '✅ High Rewards', transform=ax2.transAxes, 
             ha='center', fontsize=10, color='green', fontweight='bold')
    ax2.text(0.5, 0.85, '✅ Zero Errors', transform=ax2.transAxes, 
             ha='center', fontsize=10, color='green', fontweight='bold')
    ax2.text(0.5, 0.75, '🔥 Stagnation: 0.700', transform=ax2.transAxes, 
             ha='center', fontsize=10, color='red', fontweight='bold')
    ax2.text(0.5, 0.65, '🤔 Existential: 0.590', transform=ax2.transAxes, 
             ha='center', fontsize=10, color='red', fontweight='bold')
    ax2.text(0.5, 0.55, '🌈 Novelty Hunger: 0.480', transform=ax2.transAxes, 
             ha='center', fontsize=10, color='red', fontweight='bold')
    ax2.text(0.5, 0.15, '💡 Growth Suggestion:', transform=ax2.transAxes, 
             ha='center', fontsize=12, color='blue', fontweight='bold')
    ax2.text(0.5, 0.05, '"Try a different approach"', transform=ax2.transAxes, 
             ha='center', fontsize=10, color='blue',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    
    # Save comparison
    filename = 'dukkha_comparison_mockup.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Comparison mockup saved as: {filename}")
    
    plt.show()
    
    return filename

def main():
    """Create all dukkha dashboard mockups."""
    print("🔥 DUKKHA DASHBOARD MOCKUP GENERATOR")
    print("=" * 50)
    
    try:
        # Create main dashboard mockup
        dashboard_file = create_dukkha_dashboard_mockup()
        
        # Create comparison mockup
        comparison_file = create_comparison_mockup()
        
        print(f"\n🎉 Mockups created successfully!")
        print(f"   📊 Dashboard mockup: {dashboard_file}")
        print(f"   📈 Comparison mockup: {comparison_file}")
        
        print(f"\n🔥 The dukkha dashboard provides:")
        print("   - Real-time dissatisfaction monitoring")
        print("   - Six types of healthy dissatisfaction tracking")
        print("   - Emotional state evolution analysis")
        print("   - Growth suggestions based on dissatisfaction")
        print("   - Philosophical context about dukkha")
        
        print(f"\n✅ This solves the 'stuck agent' problem by making")
        print("   dissatisfaction visible and actionable!")
        
    except Exception as e:
        print(f"❌ Error creating mockups: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()