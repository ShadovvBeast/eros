#!/usr/bin/env python3
"""
Demo: Memory Table Tab

Demonstrates the new memory table tab with sample data.
"""

import sys
from pathlib import Path
import time

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dashboard.core import create_interactive_dashboard
from core.logging_config import InstrumentationCollector
from core.models import MemoryTrace


def create_demo_memory_traces():
    """Create demonstration memory traces."""
    traces = []
    
    # Sample trace data with varied content
    sample_data = [
        {
            'salience': 0.95,
            'reward': 0.9,
            'intention': 'Explore the mysterious glowing artifact in the ancient chamber',
            'state': 'Highly curious and alert, sensors detecting unusual energy signatures',
            'action': 'Carefully approach the artifact while scanning for potential dangers',
            'observation': 'Artifact pulses with rhythmic blue light, temperature readings show it is warm to the touch',
            'reflection': 'This discovery could be significant - the energy patterns are unlike anything in my database'
        },
        {
            'salience': 0.85,
            'reward': 0.7,
            'intention': 'Analyze the energy patterns emanating from the discovered artifact',
            'state': 'Focused analytical mode, all sensors directed toward the target',
            'action': 'Deploy spectral analysis tools and record detailed measurements',
            'observation': 'Energy signature matches theoretical models of quantum resonance fields',
            'reflection': 'The patterns suggest this artifact may be a form of advanced technology'
        },
        {
            'salience': 0.78,
            'reward': 0.8,
            'intention': 'Learn about the historical context of this ancient chamber',
            'state': 'Investigative mindset, cross-referencing historical databases',
            'action': 'Examine wall inscriptions and architectural features for clues',
            'observation': 'Symbols match ancient civilization records from 3000 years ago',
            'reflection': 'This chamber predates known advanced civilizations - a remarkable find'
        },
        {
            'salience': 0.65,
            'reward': 0.4,
            'intention': 'Reflect on the implications of this discovery for current understanding',
            'state': 'Contemplative state, processing conflicting information',
            'action': 'Compare findings with established scientific theories',
            'observation': 'Current theories cannot fully explain the observed phenomena',
            'reflection': 'This discovery challenges fundamental assumptions about historical technology'
        },
        {
            'salience': 0.72,
            'reward': 0.6,
            'intention': 'Explore deeper into the chamber system for additional artifacts',
            'state': 'Adventurous and methodical, maintaining caution',
            'action': 'Navigate through narrow passages while mapping the layout',
            'observation': 'Multiple chambers connected by intricate tunnel networks',
            'reflection': 'This appears to be part of a much larger complex than initially thought'
        },
        {
            'salience': 0.88,
            'reward': 0.85,
            'intention': 'Analyze the interconnected chamber network architecture',
            'state': 'Systematic analysis mode, building comprehensive models',
            'action': 'Create detailed 3D maps and identify structural patterns',
            'observation': 'The layout follows mathematical principles of optimal space utilization',
            'reflection': 'The builders possessed advanced knowledge of engineering and mathematics'
        },
        {
            'salience': 0.55,
            'reward': 0.3,
            'intention': 'Learn from a navigation error that led to a dead end',
            'state': 'Slightly frustrated but maintaining analytical approach',
            'action': 'Retrace steps and analyze decision-making process',
            'observation': 'Misinterpreted subtle environmental cues that indicated the wrong path',
            'reflection': 'Need to improve pattern recognition for environmental navigation'
        },
        {
            'salience': 0.92,
            'reward': 0.9,
            'intention': 'Explore the central chamber that appears to be the heart of the complex',
            'state': 'Highly excited and focused, anticipating significant discoveries',
            'action': 'Carefully enter the central chamber with all sensors active',
            'observation': 'Massive crystalline structure in center, surrounded by intricate machinery',
            'reflection': 'This appears to be some kind of advanced control or power system'
        },
        {
            'salience': 0.68,
            'reward': 0.5,
            'intention': 'Reflect on the ethical implications of studying this ancient technology',
            'state': 'Thoughtful and cautious, considering broader consequences',
            'action': 'Evaluate potential risks and benefits of continued investigation',
            'observation': 'Technology could be beneficial but also potentially dangerous if misused',
            'reflection': 'Must proceed with careful consideration of long-term implications'
        },
        {
            'salience': 0.81,
            'reward': 0.75,
            'intention': 'Learn to integrate all discoveries into a coherent understanding',
            'state': 'Synthesizing mode, connecting disparate pieces of information',
            'action': 'Compile comprehensive report linking all observations and analyses',
            'observation': 'Patterns emerge showing sophisticated civilization with advanced capabilities',
            'reflection': 'This discovery will reshape understanding of ancient technological development'
        }
    ]
    
    # Create MemoryTrace objects
    base_time = time.time()
    for i, data in enumerate(sample_data):
        trace = MemoryTrace()
        trace.timestamp = base_time - (len(sample_data) - i) * 30  # 30 seconds apart
        trace.salience = data['salience']
        trace.reward = data['reward']
        trace.metadata = {'intention': data['intention']}
        trace.state = data['state']
        trace.action = data['action']
        trace.observation = data['observation']
        trace.reflection = data['reflection']
        traces.append(trace)
    
    return traces


def main():
    """Run the memory table demo."""
    print("🧠 Memory Table Tab Demo")
    print("=" * 60)
    print("This demo showcases the new Memory Table tab with sample data.")
    print()
    
    # Create sample memory traces
    print("📊 Creating sample memory traces...")
    memory_traces = create_demo_memory_traces()
    print(f"✅ Created {len(memory_traces)} sample memory traces")
    
    # Create instrumentation collector
    collector = InstrumentationCollector()
    
    # Create dashboard
    print("🎯 Launching E.R.O.S Control Center with Memory Table...")
    dashboard = create_interactive_dashboard(collector, update_interval=1.0)
    
    # Set memory traces
    dashboard.memory_traces = memory_traces
    if 'memory_table' in dashboard.tabs:
        dashboard.tabs['memory_table'].memory_traces = memory_traces
        print("✅ Memory table tab loaded with sample data")
    
    print()
    print("🔍 Memory Table Tab Features:")
    print("   📋 Tabular view of all memory traces")
    print("   🔍 Text filtering across all trace content")
    print("   📂 Category filtering (Exploration, Analysis, Learning, Reflection)")
    print("   📊 Detailed view when selecting any trace")
    print("   💾 CSV export functionality")
    print("   ⭐ Visual indicators for high-salience traces")
    print("   📈 Real-time statistics display")
    print()
    print("💡 Instructions:")
    print("   1. Click on the 'Memory Table' tab (📊)")
    print("   2. Browse the 10 sample memory traces")
    print("   3. Click any row to see detailed information")
    print("   4. Try filtering by typing 'artifact' or 'chamber'")
    print("   5. Use the category dropdown to filter by type")
    print("   6. Export data using the 'Export CSV' button")
    print()
    print("🚀 Starting dashboard...")
    print("=" * 60)
    
    # Run the dashboard
    dashboard.run()


if __name__ == "__main__":
    main()