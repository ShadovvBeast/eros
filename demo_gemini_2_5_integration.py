#!/usr/bin/env python3
"""
Demo of Gemini 2.5 Flash Integration in Logos Layer

This demo showcases the enhanced reasoning capabilities of the logos layer
using the latest Gemini 2.5 Flash model with SDK 1.55.0.
"""

import os
import sys
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, 'src')

try:
    from src.core.config import LogosConfig
    from src.logos.logos_layer import LogosLayer
    from src.core.models import SemanticVector, MemoryTrace
    print("✅ Successfully imported components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def demo_enhanced_reasoning():
    """Demonstrate enhanced reasoning with Gemini 2.5 Flash."""
    print("\n🧠 Gemini 2.5 Flash Enhanced Reasoning Demo")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found in .env file")
        return
    
    # Create configuration
    config = LogosConfig(
        gemini_model='gemini-2.5-flash',
        gemini_api_key=api_key
    )
    
    # Initialize logos layer
    logos = LogosLayer(config)
    print(f"✅ Initialized Logos Layer with {config.gemini_model}")
    
    # Create test scenario
    print("\n📋 Creating test scenario...")
    
    # Simulate some memories
    memories = [
        MemoryTrace(
            affect_state=np.random.randn(128) * 0.4,
            semantic_vector=SemanticVector(
                embedding=np.random.randn(512),
                intention_text="Successfully analyzed complex data patterns",
                semantic_category="analysis"
            ),
            reward=0.8,
            timestamp=int(datetime.now().timestamp()) - 3600,
            salience=0.9
        ),
        MemoryTrace(
            affect_state=np.random.randn(128) * 0.3,
            semantic_vector=SemanticVector(
                embedding=np.random.randn(512),
                intention_text="Explored creative problem-solving approaches",
                semantic_category="creativity"
            ),
            reward=0.6,
            timestamp=int(datetime.now().timestamp()) - 1800,
            salience=0.7
        ),
        MemoryTrace(
            affect_state=np.random.randn(128) * 0.5,
            semantic_vector=SemanticVector(
                embedding=np.random.randn(512),
                intention_text="Reflected on learning progress and adaptation",
                semantic_category="reflection"
            ),
            reward=0.7,
            timestamp=int(datetime.now().timestamp()) - 900,
            salience=0.8
        )
    ]
    
    # Current state
    current_state = np.random.randn(128) * 0.6
    
    print(f"   Created {len(memories)} memory traces")
    print(f"   Current state energy: {np.linalg.norm(current_state):.3f}")
    
    # Demo 1: Enhanced Intention Generation
    print("\n🎯 Demo 1: Enhanced Intention Generation")
    print("-" * 40)
    
    base_semantic = SemanticVector(
        embedding=np.random.randn(512),
        intention_text="Explore advanced autonomous reasoning patterns",
        semantic_category="exploration"
    )
    
    enhanced_intention = logos.generate_enhanced_intention_with_llm(
        base_semantic, current_state, memories
    )
    
    print(f"Original: {base_semantic.intention_text}")
    print(f"Enhanced: {enhanced_intention.description[:200]}...")
    print(f"Priority: {enhanced_intention.priority:.3f}")
    print(f"Tools: {enhanced_intention.tool_candidates[:3]}")
    
    # Demo 2: Semantic Pattern Analysis
    print("\n🔬 Demo 2: Semantic Pattern Analysis")
    print("-" * 40)
    
    analysis = logos.analyze_semantic_patterns_with_llm(memories, current_state)
    
    if analysis:
        print(f"Analysis generated: {len(analysis['analysis_text'])} characters")
        print(f"Model used: {analysis['model_used']}")
        print(f"Preview: {analysis['analysis_text'][:300]}...")
    
    # Demo 3: Meta-Cognitive Reflection
    print("\n🤔 Demo 3: Meta-Cognitive Reflection")
    print("-" * 40)
    
    # Create internal evaluation
    internal_eval = logos.evaluate_internal_state(current_state, memories)
    
    reflection = logos.generate_meta_cognitive_reflection(internal_eval)
    
    if reflection:
        print(f"Reflection generated: {len(reflection)} characters")
        print(f"Preview: {reflection[:400]}...")
    
    # Demo 4: Autonomous Exploration Planning
    print("\n🗺️  Demo 4: Autonomous Exploration Planning")
    print("-" * 40)
    
    knowledge_gaps = logos._identify_knowledge_gaps()
    successful_patterns = logos.analyze_historical_patterns()
    
    exploration_plan = logos.plan_autonomous_exploration(knowledge_gaps, successful_patterns)
    
    if exploration_plan:
        print(f"Plan generated: {len(exploration_plan['exploration_plan'])} characters")
        print(f"Based on {len(knowledge_gaps)} knowledge gaps")
        print(f"Preview: {exploration_plan['exploration_plan'][:300]}...")
    
    # Demo 5: Emergent Drives Detection
    print("\n🌟 Demo 5: Emergent Drives Detection")
    print("-" * 40)
    
    drives = logos.detect_emergent_drives(current_state, memories, logos.preference_weights)
    
    print(f"Detected {len(drives)} emergent drives:")
    for i, drive in enumerate(drives):
        print(f"  {i+1}. {drive['type'].title()}: {drive['description']} (strength: {drive['strength']:.2f})")
    
    print("\n🎉 Demo completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    print("🚀 Gemini 2.5 Flash Integration Demo")
    print("Using latest SDK 1.55.0 with enhanced reasoning capabilities")
    
    demo_enhanced_reasoning()