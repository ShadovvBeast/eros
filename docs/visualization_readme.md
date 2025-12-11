# Visualization and Analysis Tools

This module provides comprehensive visualization and analysis capabilities for the Autonomous Logos-Pathos-Memory Agent.

## Overview

The visualization system consists of several specialized components:

1. **PathosStateVisualizer** - Real-time visualization of affective state dynamics
2. **PreferenceDriftAnalyzer** - Analysis and visualization of preference changes over time
3. **AttractorPatternDetector** - Detection and visualization of behavioral attractor patterns
4. **MemoryNetworkVisualizer** - Network visualization of memory relationships
5. **SystemHealthDashboard** - Comprehensive system health monitoring

## Dependencies

To use the visualization tools, install the following dependencies:

```bash
pip install matplotlib>=3.5.0 seaborn>=0.11.0 pandas>=1.3.0 networkx>=2.6.0
```

## Usage Examples

### Real-time Pathos State Visualization

```python
from src.visualization import PathosStateVisualizer
import numpy as np

# Create visualizer
viz = PathosStateVisualizer(state_dimension=128, history_length=100)

# Update with new states (typically called from agent loop)
for i in range(50):
    state_vector = np.random.random(128)
    internal_reward = 0.5 + 0.3 * np.sin(i * 0.1)
    viz.update_state(state_vector, internal_reward)

# Save visualization
viz.save_visualization("pathos_state.png")
viz.close()
```

### Preference Drift Analysis

```python
from src.visualization import PreferenceDriftAnalyzer
from src.logging_config import instrumentation

# Analyze preference drift from collected data
analyzer = PreferenceDriftAnalyzer()
analysis = analyzer.analyze_drift_from_collector(instrumentation)

# Generate visualization
analyzer.visualize_preference_drift(instrumentation, "preference_drift.png")

# Print analysis results
for category, metrics in analysis.items():
    if isinstance(metrics, dict):
        print(f"{category}: stability={metrics.get('stability_score', 0):.2f}")
```

### Attractor Pattern Detection

```python
from src.visualization import AttractorPatternDetector

# Detect attractor patterns
detector = AttractorPatternDetector()
patterns = detector.detect_patterns_from_collector(instrumentation)

# Visualize patterns
detector.visualize_attractor_patterns(instrumentation, "attractor_patterns.png")

print(f"Average attractor count: {patterns['average_attractor_count']:.2f}")
print(f"Pattern recognition rate: {patterns['pattern_recognition_rate']:.2f}")
```

### Memory Network Visualization

```python
from src.visualization import MemoryNetworkVisualizer

# Visualize memory network (requires memory traces)
memory_viz = MemoryNetworkVisualizer()
memory_viz.visualize_memory_network(memory_traces, max_nodes=50, 
                                   save_path="memory_network.png")
```

### System Health Dashboard

```python
from src.visualization import SystemHealthDashboard

# Generate comprehensive dashboard
dashboard = SystemHealthDashboard()
dashboard.generate_dashboard(instrumentation, "system_health.png")
```

### Comprehensive Analysis

```python
from src.visualization import create_comprehensive_analysis

# Create all visualizations at once
analysis_results = create_comprehensive_analysis(
    collector=instrumentation,
    memory_traces=memory_traces,
    save_directory="analysis_output"
)

print("Analysis complete!")
print(f"System metrics: {analysis_results['system_metrics']}")
```

## Visualization Components

### PathosStateVisualizer

Provides real-time visualization of:
- State vector norm over time
- Internal reward trajectory
- State vector components (PCA projection)
- Phase space plot (state norm vs reward)

### PreferenceDriftAnalyzer

Analyzes and visualizes:
- Preference weights over time
- Drift magnitude distribution
- Volatility analysis by category
- Stability scores

Key metrics:
- `drift_from_baseline`: Total change from initial weight
- `trend_slope`: Linear trend direction
- `volatility`: Standard deviation of changes
- `stability_score`: Overall stability measure (0-1)

### AttractorPatternDetector

Detects and visualizes:
- Attractor count evolution
- Attractor strength over time
- Pattern recognition events
- Attractor dynamics phase space

Key metrics:
- `average_attractor_count`: Mean number of attractors
- `pattern_recognition_rate`: Frequency of pattern detection
- `stable_periods`: Periods of consistent attractor behavior

### MemoryNetworkVisualizer

Creates network visualizations showing:
- Memory trace relationships based on similarity
- Node colors representing reward values
- Node sizes representing salience
- Memory distribution over time

### SystemHealthDashboard

Comprehensive dashboard including:
- System overview (runtime, cycles, events)
- Performance metrics (cycle time, efficiency)
- Preference system health
- Behavior stability
- Memory system health
- Tool usage analysis
- Error rate analysis
- Overall health scores
- System alerts and warnings

## Health Scoring

The system calculates health scores (0-1) for:

- **Performance**: Based on cycle timing and efficiency
- **Preferences**: Based on stability and drift patterns
- **Behavior**: Based on stability and exploration balance
- **Memory**: Based on activity and storage patterns
- **Reliability**: Based on error rates

## Alert System

The dashboard automatically detects and displays alerts for:
- High error rates (>10%)
- Excessive preference drift (>50% of categories)
- Low behavior stability (<0.3)
- Memory system inactivity
- High tool failure rates (>50%)

## Integration with Agent

The visualization tools are designed to work seamlessly with the agent's instrumentation system:

```python
# In agent cycle
from src.logging_config import instrumentation

# Data is automatically collected during agent operation
instrumentation.record_pathos_state(cycle, state_vector, reward, balance)
instrumentation.record_preference_change(category, old_weight, new_weight)
instrumentation.record_behavior_pattern(cycle, intention, tool, reward, similarity)

# Generate visualizations periodically
if cycle % 100 == 0:  # Every 100 cycles
    create_comprehensive_analysis(instrumentation, memory_traces, f"analysis_{cycle}")
```

## Performance Considerations

- **Real-time visualization**: Use `PathosStateVisualizer` sparingly in production
- **Memory usage**: Limit `max_nodes` in network visualization for large memory systems
- **File I/O**: Save visualizations to disk rather than displaying for headless operation
- **Update frequency**: Generate comprehensive analysis periodically, not every cycle

## Customization

All visualizers support customization:
- Color schemes and styling
- Plot dimensions and layouts
- Analysis parameters (thresholds, window sizes)
- Export formats and quality settings

The visualization system provides essential insights into agent behavior, helping developers understand:
- How the agent's internal state evolves over time
- Whether preferences are developing appropriately
- If attractor patterns are forming as expected
- How memory relationships are structured
- Overall system health and performance