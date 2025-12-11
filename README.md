# Autonomous Logos-Pathos-Memory Agent

An autonomous AI agent that operates continuously without user interaction, featuring three interconnected layers: Logos (reasoning), Pathos (affective dynamics), and Memory (associative storage), all operating within an Ethos framework.

## Project Structure

```
src/
├── __init__.py                 # Main package
├── agent.py                    # Main agent coordination class
├── models.py                   # Core data structures
├── config.py                   # Configuration classes
├── logging_config.py           # Logging and instrumentation
├── logos/                      # Cognitive reasoning layer
│   ├── __init__.py
│   └── interfaces.py           # Logos layer interfaces
├── pathos/                     # Affective dynamics layer
│   ├── __init__.py
│   └── interfaces.py           # Pathos layer interfaces
├── memory/                     # Associative memory system
│   ├── __init__.py
│   └── interfaces.py           # Memory system interfaces
├── ethos/                      # Identity and constraints
│   ├── __init__.py
│   └── interfaces.py           # Ethos framework interfaces
└── tools/                      # Environment interaction
    ├── __init__.py
    └── interfaces.py           # Tool layer interfaces

tests/
├── __init__.py
├── conftest.py                 # Test configuration and fixtures
└── test_models.py              # Core model tests

.env                            # Environment configuration
requirements.txt                # Python dependencies
pytest.ini                     # Test configuration
setup.py                       # Package setup
```

## Architecture Overview

The agent implements a continuous operation loop:
1. **Logos** reads current Pathos state and recent memories
2. **Logos** generates semantic representation and intention
3. **Pathos** updates affective state based on semantic input
4. **Pathos** computes salience and optionally writes memory
5. **Memory** retrieval influences Pathos through affect echoes
6. **Logos** optionally executes tools based on updated state
7. **Tool** results feed back into Pathos as external reward

## Key Features

- **High-dimensional continuous affective state** with mathematical dynamics
- **Salience-based memory storage** indexed by emotional significance
- **Emergent preference development** through reward-based learning
- **Safety constraints** via Ethos framework validation
- **Property-based testing** for correctness verification
- **Comprehensive instrumentation** for monitoring and analysis
- **Interactive Real-time Dashboard** with professional tabbed interface
- **Advanced Visualization Suite** for system health and behavior analysis

## Interactive Dashboard

The project includes a comprehensive interactive dashboard for real-time monitoring and analysis of the autonomous agent system.

### Features

- **🏥 System Health**: Real-time overview of agent performance, memory usage, and system status
- **🖥️ Hardware Monitor**: Comprehensive CPU, memory, disk, network, GPU, and temperature monitoring
- **💝 Pathos State**: Live visualization of affective state dynamics and internal reward patterns
- **🧠 Memory Network**: Interactive memory trace analysis and relationship mapping
- **📊 Preference Drift**: Analysis of preference evolution and stability over time
- **🌀 Attractor Patterns**: Detection and visualization of behavioral attractor dynamics
- **⚡ Performance**: Comprehensive performance metrics and trend analysis
- **🎮 Session Control**: Full session lifecycle management with start/stop/pause controls
- **⚙️ Agent Configuration**: Interactive agent parameter configuration
- **🛡️ Safety Monitoring**: Real-time safety checks with emergency stop capabilities

### Quick Start

#### Primary Interface (Recommended)
```bash
# Launch the Autonomous Agent Control Center
python autonomous_agent_gui.py
```

#### Alternative Options
```bash
# Test dashboard with mock data
python test_interactive_dashboard.py

# Legacy integrated system
python interactive_agent_monitor.py
```

### Dashboard Controls

- **▶️ Start/Stop Monitoring**: Toggle real-time data updates
- **🔄 Refresh**: Manually update all visualizations
- **💾 Export Data**: Save current visualizations and data to files
- **⚙️ Update Interval**: Adjust refresh rate (0.5-10 seconds)

### Control Center Interface

**Session Controls:**
- **▶️ Start/Stop Session**: Full session lifecycle management
- **⏸️ Pause/Resume**: Pause and resume agent operations
- **🛑 Emergency Stop**: Immediate session termination with safety logging
- **⚙️ Configure Agent**: Interactive agent parameter configuration
- **📊 Start/Stop Monitoring**: Toggle real-time visualization updates

**Dashboard Tabs:**
1. **System Health**: Enhanced overview with hardware integration and session status
2. **Hardware Monitor**: Comprehensive system resource monitoring (CPU, memory, disk, network, GPU, temperatures)
3. **Pathos State**: Real-time affective dynamics, state evolution, phase space analysis
4. **Memory Network**: Memory trace relationships, network statistics, and connectivity analysis
5. **Preference Drift**: Preference weight evolution, stability analysis, and drift detection
6. **Attractor Patterns**: Behavioral pattern detection, attractor dynamics, and stability analysis
7. **Performance**: Cycle times, tool success rates, efficiency trends, and optimization metrics

## Configuration

The agent is configured through environment variables and configuration classes:

- `GEMINI_API_KEY`: API key for Gemini integration
- `PATHOS_STATE_DIMENSION`: Dimensionality of affective state (default: 128)
- `PATHOS_DECAY_FACTOR`: State decay parameter α (default: 0.95)
- `LOG_LEVEL`: Logging level (default: INFO)

## Testing

The project uses pytest with Hypothesis for property-based testing:

```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run specific test file
python -m pytest tests/test_models.py
```

## Development Status

**✅ FULLY IMPLEMENTED** - All core agent layers and systems are complete and operational:

- **🧠 Logos Layer**: Complete cognitive reasoning engine with preference learning
- **💝 Pathos Layer**: Full affective dynamics with attractor patterns and homeostasis
- **🧠 Memory System**: Associative memory with novelty detection and similarity-based retrieval
- **🛡️ Ethos Framework**: Identity management and safety constraint validation
- **🔧 Tools Layer**: MCP-based tool integration with comprehensive built-in tools
- **📊 Interactive Dashboard**: Professional GUI with hardware monitoring and session control
- **🎮 Session Management**: Complete lifecycle management with safety monitoring

## Visualization System

The project includes a comprehensive visualization system for analyzing agent behavior:

### Core Visualization Components

- **`src/visualization.py`**: Core visualization classes and analysis tools
- **`src/interactive_dashboard.py`**: Unified interactive dashboard interface
- **`interactive_agent_monitor.py`**: Integrated agent + dashboard launcher

### Visualization Classes

- **`PathosStateVisualizer`**: Real-time Pathos affective state dynamics
- **`PreferenceDriftAnalyzer`**: Preference evolution and drift detection
- **`AttractorPatternDetector`**: Behavioral attractor pattern analysis
- **`MemoryNetworkVisualizer`**: Memory trace relationship mapping
- **`SystemHealthDashboard`**: Comprehensive system health monitoring
- **`InteractiveDashboard`**: Unified tabbed interface for all visualizations

### Export Capabilities

All visualizations can be exported as:
- High-resolution PNG images (300 DPI)
- JSON data summaries
- Complete analysis reports
- Real-time dashboard snapshots

## Requirements

- Python 3.8+
- NumPy for mathematical operations
- Matplotlib for visualization and plotting
- Tkinter for interactive GUI (included with Python)
- Hypothesis for property-based testing
- Google Generative AI for Logos layer integration
- See `requirements.txt` for complete dependency list

## Agent Operation Modes

The system provides multiple ways to run the autonomous agent:

### 1. Interactive GUI Control Center (Recommended)
```bash
python autonomous_agent_gui.py
```
- **Professional interface** with session management
- **Real-time monitoring** of all agent layers
- **Hardware monitoring** (CPU, memory, disk, network, GPU)
- **Safety controls** with emergency stop
- **Data export** and comprehensive analysis

### 2. Ultimate Autonomous Agent
```bash
python ultimate_autonomous_agent.py
```
- **Extended runtime** capabilities (2-20 minutes)
- **Advanced tool creation** and optimization
- **Comprehensive monitoring** with automatic analysis
- **Professional visualizations** and reports

### 3. Legacy Systems
```bash
# Simple test runs
python simple_agent_test.py

# Comprehensive agent with visualization
python comprehensive_agent_with_visualization.py
```

Eros - Pathos Engine
