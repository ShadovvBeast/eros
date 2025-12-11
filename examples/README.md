# Autonomous Logos-Pathos-Memory Agent Examples

This directory contains example configurations, tools, and scenarios that demonstrate the capabilities of the Autonomous Logos-Pathos-Memory Agent system.

## Directory Structure

```
examples/
├── configurations/          # Agent personality configurations
│   ├── curious_explorer.py     # Exploration-focused agent
│   ├── cautious_analyst.py     # Analysis-focused agent
│   └── creative_collaborator.py # Creativity-focused agent
├── tools/                   # Example tool implementations
│   └── analysis_tools.py       # Analysis and research tools
├── scenarios/               # Complete usage scenarios
│   ├── research_assistant_scenario.py  # Academic research assistant
│   └── creative_writing_scenario.py    # Creative writing assistant
└── README.md               # This file
```

## Agent Configurations

### 1. Curious Explorer (`curious_explorer.py`)

A highly curious agent optimized for exploration and discovery:

- **Personality**: High openness (0.95), moderate risk tolerance (0.7)
- **Behavior**: Seeks novelty, takes calculated risks, shares discoveries
- **Use Cases**: Research, investigation, learning new domains
- **Scenarios**: Scientific research, creative exploration, technical investigation

```python
from examples.configurations.curious_explorer import create_curious_explorer_config

config = create_curious_explorer_config()
agent = AutonomousAgent(config)
```

### 2. Cautious Analyst (`cautious_analyst.py`)

A methodical agent focused on accuracy and thorough analysis:

- **Personality**: High conscientiousness (0.95), high caution (0.9)
- **Behavior**: Validates information, prefers proven approaches, systematic
- **Use Cases**: Data analysis, risk assessment, quality assurance
- **Scenarios**: Financial analysis, security analysis, compliance monitoring

```python
from examples.configurations.cautious_analyst import create_cautious_analyst_config

config = create_cautious_analyst_config()
agent = AutonomousAgent(config)
```

### 3. Creative Collaborator (`creative_collaborator.py`)

A highly creative agent optimized for innovation and collaboration:

- **Personality**: Maximum creativity (0.98), high agreeableness (0.95)
- **Behavior**: Generates novel ideas, works well with others, embraces ambiguity
- **Use Cases**: Creative projects, brainstorming, team facilitation
- **Scenarios**: Artistic creation, product innovation, storytelling

```python
from examples.configurations.creative_collaborator import create_creative_collaborator_config

config = create_creative_collaborator_config()
agent = AutonomousAgent(config)
```

## Example Tools

### Analysis Tools (`tools/analysis_tools.py`)

Demonstrates how to create domain-specific tools that integrate with the agent:

1. **TextAnalysisTool**: Analyzes text for sentiment, complexity, and themes
2. **DataPatternTool**: Finds patterns, trends, and anomalies in numerical data
3. **KnowledgeGraphTool**: Builds and queries knowledge graphs from information

```python
from examples.tools.analysis_tools import get_analysis_tools

# Register tools with agent
tools = get_analysis_tools()
for tool_info in tools:
    agent.tools.register_tool(
        tool_info["name"],
        tool_info["function"], 
        tool_info["metadata"]
    )
```

## Usage Scenarios

### 1. Research Assistant (`scenarios/research_assistant_scenario.py`)

Complete scenario showing an agent acting as a research assistant:

- **Features**: Literature search, citation analysis, research synthesis
- **Capabilities**: Autonomous research cycles, insight generation, progress tracking
- **Tools**: Academic databases, analysis tools, synthesis engines

```bash
cd examples/scenarios
python research_assistant_scenario.py
```

### 2. Creative Writing (`scenarios/creative_writing_scenario.py`)

Scenario demonstrating creative writing assistance:

- **Features**: Character generation, plot development, dialogue enhancement
- **Capabilities**: Story analysis, world building, collaborative writing
- **Tools**: Creative writing aids, narrative analysis, style enhancement

```bash
cd examples/scenarios  
python creative_writing_scenario.py
```

## Configuration Customization

### Personality Traits

All configurations support personality trait customization:

```python
config.ethos.personality_traits = {
    'openness': 0.8,           # Openness to experience
    'conscientiousness': 0.7,  # Organization and discipline
    'extraversion': 0.6,       # Social orientation
    'agreeableness': 0.8,      # Cooperation and trust
    'neuroticism': 0.3,        # Emotional stability
    # Custom traits
    'curiosity': 0.9,
    'creativity': 0.8,
    'caution': 0.4
}
```

### Pathos Configuration

Adjust affective dynamics for different behaviors:

```python
config.pathos = PathosConfig(
    state_dimension=128,        # Complexity of internal state
    decay_factor=0.95,         # State persistence (higher = more stable)
    echo_strength=0.1,         # Memory influence (higher = more influenced by past)
    salience_threshold=0.7,    # Memory storage threshold (higher = more selective)
    # Salience weights
    c3=1.5,  # Higher novelty seeking
    c5=1.2   # Higher interest responsiveness
)
```

### Tool Integration

Create custom tools for specific domains:

```python
def custom_tool(param1: str, param2: int) -> dict:
    """Custom tool implementation"""
    # Tool logic here
    return {
        "success": True,
        "result": "Tool output",
        "metadata": {"param1": param1, "param2": param2}
    }

# Register with agent
agent.tools.register_tool("custom_tool", custom_tool, {
    "description": "Custom tool for specific domain",
    "parameters": {
        "param1": {"type": "string", "description": "First parameter"},
        "param2": {"type": "integer", "description": "Second parameter"}
    }
})
```

## Running Examples

### Prerequisites

1. Install the agent system:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (if using external APIs):
```bash
export GEMINI_API_KEY="your_api_key_here"
```

### Running Configurations

Test individual configurations:

```python
from examples.configurations.curious_explorer import create_curious_explorer_config
from src.agent import AutonomousAgent

config = create_curious_explorer_config()
agent = AutonomousAgent(config)

# Initialize layers and run
# ... (see scenario files for complete setup)
```

### Running Scenarios

Execute complete scenarios:

```bash
# Research assistant
python examples/scenarios/research_assistant_scenario.py

# Creative writing
python examples/scenarios/creative_writing_scenario.py
```

## Extending Examples

### Creating New Configurations

1. Copy an existing configuration file
2. Modify personality traits and parameters
3. Adjust core values and safety constraints
4. Test with different scenarios

### Adding New Tools

1. Implement tool function with proper signature
2. Define metadata with description and parameters
3. Register with ToolLayer
4. Test integration with agent cycles

### Building New Scenarios

1. Choose appropriate base configuration
2. Register scenario-specific tools
3. Implement scenario logic and metrics
4. Add demonstration and testing functions

## Best Practices

### Configuration Design

- **Coherent Personalities**: Ensure personality traits align with intended behavior
- **Balanced Parameters**: Avoid extreme values that might cause instability
- **Clear Identity**: Write descriptive identity descriptions and core values

### Tool Development

- **Error Handling**: Always include proper error handling and validation
- **Consistent Interface**: Follow the standard tool interface pattern
- **Documentation**: Provide clear descriptions and parameter specifications

### Scenario Implementation

- **Gradual Complexity**: Start simple and add complexity incrementally
- **Monitoring**: Include metrics and status reporting
- **User Feedback**: Provide clear output and progress indicators

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all paths are correctly set up
2. **Configuration Errors**: Validate all configuration parameters
3. **Tool Registration**: Check tool signatures and metadata
4. **Memory Issues**: Monitor memory usage in long-running scenarios

### Debugging Tips

- Enable debug logging: `config.log_level = 'DEBUG'`
- Use smaller state dimensions for testing
- Implement tool mocking for development
- Add cycle-by-cycle output for detailed analysis

## Contributing

When adding new examples:

1. Follow existing naming conventions
2. Include comprehensive documentation
3. Add error handling and validation
4. Test with different configurations
5. Update this README with new examples