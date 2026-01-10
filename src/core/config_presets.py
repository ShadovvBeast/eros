"""
Configuration Presets

Pre-defined configuration profiles for different use cases.
"""

from typing import Dict
from .config import AgentConfig


def create_development_config() -> AgentConfig:
    """Create configuration optimized for development and testing."""
    config = AgentConfig()
    
    # Development-friendly settings
    config.cycle_interval_seconds = 2.0  # Slower for debugging
    config.log_level = 'DEBUG'
    config.enable_monitoring = True
    
    # Pathos settings for development
    config.pathos.state_dimension = 64  # Smaller for faster processing
    config.pathos.decay_factor = 0.9
    config.pathos.echo_strength = 0.2
    config.pathos.salience_threshold = 0.3  # Lower threshold for more activity
    
    # Logos settings for development
    config.logos.semantic_dimension = 256  # Smaller for faster processing
    config.logos.max_intentions_per_cycle = 2
    config.logos.planning_horizon = 3
    config.logos.model_provider = 'gemini'  # Default to Gemini for development
    config.logos.gemini_model = 'gemini-2.5-flash'  # Fast model
    
    # Memory settings for development
    config.memory.max_memory_traces = 1000  # Smaller for testing
    config.memory.storage_backend = 'memory'  # In-memory for speed
    
    # Tool settings for development
    config.tools.max_concurrent_tools = 5
    config.tools.tool_timeout_seconds = 30
    
    # Autonomous reward settings for development
    config.autonomous_reward.enable_autonomous_rewards = True
    config.autonomous_reward.learning_rate_min = 0.01
    config.autonomous_reward.learning_rate_max = 0.2
    
    return config


def create_local_development_config() -> AgentConfig:
    """Create configuration optimized for local model development."""
    config = AgentConfig()
    
    # Development-friendly settings
    config.cycle_interval_seconds = 3.0  # Slower for local models
    config.log_level = 'DEBUG'
    config.enable_monitoring = True
    
    # Pathos settings for local development
    config.pathos.state_dimension = 64  # Smaller for faster processing
    config.pathos.decay_factor = 0.9
    config.pathos.echo_strength = 0.2
    config.pathos.salience_threshold = 0.3
    
    # Logos settings for local development
    config.logos.semantic_dimension = 256
    config.logos.max_intentions_per_cycle = 2
    config.logos.planning_horizon = 3
    config.logos.model_provider = 'ollama'  # Use Ollama for local models
    config.logos.ollama_base_url = 'http://localhost:11434'
    config.logos.ollama_model = 'qwen2.5:7b'
    config.logos.ollama_temperature = 0.7
    config.logos.ollama_timeout = 120  # Longer timeout for local models
    
    # Memory settings
    config.memory.max_memory_traces = 1000
    config.memory.storage_backend = 'memory'
    
    # Tool settings
    config.tools.max_concurrent_tools = 3  # Fewer concurrent tools for local
    config.tools.tool_timeout_seconds = 60
    
    # Autonomous reward settings
    config.autonomous_reward.enable_autonomous_rewards = True
    config.autonomous_reward.learning_rate_min = 0.01
    config.autonomous_reward.learning_rate_max = 0.2
    
    return config


def create_offline_config() -> AgentConfig:
    """Create configuration for completely offline operation."""
    config = AgentConfig()
    
    # Offline settings
    config.cycle_interval_seconds = 4.0  # Slower for local processing
    config.log_level = 'INFO'
    config.enable_monitoring = True
    
    # Pathos settings
    config.pathos.state_dimension = 64
    config.pathos.decay_factor = 0.95
    config.pathos.echo_strength = 0.1
    config.pathos.salience_threshold = 0.4
    
    # Logos settings for offline
    config.logos.semantic_dimension = 256
    config.logos.max_intentions_per_cycle = 2
    config.logos.planning_horizon = 3
    config.logos.model_provider = 'ollama'  # Local models only
    config.logos.ollama_base_url = 'http://localhost:11434'
    config.logos.ollama_model = 'qwen2.5:7b'
    config.logos.ollama_temperature = 0.8  # More creative for offline
    config.logos.ollama_timeout = 180  # Very long timeout
    
    # Memory settings
    config.memory.max_memory_traces = 5000
    config.memory.storage_backend = 'json'  # Persistent for offline
    
    # Tool settings
    config.tools.max_concurrent_tools = 2
    config.tools.tool_timeout_seconds = 120
    
    # Autonomous reward settings
    config.autonomous_reward.enable_autonomous_rewards = True
    config.autonomous_reward.learning_rate_min = 0.005
    config.autonomous_reward.learning_rate_max = 0.1
    
    return config


def create_production_config() -> AgentConfig:
    """Create configuration optimized for production use."""
    config = AgentConfig()
    
    # Production settings
    config.cycle_interval_seconds = 1.0
    config.log_level = 'INFO'
    config.enable_monitoring = True
    
    # Pathos settings for production
    config.pathos.state_dimension = 128
    config.pathos.decay_factor = 0.95
    config.pathos.echo_strength = 0.1
    config.pathos.salience_threshold = 0.4
    
    # Logos settings for production
    config.logos.semantic_dimension = 512
    config.logos.max_intentions_per_cycle = 3
    config.logos.planning_horizon = 5
    config.logos.gemini_model = 'gemini-2.5-flash'
    
    # Memory settings for production
    config.memory.max_memory_traces = 10000
    config.memory.storage_backend = 'json'  # Persistent storage
    
    # Tool settings for production
    config.tools.max_concurrent_tools = 10
    config.tools.tool_timeout_seconds = 60
    
    # Autonomous reward settings for production
    config.autonomous_reward.enable_autonomous_rewards = True
    config.autonomous_reward.learning_rate_min = 0.001
    config.autonomous_reward.learning_rate_max = 0.1
    
    return config


def create_research_config() -> AgentConfig:
    """Create configuration optimized for research and experimentation."""
    config = AgentConfig()
    
    # Research settings
    config.cycle_interval_seconds = 0.5  # Fast cycles for experimentation
    config.log_level = 'DEBUG'
    config.enable_monitoring = True
    
    # Pathos settings for research
    config.pathos.state_dimension = 256  # Larger for complex dynamics
    config.pathos.decay_factor = 0.98  # Slower decay for longer memory
    config.pathos.echo_strength = 0.15
    config.pathos.salience_threshold = 0.2  # Very low for maximum activity
    
    # Logos settings for research
    config.logos.semantic_dimension = 1024  # Large for complex reasoning
    config.logos.max_intentions_per_cycle = 5  # More intentions
    config.logos.planning_horizon = 10  # Longer planning
    config.logos.gemini_model = 'gemini-2.5-flash'
    
    # Memory settings for research
    config.memory.max_memory_traces = 50000  # Large memory for experiments
    config.memory.storage_backend = 'sqlite'  # Database for analysis
    
    # Tool settings for research
    config.tools.max_concurrent_tools = 20  # Many tools for exploration
    config.tools.tool_timeout_seconds = 120  # Longer timeouts
    
    # Autonomous reward settings for research
    config.autonomous_reward.enable_autonomous_rewards = True
    config.autonomous_reward.learning_rate_min = 0.005
    config.autonomous_reward.learning_rate_max = 0.3  # Higher for exploration
    
    # Enhanced research-specific settings
    config.autonomous_reward.curiosity_base_weight = 2.0  # High curiosity
    config.autonomous_reward.discovery_base_reward = 3.0  # High discovery rewards
    config.autonomous_reward.creativity_base_reward = 2.5  # High creativity rewards
    
    return config


def create_minimal_config() -> AgentConfig:
    """Create minimal configuration for resource-constrained environments."""
    config = AgentConfig()
    
    # Minimal settings
    config.cycle_interval_seconds = 5.0  # Very slow cycles
    config.log_level = 'WARNING'
    config.enable_monitoring = False  # Disable monitoring to save resources
    
    # Minimal Pathos settings
    config.pathos.state_dimension = 32  # Very small
    config.pathos.decay_factor = 0.8
    config.pathos.echo_strength = 0.05
    config.pathos.salience_threshold = 0.8  # High threshold
    
    # Minimal Logos settings
    config.logos.semantic_dimension = 128  # Small
    config.logos.max_intentions_per_cycle = 1  # Single intention
    config.logos.planning_horizon = 2  # Short planning
    config.logos.gemini_model = 'gemini-2.5-flash'
    
    # Minimal Memory settings
    config.memory.max_memory_traces = 100  # Very small memory
    config.memory.storage_backend = 'memory'  # In-memory only
    
    # Minimal Tool settings
    config.tools.max_concurrent_tools = 1  # Single tool at a time
    config.tools.tool_timeout_seconds = 15  # Short timeouts
    
    # Minimal Autonomous reward settings
    config.autonomous_reward.enable_autonomous_rewards = False  # Disabled
    
    return config


def create_creative_config() -> AgentConfig:
    """Create configuration optimized for creative and exploratory tasks."""
    config = AgentConfig()
    
    # Creative settings
    config.cycle_interval_seconds = 1.5
    config.log_level = 'INFO'
    config.enable_monitoring = True
    
    # Creative Pathos settings
    config.pathos.state_dimension = 192
    config.pathos.decay_factor = 0.92  # Faster decay for dynamic states
    config.pathos.echo_strength = 0.25  # Strong echoes for creativity
    config.pathos.salience_threshold = 0.3  # Low threshold for activity
    
    # Creative Logos settings
    config.logos.semantic_dimension = 768
    config.logos.max_intentions_per_cycle = 4  # Multiple creative intentions
    config.logos.planning_horizon = 7
    config.logos.interest_threshold = 0.3  # Low threshold for exploration
    config.logos.gemini_model = 'gemini-2.5-flash'
    
    # Creative Memory settings
    config.memory.max_memory_traces = 20000
    config.memory.similarity_threshold = 0.6  # Lower for diverse memories
    config.memory.storage_backend = 'json'
    
    # Creative Tool settings
    config.tools.max_concurrent_tools = 15
    config.tools.tool_timeout_seconds = 90
    
    # Creative Autonomous reward settings
    config.autonomous_reward.enable_autonomous_rewards = True
    config.autonomous_reward.curiosity_base_weight = 1.5
    config.autonomous_reward.creativity_base_reward = 2.0
    config.autonomous_reward.discovery_base_reward = 1.8
    config.autonomous_reward.learning_rate_max = 0.15
    
    # Enhanced creativity settings
    config.autonomous_reward.state_reward_emergence_weight = 1.5  # High emergence
    config.autonomous_reward.autonomy_base_weight = 1.3  # High autonomy
    
    return config


# Registry of all available presets
PRESET_REGISTRY: Dict[str, callable] = {
    'development': create_development_config,
    'production': create_production_config,
    'research': create_research_config,
    'minimal': create_minimal_config,
    'creative': create_creative_config,
}


def get_preset_config(preset_name: str) -> AgentConfig:
    """
    Get a preset configuration by name.
    
    Args:
        preset_name: Name of the preset ('development', 'production', etc.)
        
    Returns:
        AgentConfig instance for the preset
        
    Raises:
        ValueError: If preset name is not found
    """
    if preset_name not in PRESET_REGISTRY:
        available = ', '.join(PRESET_REGISTRY.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")
    
    return PRESET_REGISTRY[preset_name]()


def list_available_presets() -> Dict[str, str]:
    """
    List all available configuration presets with descriptions.
    
    Returns:
        Dictionary mapping preset names to descriptions
    """
    descriptions = {
        'development': 'Optimized for development and testing with debug logging and smaller dimensions',
        'production': 'Balanced configuration for production use with standard settings',
        'research': 'High-capacity configuration for research and experimentation',
        'minimal': 'Minimal resource usage for constrained environments',
        'creative': 'Enhanced creativity and exploration settings for artistic tasks',
    }
    
    return descriptions


def get_preset_comparison() -> Dict[str, Dict[str, any]]:
    """
    Get a comparison of key settings across all presets.
    
    Returns:
        Dictionary with preset names as keys and setting comparisons as values
    """
    comparison = {}
    
    for preset_name in PRESET_REGISTRY:
        config = get_preset_config(preset_name)
        comparison[preset_name] = {
            'cycle_interval': config.cycle_interval_seconds,
            'pathos_dimension': config.pathos.state_dimension,
            'logos_dimension': config.logos.semantic_dimension,
            'max_memory': config.memory.max_memory_traces,
            'max_tools': config.tools.max_concurrent_tools,
            'autonomous_rewards': config.autonomous_reward.enable_autonomous_rewards,
            'log_level': config.log_level,
        }
    
    return comparison