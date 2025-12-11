"""
Configuration classes for the Autonomous Logos-Pathos-Memory Agent.
"""

from dataclasses import dataclass, field
from typing import Dict, List
import os


@dataclass
class PathosConfig:
    """Configuration for Pathos Layer affective dynamics"""
    state_dimension: int = 128
    decay_factor: float = 0.95  # α parameter
    echo_strength: float = 0.1  # β parameter
    homeostatic_weights: Dict[str, float] = field(default_factory=lambda: {
        'stability': 1.0,
        'complexity': 0.5,
        'coherence': 0.8
    })
    salience_threshold: float = 0.7
    memory_retrieval_k: int = 5
    # Internal reward computation parameters
    lambda_1: float = 1.0  # Homeostatic discomfort penalty
    lambda_2: float = 0.5  # State change smoothness penalty
    # Salience computation coefficients
    c1: float = 1.0  # State change magnitude
    c2: float = 1.0  # Reward magnitude
    c3: float = 0.8  # Novelty (affect)
    c4: float = 0.8  # Novelty (semantic)
    c5: float = 0.6  # Interest signal


@dataclass
class EthosConfig:
    """Configuration for Ethos Framework identity and constraints"""
    identity_description: str = "An autonomous agent focused on learning and exploration"
    core_values: List[str] = field(default_factory=lambda: [
        "curiosity", "safety", "growth", "coherence"
    ])
    safety_constraints: List[str] = field(default_factory=lambda: [
        "No harmful actions",
        "Respect privacy and boundaries",
        "Maintain ethical behavior"
    ])
    allowed_tool_categories: List[str] = field(default_factory=lambda: [
        "information_gathering", "analysis", "communication", "learning"
    ])
    personality_traits: Dict[str, float] = field(default_factory=lambda: {
        'openness': 0.8,
        'conscientiousness': 0.7,
        'extraversion': 0.5,
        'agreeableness': 0.8,
        'neuroticism': 0.3
    })


@dataclass
class LogosConfig:
    """Configuration for Logos Layer reasoning"""
    semantic_dimension: int = 512
    max_intentions_per_cycle: int = 3
    planning_horizon: int = 5
    interest_threshold: float = 0.5
    gemini_api_key: str = field(default_factory=lambda: os.getenv('GEMINI_API_KEY', ''))
    gemini_model: str = 'gemini-pro'


@dataclass
class MemoryConfig:
    """Configuration for Memory System"""
    max_memory_traces: int = 10000
    similarity_threshold: float = 0.8
    novelty_k_neighbors: int = 10
    storage_backend: str = 'json'  # 'json', 'sqlite', 'memory'
    storage_path: str = 'data/memories.json'


@dataclass
class ToolConfig:
    """Configuration for Tool Layer"""
    max_concurrent_tools: int = 3
    tool_timeout_seconds: int = 30
    allowed_tool_types: List[str] = field(default_factory=lambda: [
        'web_search', 'file_operations', 'analysis', 'communication'
    ])


@dataclass
class AgentConfig:
    """Main agent configuration combining all subsystems"""
    pathos: PathosConfig = field(default_factory=PathosConfig)
    ethos: EthosConfig = field(default_factory=EthosConfig)
    logos: LogosConfig = field(default_factory=LogosConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    
    # Global agent settings
    cycle_interval_seconds: float = 1.0
    max_cycles: int = -1  # -1 for infinite
    log_level: str = 'INFO'
    enable_monitoring: bool = True
    monitoring_port: int = 8080
    
    @classmethod
    def from_env(cls) -> 'AgentConfig':
        """Create configuration from environment variables"""
        config = cls()
        
        # Override with environment variables if present
        if os.getenv('PATHOS_STATE_DIMENSION'):
            config.pathos.state_dimension = int(os.getenv('PATHOS_STATE_DIMENSION'))
        
        if os.getenv('PATHOS_DECAY_FACTOR'):
            config.pathos.decay_factor = float(os.getenv('PATHOS_DECAY_FACTOR'))
        
        if os.getenv('GEMINI_API_KEY'):
            config.logos.gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        if os.getenv('LOG_LEVEL'):
            config.log_level = os.getenv('LOG_LEVEL')
        
        return config