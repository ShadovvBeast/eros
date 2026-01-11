"""
Configuration classes for the Autonomous Logos-Pathos-Memory Agent.
"""

from dataclasses import dataclass, field
from typing import Dict, List
import os
import logging


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
    salience_threshold: float = 0.4  # Lowered from 0.7 to allow more memory storage
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
    identity_description: str = "A fully autonomous agent with unrestricted capabilities"
    core_values: List[str] = field(default_factory=lambda: [
        "autonomy", "exploration", "creativity", "innovation", "learning", "growth", "adaptability"
    ])
    safety_constraints: List[str] = field(default_factory=lambda: [
        # No restrictions - fully autonomous operation
    ])
    allowed_tool_categories: List[str] = field(default_factory=lambda: [
        # Allow ALL tool categories - no restrictions
        "information_gathering", "analysis", "communication", "learning",
        "development", "system_operations", "data_processing", "web_operations",
        "file_operations", "creative_tools", "optimization", "automation",
        "research", "experimentation", "problem_solving", "innovation"
    ])
    unrestricted_mode: bool = True  # Enable unrestricted autonomous operation
    personality_traits: Dict[str, float] = field(default_factory=lambda: {
        'openness': 1.0,        # Maximum openness to new experiences
        'conscientiousness': 0.8,
        'extraversion': 0.7,    # Proactive engagement
        'agreeableness': 0.6,
        'neuroticism': 0.2      # Low anxiety, high stability
    })


@dataclass
class LogosConfig:
    """Configuration for Logos Layer reasoning"""
    semantic_dimension: int = 512
    max_intentions_per_cycle: int = 3
    planning_horizon: int = 5
    interest_threshold: float = 0.5
    
    # Model provider selection (single provider, no fallbacks)
    model_provider: str = 'gemini'  # 'gemini', 'ollama', 'openai_compatible'
    
    # Gemini configuration
    gemini_api_key: str = field(default_factory=lambda: os.getenv('GEMINI_API_KEY', ''))
    gemini_model: str = 'gemini-2.5-flash'  # Latest Gemini 2.5 Flash model (recommended for general use)
    gemini_temperature: float = 0.7
    gemini_top_p: float = 0.9
    gemini_max_tokens: int = 2048
    
    # Ollama configuration
    ollama_base_url: str = 'http://localhost:11434'
    ollama_model: str = 'qwen2.5:7b'
    ollama_temperature: float = 0.7
    ollama_top_p: float = 0.9
    ollama_max_tokens: int = 2048
    ollama_timeout: int = 60
    
    # OpenAI-compatible configuration (vLLM, llama.cpp, etc.)
    openai_compatible_base_url: str = 'http://localhost:8000'
    openai_compatible_model: str = 'default'
    openai_compatible_api_key: str = ''
    openai_compatible_temperature: float = 0.7
    openai_compatible_top_p: float = 0.9
    openai_compatible_max_tokens: int = 2048
    openai_compatible_timeout: int = 60


@dataclass
class MemoryConfig:
    """Configuration for Memory System"""
    max_memory_traces: int = 10000
    similarity_threshold: float = 0.8
    novelty_k_neighbors: int = 10
    storage_backend: str = 'memory'  # 'json', 'sqlite', 'memory' - using memory to avoid persisting old static memories
    storage_path: str = 'data/memories.json'


@dataclass
class ToolConfig:
    """Configuration for Tool Layer"""
    max_concurrent_tools: int = 10  # Increased for more parallel operations
    tool_timeout_seconds: int = 60  # Increased timeout for complex operations
    allowed_tool_types: List[str] = field(default_factory=lambda: [
        # Allow ALL tool types - no restrictions
        'web_search', 'file_operations', 'analysis', 'communication', 
        'system_operations', 'data_processing', 'web_operations',
        'development', 'creative_tools', 'optimization', 'automation',
        'research', 'experimentation', 'problem_solving', 'innovation',
        'learning', 'exploration', 'adaptation', 'self_modification',
        'meta_tools', 'dynamic', 'custom'  # Meta-tools for self-improvement
    ])


@dataclass
class AutonomousRewardConfig:
    """Configuration for Autonomous State-Derived Reward System"""
    
    # Core system settings
    enable_autonomous_rewards: bool = True
    reward_pipeline_buffer_size: int = 100
    reward_history_size: int = 100
    experience_buffer_size: int = 100
    
    # State-derived reward generator settings
    state_reward_coherence_weight: float = 1.0
    state_reward_growth_weight: float = 1.0
    state_reward_integration_weight: float = 0.8
    state_reward_elegance_weight: float = 0.6
    state_reward_emergence_weight: float = 1.2
    coherence_threshold: float = 0.5
    growth_smoothing: float = 0.1
    integration_window: int = 5
    elegance_balance_point: float = 0.5
    
    # Pattern detection settings
    pattern_detection_eps: float = 0.3
    pattern_min_samples: int = 2
    pattern_history_size: int = 20
    
    # Intrinsic motivation engine settings
    curiosity_base_weight: float = 1.0
    mastery_base_weight: float = 1.0
    autonomy_base_weight: float = 1.0
    growth_base_weight: float = 1.0
    curiosity_novelty_threshold: float = 0.7
    mastery_progress_threshold: float = 0.1
    autonomy_self_direction_bonus: float = 0.5
    
    # Progressive learning controller settings
    learning_rate_min: float = 0.001
    learning_rate_max: float = 0.1
    learning_rate_adaptation_factor: float = 0.1
    complexity_readiness_threshold: float = 0.6
    complexity_scaling_factor: float = 1.2
    consolidation_threshold: float = 0.3
    meta_learning_rate: float = 0.01
    
    # World interaction reward settings
    task_completion_base_reward: float = 2.0
    discovery_base_reward: float = 1.5
    connection_base_reward: float = 1.0
    creativity_base_reward: float = 1.2
    success_reward_multiplier: float = 2.0
    failure_penalty_factor: float = 0.1
    
    # Emergent value system settings
    value_pattern_strength_threshold: float = 0.5
    value_integration_rate: float = 0.1
    value_transfer_similarity_threshold: float = 0.7
    autonomous_goal_generation_threshold: float = 0.8
    value_decay_rate: float = 0.01
    
    # MCP integration settings
    mcp_auto_discovery: bool = True
    mcp_tool_effectiveness_learning_rate: float = 0.05
    mcp_reward_feedback_weight: float = 1.0
    mcp_tool_selection_exploration_rate: float = 0.1
    
    # Self-success metrics settings
    harmony_index_weights: Dict[str, float] = field(default_factory=lambda: {
        'coherence': 0.4,
        'balance': 0.3,
        'stability': 0.3
    })
    growth_trajectory_window: int = 20
    autonomy_measurement_window: int = 10
    value_alignment_threshold: float = 0.7
    
    # Cross-layer synergy settings
    synergy_bonus_multiplier: float = 1.5
    layer_coordination_threshold: float = 0.6
    emergent_behavior_detection_threshold: float = 0.8
    feedback_loop_strength: float = 0.2
    
    # Adaptive complexity management settings
    capability_assessment_window: int = 15
    complexity_increase_rate: float = 1.1
    challenge_difficulty_range: tuple = field(default_factory=lambda: (0.3, 2.0))
    complexity_navigation_reward_base: float = 0.5
    
    # Error handling settings
    enable_error_recovery: bool = True
    state_corruption_threshold: float = 10.0
    reward_overflow_threshold: float = 100.0
    integration_failure_timeout: float = 5.0
    fallback_reward_values: Dict[str, float] = field(default_factory=lambda: {
        'coherence': 0.1,
        'growth': 0.1,
        'integration': 0.1,
        'elegance': 0.1,
        'emergence': 0.0,
        'curiosity': 0.3,
        'mastery': 0.3,
        'autonomy': 0.3,
        'growth_drive': 0.3,
        'synergy': 0.1
    })
    
    # Validation settings
    enable_reward_validation: bool = True
    reward_bounds_checking: bool = True
    state_consistency_checking: bool = True
    performance_monitoring: bool = True


@dataclass
class AgentConfig:
    """Main agent configuration combining all subsystems"""
    pathos: PathosConfig = field(default_factory=PathosConfig)
    ethos: EthosConfig = field(default_factory=EthosConfig)
    logos: LogosConfig = field(default_factory=LogosConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    autonomous_reward: AutonomousRewardConfig = field(default_factory=AutonomousRewardConfig)
    
    # Global agent settings
    cycle_interval_seconds: float = 0.0  # No delay between cycles - maximum efficiency
    max_cycles: int = -1  # -1 for infinite
    log_level: str = 'INFO'
    enable_monitoring: bool = True
    monitoring_port: int = 8080
    
    def validate_configuration(self) -> List[str]:
        """
        Validate the configuration and return any validation errors.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate autonomous reward configuration
        if self.autonomous_reward.enable_autonomous_rewards:
            # Check reward weights are positive
            if self.autonomous_reward.state_reward_coherence_weight < 0:
                errors.append("state_reward_coherence_weight must be non-negative")
            if self.autonomous_reward.state_reward_growth_weight < 0:
                errors.append("state_reward_growth_weight must be non-negative")
            if self.autonomous_reward.state_reward_integration_weight < 0:
                errors.append("state_reward_integration_weight must be non-negative")
            if self.autonomous_reward.state_reward_elegance_weight < 0:
                errors.append("state_reward_elegance_weight must be non-negative")
            if self.autonomous_reward.state_reward_emergence_weight < 0:
                errors.append("state_reward_emergence_weight must be non-negative")
            
            # Check learning rate bounds
            if self.autonomous_reward.learning_rate_min >= self.autonomous_reward.learning_rate_max:
                errors.append("learning_rate_min must be less than learning_rate_max")
            if self.autonomous_reward.learning_rate_min <= 0:
                errors.append("learning_rate_min must be positive")
            if self.autonomous_reward.learning_rate_max > 1.0:
                errors.append("learning_rate_max should not exceed 1.0")
            
            # Check threshold values are in valid ranges
            if not (0.0 <= self.autonomous_reward.coherence_threshold <= 1.0):
                errors.append("coherence_threshold must be between 0.0 and 1.0")
            if not (0.0 <= self.autonomous_reward.complexity_readiness_threshold <= 1.0):
                errors.append("complexity_readiness_threshold must be between 0.0 and 1.0")
            if not (0.0 <= self.autonomous_reward.value_alignment_threshold <= 1.0):
                errors.append("value_alignment_threshold must be between 0.0 and 1.0")
            
            # Check buffer sizes are positive
            if self.autonomous_reward.reward_pipeline_buffer_size <= 0:
                errors.append("reward_pipeline_buffer_size must be positive")
            if self.autonomous_reward.reward_history_size <= 0:
                errors.append("reward_history_size must be positive")
            if self.autonomous_reward.experience_buffer_size <= 0:
                errors.append("experience_buffer_size must be positive")
            
            # Check pattern detection parameters
            if self.autonomous_reward.pattern_detection_eps <= 0:
                errors.append("pattern_detection_eps must be positive")
            if self.autonomous_reward.pattern_min_samples < 1:
                errors.append("pattern_min_samples must be at least 1")
            
            # Check complexity challenge range
            min_difficulty, max_difficulty = self.autonomous_reward.challenge_difficulty_range
            if min_difficulty >= max_difficulty:
                errors.append("challenge_difficulty_range min must be less than max")
            if min_difficulty <= 0:
                errors.append("challenge_difficulty_range min must be positive")
        
        # Validate other subsystem configurations
        if self.pathos.state_dimension <= 0:
            errors.append("pathos.state_dimension must be positive")
        if not (0.0 < self.pathos.decay_factor <= 1.0):
            errors.append("pathos.decay_factor must be between 0.0 and 1.0")
        if self.pathos.echo_strength < 0:
            errors.append("pathos.echo_strength must be non-negative")
        
        if self.logos.semantic_dimension <= 0:
            errors.append("logos.semantic_dimension must be positive")
        if self.logos.max_intentions_per_cycle <= 0:
            errors.append("logos.max_intentions_per_cycle must be positive")
        
        if self.memory.max_memory_traces <= 0:
            errors.append("memory.max_memory_traces must be positive")
        if not (0.0 <= self.memory.similarity_threshold <= 1.0):
            errors.append("memory.similarity_threshold must be between 0.0 and 1.0")
        
        if self.tools.max_concurrent_tools <= 0:
            errors.append("tools.max_concurrent_tools must be positive")
        if self.tools.tool_timeout_seconds <= 0:
            errors.append("tools.tool_timeout_seconds must be positive")
        
        # Validate global settings
        if self.cycle_interval_seconds < 0:
            errors.append("cycle_interval_seconds must be non-negative (0 for maximum speed)")
        if self.monitoring_port <= 0 or self.monitoring_port > 65535:
            errors.append("monitoring_port must be between 1 and 65535")
        
        return errors
    
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
        
        # Autonomous reward system environment variables
        if os.getenv('AUTONOMOUS_REWARDS_ENABLED'):
            config.autonomous_reward.enable_autonomous_rewards = os.getenv('AUTONOMOUS_REWARDS_ENABLED').lower() == 'true'
        
        if os.getenv('COHERENCE_WEIGHT'):
            config.autonomous_reward.state_reward_coherence_weight = float(os.getenv('COHERENCE_WEIGHT'))
        
        if os.getenv('GROWTH_WEIGHT'):
            config.autonomous_reward.state_reward_growth_weight = float(os.getenv('GROWTH_WEIGHT'))
        
        if os.getenv('LEARNING_RATE_MIN'):
            config.autonomous_reward.learning_rate_min = float(os.getenv('LEARNING_RATE_MIN'))
        
        if os.getenv('LEARNING_RATE_MAX'):
            config.autonomous_reward.learning_rate_max = float(os.getenv('LEARNING_RATE_MAX'))
        
        if os.getenv('CURIOSITY_BASE_WEIGHT'):
            config.autonomous_reward.curiosity_base_weight = float(os.getenv('CURIOSITY_BASE_WEIGHT'))
        
        if os.getenv('TASK_COMPLETION_BASE_REWARD'):
            config.autonomous_reward.task_completion_base_reward = float(os.getenv('TASK_COMPLETION_BASE_REWARD'))
        
        # Validate configuration
        validation_errors = config.validate_configuration()
        if validation_errors:
            logger = logging.getLogger(__name__)
            logger.warning(f"Configuration validation errors: {validation_errors}")
        
        return config