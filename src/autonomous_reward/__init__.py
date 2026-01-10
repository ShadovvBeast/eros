"""
Autonomous State-Derived Reward System

This module implements a truly autonomous reward system that derives all reward signals
from the agent's own continuous internal state dynamics, eliminating external reward
definitions and enabling authentic value development.
"""

from .interfaces import (
    AutonomousRewardSystemInterface,
    StateDerivedRewardGeneratorInterface,
    IntrinsicMotivationEngineInterface,
    ProgressiveLearningControllerInterface,
    WorldInteractionRewardSystemInterface,
    EmergentValueSystemInterface,
    MCPIntegrationLayerInterface,
    SelfSuccessMetricsSystemInterface,
    CrossLayerSynergyInterface,
    AdaptiveComplexityManagementInterface
)

from .core import AutonomousRewardSystem
from .state_reward_generator import StateDerivedRewardGenerator
from .intrinsic_motivation_engine import IntrinsicMotivationEngine
from .progressive_learning_controller import ProgressiveLearningController
from .world_interaction_reward_system import WorldInteractionRewardSystem
from .emergent_value_system import EmergentValueSystem
from .self_success_metrics_system import SelfSuccessMetricsSystem
from .cross_layer_synergy import CrossLayerSynergy
from .adaptive_complexity_management import AdaptiveComplexityManagement
from .factory import (
    AutonomousRewardSystemFactory,
    create_autonomous_reward_system,
    create_autonomous_reward_system_with_factory
)
from .models import (
    StateReward,
    IntrinsicMotivation,
    ValuePattern,
    LearningState,
    WorldInteractionResult,
    AutonomousGoal
)

__all__ = [
    # Interfaces
    'AutonomousRewardSystemInterface',
    'StateDerivedRewardGeneratorInterface',
    'IntrinsicMotivationEngineInterface',
    'ProgressiveLearningControllerInterface',
    'WorldInteractionRewardSystemInterface',
    'EmergentValueSystemInterface',
    'MCPIntegrationLayerInterface',
    'SelfSuccessMetricsSystemInterface',
    'CrossLayerSynergyInterface',
    'AdaptiveComplexityManagementInterface',
    
    # Core implementation
    'AutonomousRewardSystem',
    'StateDerivedRewardGenerator',
    'IntrinsicMotivationEngine',
    'ProgressiveLearningController',
    'WorldInteractionRewardSystem',
    'EmergentValueSystem',
    'SelfSuccessMetricsSystem',
    'CrossLayerSynergy',
    'AdaptiveComplexityManagement',
    
    # Factory and creation utilities
    'AutonomousRewardSystemFactory',
    'create_autonomous_reward_system',
    'create_autonomous_reward_system_with_factory',
    
    # Data models
    'StateReward',
    'IntrinsicMotivation',
    'ValuePattern',
    'LearningState',
    'WorldInteractionResult',
    'AutonomousGoal'
]