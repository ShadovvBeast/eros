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
    
    # Data models
    'StateReward',
    'IntrinsicMotivation',
    'ValuePattern',
    'LearningState',
    'WorldInteractionResult',
    'AutonomousGoal'
]