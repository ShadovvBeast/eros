"""
Factory for creating and configuring the Autonomous State-Derived Reward System.

This module implements the factory pattern for reward system creation with dependency
injection for all components, initialization and cleanup procedures.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..core.config import AgentConfig, AutonomousRewardConfig
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
from .error_handling import RewardSystemErrorHandler

# Try to import MCP integration layer
try:
    from .mcp_integration_layer import MCPIntegrationLayer
    MCP_AVAILABLE = True
except ImportError:
    from .mcp_integration_layer_standalone import MCPIntegrationLayer
    MCP_AVAILABLE = False

logger = logging.getLogger(__name__)


class AutonomousRewardSystemFactory:
    """
    Factory for creating and configuring the complete Autonomous State-Derived Reward System.
    
    This factory implements dependency injection for all components and provides
    initialization and cleanup procedures for the entire reward system.
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the factory with configuration.
        
        Args:
            config: Complete agent configuration including autonomous reward settings
        """
        self.config = config
        self.autonomous_reward_config = config.autonomous_reward
        
        # Track created components for cleanup
        self._created_components: Dict[str, Any] = {}
        self._initialization_order: List[str] = []
        self._is_initialized = False
        
        logger.info("Initialized AutonomousRewardSystemFactory")
    
    def create_autonomous_reward_system(self) -> AutonomousRewardSystemInterface:
        """
        Create and configure the complete autonomous reward system.
        
        Returns:
            Fully configured AutonomousRewardSystem instance
            
        Raises:
            RuntimeError: If system creation fails
        """
        try:
            logger.info("Creating autonomous reward system...")
            
            # Validate configuration first
            validation_errors = self.config.validate_configuration()
            if validation_errors:
                error_msg = f"Configuration validation failed: {validation_errors}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Create core system first
            core_system = self._create_core_system()
            
            # Create all subsystems with dependency injection
            state_reward_generator = self._create_state_reward_generator()
            intrinsic_motivation_engine = self._create_intrinsic_motivation_engine()
            progressive_learning_controller = self._create_progressive_learning_controller()
            world_interaction_reward_system = self._create_world_interaction_reward_system()
            emergent_value_system = self._create_emergent_value_system()
            mcp_integration_layer = self._create_mcp_integration_layer()
            self_success_metrics_system = self._create_self_success_metrics_system()
            cross_layer_synergy = self._create_cross_layer_synergy()
            adaptive_complexity_management = self._create_adaptive_complexity_management()
            
            # Initialize the core system with all subsystems
            core_system.initialize_subsystems(
                state_reward_generator=state_reward_generator,
                intrinsic_motivation_engine=intrinsic_motivation_engine,
                progressive_learning_controller=progressive_learning_controller,
                world_interaction_reward_system=world_interaction_reward_system,
                emergent_value_system=emergent_value_system,
                mcp_integration_layer=mcp_integration_layer,
                self_success_metrics_system=self_success_metrics_system,
                cross_layer_synergy=cross_layer_synergy,
                adaptive_complexity_management=adaptive_complexity_management
            )
            
            self._is_initialized = True
            
            logger.info("Successfully created and initialized autonomous reward system")
            return core_system
            
        except Exception as e:
            logger.error(f"Failed to create autonomous reward system: {e}")
            # Cleanup any partially created components
            self.cleanup()
            raise RuntimeError(f"Autonomous reward system creation failed: {e}")
    
    def _create_core_system(self) -> AutonomousRewardSystem:
        """Create the core autonomous reward system."""
        logger.debug("Creating core autonomous reward system...")
        
        core_system = AutonomousRewardSystem(self.config)
        
        self._created_components['core_system'] = core_system
        self._initialization_order.append('core_system')
        
        return core_system
    
    def _create_state_reward_generator(self) -> StateDerivedRewardGeneratorInterface:
        """Create the state-derived reward generator."""
        logger.debug("Creating state-derived reward generator...")
        
        # Extract configuration for state reward generator
        generator_config = {
            'coherence_weight': self.autonomous_reward_config.state_reward_coherence_weight,
            'growth_weight': self.autonomous_reward_config.state_reward_growth_weight,
            'integration_weight': self.autonomous_reward_config.state_reward_integration_weight,
            'elegance_weight': self.autonomous_reward_config.state_reward_elegance_weight,
            'emergence_weight': self.autonomous_reward_config.state_reward_emergence_weight,
            'coherence_threshold': self.autonomous_reward_config.coherence_threshold,
            'growth_smoothing': self.autonomous_reward_config.growth_smoothing,
            'integration_window': self.autonomous_reward_config.integration_window,
            'elegance_balance_point': self.autonomous_reward_config.elegance_balance_point,
            'pattern_detection_eps': self.autonomous_reward_config.pattern_detection_eps,
            'pattern_min_samples': self.autonomous_reward_config.pattern_min_samples,
            'pattern_history_size': self.autonomous_reward_config.pattern_history_size
        }
        
        generator = StateDerivedRewardGenerator(generator_config)
        
        self._created_components['state_reward_generator'] = generator
        self._initialization_order.append('state_reward_generator')
        
        return generator
    
    def _create_intrinsic_motivation_engine(self) -> IntrinsicMotivationEngineInterface:
        """Create the intrinsic motivation engine."""
        logger.debug("Creating intrinsic motivation engine...")
        
        # Extract configuration for intrinsic motivation engine
        motivation_config = {
            'curiosity_base_weight': self.autonomous_reward_config.curiosity_base_weight,
            'mastery_base_weight': self.autonomous_reward_config.mastery_base_weight,
            'autonomy_base_weight': self.autonomous_reward_config.autonomy_base_weight,
            'growth_base_weight': self.autonomous_reward_config.growth_base_weight,
            'curiosity_novelty_threshold': self.autonomous_reward_config.curiosity_novelty_threshold,
            'mastery_progress_threshold': self.autonomous_reward_config.mastery_progress_threshold,
            'autonomy_self_direction_bonus': self.autonomous_reward_config.autonomy_self_direction_bonus
        }
        
        engine = IntrinsicMotivationEngine(motivation_config)
        
        self._created_components['intrinsic_motivation_engine'] = engine
        self._initialization_order.append('intrinsic_motivation_engine')
        
        return engine
    
    def _create_progressive_learning_controller(self) -> ProgressiveLearningControllerInterface:
        """Create the progressive learning controller."""
        logger.debug("Creating progressive learning controller...")
        
        # ProgressiveLearningController takes the full AgentConfig
        controller = ProgressiveLearningController(self.config)
        
        self._created_components['progressive_learning_controller'] = controller
        self._initialization_order.append('progressive_learning_controller')
        
        return controller
    
    def _create_world_interaction_reward_system(self) -> WorldInteractionRewardSystemInterface:
        """Create the world interaction reward system."""
        logger.debug("Creating world interaction reward system...")
        
        # WorldInteractionRewardSystem doesn't take configuration parameters
        system = WorldInteractionRewardSystem()
        
        self._created_components['world_interaction_reward_system'] = system
        self._initialization_order.append('world_interaction_reward_system')
        
        return system
    
    def _create_emergent_value_system(self) -> EmergentValueSystemInterface:
        """Create the emergent value system."""
        logger.debug("Creating emergent value system...")
        
        # Extract configuration for emergent value system
        value_config = {
            'value_pattern_strength_threshold': self.autonomous_reward_config.value_pattern_strength_threshold,
            'value_integration_rate': self.autonomous_reward_config.value_integration_rate,
            'value_transfer_similarity_threshold': self.autonomous_reward_config.value_transfer_similarity_threshold,
            'autonomous_goal_generation_threshold': self.autonomous_reward_config.autonomous_goal_generation_threshold,
            'value_decay_rate': self.autonomous_reward_config.value_decay_rate
        }
        
        system = EmergentValueSystem(value_config)
        
        self._created_components['emergent_value_system'] = system
        self._initialization_order.append('emergent_value_system')
        
        return system
    
    def _create_mcp_integration_layer(self) -> MCPIntegrationLayerInterface:
        """Create the MCP integration layer."""
        logger.debug("Creating MCP integration layer...")
        
        # Create a mock ToolLayer for now (in real implementation, this would be injected)
        from .mcp_integration_layer_standalone import ToolLayer
        tool_layer = ToolLayer()
        
        # MCPIntegrationLayer takes AgentConfig and ToolLayer
        layer = MCPIntegrationLayer(self.config, tool_layer)
        
        self._created_components['mcp_integration_layer'] = layer
        self._initialization_order.append('mcp_integration_layer')
        
        if not MCP_AVAILABLE:
            logger.warning("Using standalone MCP integration layer (full MCP not available)")
        
        return layer
    
    def _create_self_success_metrics_system(self) -> SelfSuccessMetricsSystemInterface:
        """Create the self-success metrics system."""
        logger.debug("Creating self-success metrics system...")
        
        # Extract configuration for self-success metrics system
        metrics_config = {
            'harmony_index_weights': self.autonomous_reward_config.harmony_index_weights,
            'growth_trajectory_window': self.autonomous_reward_config.growth_trajectory_window,
            'autonomy_measurement_window': self.autonomous_reward_config.autonomy_measurement_window,
            'value_alignment_threshold': self.autonomous_reward_config.value_alignment_threshold
        }
        
        system = SelfSuccessMetricsSystem(metrics_config)
        
        self._created_components['self_success_metrics_system'] = system
        self._initialization_order.append('self_success_metrics_system')
        
        return system
    
    def _create_cross_layer_synergy(self) -> CrossLayerSynergyInterface:
        """Create the cross-layer synergy system."""
        logger.debug("Creating cross-layer synergy system...")
        
        # Extract configuration for cross-layer synergy system
        synergy_config = {
            'synergy_bonus_multiplier': self.autonomous_reward_config.synergy_bonus_multiplier,
            'layer_coordination_threshold': self.autonomous_reward_config.layer_coordination_threshold,
            'emergent_behavior_detection_threshold': self.autonomous_reward_config.emergent_behavior_detection_threshold,
            'feedback_loop_strength': self.autonomous_reward_config.feedback_loop_strength
        }
        
        system = CrossLayerSynergy(synergy_config)
        
        self._created_components['cross_layer_synergy'] = system
        self._initialization_order.append('cross_layer_synergy')
        
        return system
    
    def _create_adaptive_complexity_management(self) -> AdaptiveComplexityManagementInterface:
        """Create the adaptive complexity management system."""
        logger.debug("Creating adaptive complexity management system...")
        
        # Extract configuration for adaptive complexity management
        complexity_config = {
            'capability_assessment_window': self.autonomous_reward_config.capability_assessment_window,
            'complexity_increase_rate': self.autonomous_reward_config.complexity_increase_rate,
            'challenge_difficulty_range': self.autonomous_reward_config.challenge_difficulty_range,
            'complexity_navigation_reward_base': self.autonomous_reward_config.complexity_navigation_reward_base
        }
        
        system = AdaptiveComplexityManagement(complexity_config)
        
        self._created_components['adaptive_complexity_management'] = system
        self._initialization_order.append('adaptive_complexity_management')
        
        return system
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current status of the factory and created components.
        
        Returns:
            Dictionary containing factory status information
        """
        return {
            'is_initialized': self._is_initialized,
            'created_components': list(self._created_components.keys()),
            'initialization_order': self._initialization_order.copy(),
            'mcp_available': MCP_AVAILABLE,
            'config_valid': len(self.config.validate_configuration()) == 0,
            'autonomous_rewards_enabled': self.autonomous_reward_config.enable_autonomous_rewards,
            'creation_timestamp': datetime.now().isoformat()
        }
    
    def cleanup(self) -> None:
        """
        Clean up all created components in reverse initialization order.
        
        This method ensures proper cleanup of resources and prevents memory leaks.
        """
        logger.info("Cleaning up autonomous reward system components...")
        
        # Cleanup in reverse order of initialization
        for component_name in reversed(self._initialization_order):
            try:
                component = self._created_components.get(component_name)
                if component and hasattr(component, 'cleanup'):
                    logger.debug(f"Cleaning up {component_name}...")
                    component.cleanup()
                elif component and hasattr(component, 'close'):
                    logger.debug(f"Closing {component_name}...")
                    component.close()
            except Exception as e:
                logger.warning(f"Error cleaning up {component_name}: {e}")
        
        # Clear tracking structures
        self._created_components.clear()
        self._initialization_order.clear()
        self._is_initialized = False
        
        logger.info("Autonomous reward system cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


def create_autonomous_reward_system(config: AgentConfig) -> AutonomousRewardSystemInterface:
    """
    Convenience function to create an autonomous reward system.
    
    Args:
        config: Agent configuration
        
    Returns:
        Configured AutonomousRewardSystem instance
        
    Raises:
        RuntimeError: If system creation fails
    """
    factory = AutonomousRewardSystemFactory(config)
    return factory.create_autonomous_reward_system()


def create_autonomous_reward_system_with_factory(config: AgentConfig) -> tuple[AutonomousRewardSystemInterface, AutonomousRewardSystemFactory]:
    """
    Create an autonomous reward system and return both the system and factory.
    
    This is useful when you need access to the factory for cleanup or status monitoring.
    
    Args:
        config: Agent configuration
        
    Returns:
        Tuple of (AutonomousRewardSystem, AutonomousRewardSystemFactory)
        
    Raises:
        RuntimeError: If system creation fails
    """
    factory = AutonomousRewardSystemFactory(config)
    system = factory.create_autonomous_reward_system()
    return system, factory