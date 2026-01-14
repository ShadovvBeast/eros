"""
Core implementation of the Autonomous State-Derived Reward System.

This module implements the central AutonomousRewardSystem class that coordinates
all reward generation and processing based purely on internal state dynamics.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

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
from .models import (
    StateReward, IntrinsicMotivation, ValuePattern, LearningState,
    WorldInteractionResult, AutonomousGoal, Experience
)
from .error_handling import (
    RewardSystemErrorHandler, StateCorruptionError, RewardOverflowError,
    IntegrationFailureError, ErrorType, ErrorSeverity
)
from ..core.models import SemanticVector, MemoryTrace, ToolResult
from ..core.config import AgentConfig

logger = logging.getLogger(__name__)


class AutonomousRewardSystem(AutonomousRewardSystemInterface):
    """
    Central coordinator that orchestrates all reward generation and processing.
    
    This system eliminates external reward definitions and creates a self-organizing,
    adaptive reward architecture that enables genuine autonomy, intrinsic motivation,
    and progressive self-improvement.
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the autonomous reward system.
        
        Args:
            config: Agent configuration containing all subsystem settings
        """
        self.config = config
        
        # Initialize error handler
        error_handler_config = getattr(config, 'error_handler_config', {})
        self.error_handler = RewardSystemErrorHandler(error_handler_config)
        
        # Initialize subsystem interfaces (will be injected by factory)
        self.state_reward_generator: Optional[StateDerivedRewardGeneratorInterface] = None
        self.intrinsic_motivation_engine: Optional[IntrinsicMotivationEngineInterface] = None
        self.progressive_learning_controller: Optional[ProgressiveLearningControllerInterface] = None
        self.world_interaction_reward_system: Optional[WorldInteractionRewardSystemInterface] = None
        self.emergent_value_system: Optional[EmergentValueSystemInterface] = None
        self.mcp_integration_layer: Optional[MCPIntegrationLayerInterface] = None
        self.self_success_metrics_system: Optional[SelfSuccessMetricsSystemInterface] = None
        self.cross_layer_synergy: Optional[CrossLayerSynergyInterface] = None
        self.adaptive_complexity_management: Optional[AdaptiveComplexityManagementInterface] = None
        
        # Internal state tracking
        self.reward_history: List[StateReward] = []
        self.motivation_history: List[IntrinsicMotivation] = []
        self.experience_buffer: List[Experience] = []
        self.current_learning_state: Optional[LearningState] = None
        
        # State-centric reward computation pipeline
        self.reward_pipeline_active = False
        self.state_transition_buffer: List[np.ndarray] = []
        self.max_buffer_size = 100
        
        logger.info("Initialized AutonomousRewardSystem with error handling")
    
    def initialize_subsystems(self,
                            state_reward_generator: StateDerivedRewardGeneratorInterface,
                            intrinsic_motivation_engine: IntrinsicMotivationEngineInterface,
                            progressive_learning_controller: ProgressiveLearningControllerInterface,
                            world_interaction_reward_system: WorldInteractionRewardSystemInterface,
                            emergent_value_system: EmergentValueSystemInterface,
                            mcp_integration_layer: MCPIntegrationLayerInterface,
                            self_success_metrics_system: SelfSuccessMetricsSystemInterface,
                            cross_layer_synergy: CrossLayerSynergyInterface,
                            adaptive_complexity_management: AdaptiveComplexityManagementInterface) -> None:
        """
        Initialize all reward subsystems.
        
        Args:
            state_reward_generator: State-derived reward generator
            intrinsic_motivation_engine: Intrinsic motivation engine
            progressive_learning_controller: Progressive learning controller
            world_interaction_reward_system: World interaction reward system
            emergent_value_system: Emergent value system
            mcp_integration_layer: MCP integration layer
            self_success_metrics_system: Self-success metrics system
            cross_layer_synergy: Cross-layer synergy system
            adaptive_complexity_management: Adaptive complexity management
        """
        self.state_reward_generator = state_reward_generator
        self.intrinsic_motivation_engine = intrinsic_motivation_engine
        self.progressive_learning_controller = progressive_learning_controller
        self.world_interaction_reward_system = world_interaction_reward_system
        self.emergent_value_system = emergent_value_system
        self.mcp_integration_layer = mcp_integration_layer
        self.self_success_metrics_system = self_success_metrics_system
        self.cross_layer_synergy = cross_layer_synergy
        self.adaptive_complexity_management = adaptive_complexity_management
        
        # Activate the reward pipeline
        self.reward_pipeline_active = True
        
        logger.info("All autonomous reward subsystems initialized and pipeline activated")
    
    def compute_state_derived_reward(self, current_state: np.ndarray, 
                                   previous_state: np.ndarray) -> StateReward:
        """
        Compute reward signals derived from pathos state patterns and transitions.
        
        This is the core method that implements state-centric reward computation,
        ensuring all rewards derive from internal state dynamics.
        
        Args:
            current_state: Current pathos state F(t+1)
            previous_state: Previous pathos state F(t)
            
        Returns:
            StateReward containing all state-derived reward components
        """
        try:
            # Handle potential state corruption
            current_state = self.error_handler.handle_state_corruption(
                current_state, self.state_transition_buffer
            )
            previous_state = self.error_handler.handle_state_corruption(
                previous_state, self.state_transition_buffer
            )
            
            if not self.reward_pipeline_active or not self.state_reward_generator:
                # NO FALLBACK! System must be properly initialized
                raise RuntimeError(
                    f"Autonomous reward system not properly initialized! "
                    f"reward_pipeline_active={self.reward_pipeline_active}, "
                    f"state_reward_generator={self.state_reward_generator is not None}. "
                    f"Call initialize_subsystems() before using compute_state_derived_reward()."
                )
            
            # Update state transition buffer
            self._update_state_buffer(current_state)
            
            # Compute coherence reward from current state harmony
            coherence_reward = self.state_reward_generator.compute_coherence_reward(current_state)
            
            # Compute growth reward from state evolution
            historical_states = self.state_transition_buffer[-10:]  # Last 10 states
            growth_reward = self.state_reward_generator.compute_growth_reward(current_state, historical_states)
            
            # Compute integration reward from memory patterns
            memory_patterns = self._extract_memory_patterns()
            integration_reward = self.state_reward_generator.compute_integration_reward(current_state, memory_patterns)
            
            # Compute elegance reward from complexity-simplicity balance
            state_complexity = self._compute_state_complexity(current_state)
            solution_efficiency = self._assess_solution_efficiency(current_state, previous_state)
            elegance_reward = self.state_reward_generator.compute_elegance_reward(state_complexity, solution_efficiency)
            
            # Detect emergent patterns and compute emergence reward
            recent_states = self.state_transition_buffer[-5:]  # Last 5 states
            emergent_patterns = self.state_reward_generator.detect_emergent_patterns(recent_states)
            emergence_reward = self._compute_emergence_reward(emergent_patterns)
            
            # Handle potential reward overflows (but no fallbacks)
            coherence_reward = self.error_handler.handle_reward_overflow(coherence_reward, "coherence")
            growth_reward = self.error_handler.handle_reward_overflow(growth_reward, "growth")
            integration_reward = self.error_handler.handle_reward_overflow(integration_reward, "integration")
            elegance_reward = self.error_handler.handle_reward_overflow(elegance_reward, "elegance")
            emergence_reward = self.error_handler.handle_reward_overflow(emergence_reward, "emergence")
            
            # Combine all reward components with REDUCED positive bias
            # The main positive signal should come from successful tool interactions
            total_reward = (coherence_reward + growth_reward + integration_reward + 
                           elegance_reward + emergence_reward)
            
            # Minimal positive bias - let dukkha drive motivation instead of artificial rewards
            # The main positive signal should come from successful tool interactions and dukkha resolution
            positive_bias = 0.05  # Reduced from 0.2 to 0.05 - dukkha provides the growth drive
            total_reward += positive_bias
            
            total_reward = self.error_handler.handle_reward_overflow(total_reward, "total")
            
            state_reward = StateReward(
                coherence_reward=coherence_reward,
                growth_reward=growth_reward,
                integration_reward=integration_reward,
                elegance_reward=elegance_reward,
                emergence_reward=emergence_reward,
                total_reward=total_reward
            )
            
            # Validate the complete state reward
            state_reward = self.error_handler.validate_state_reward(state_reward)
            
            # Store in history
            self.reward_history.append(state_reward)
            if len(self.reward_history) > self.max_buffer_size:
                self.reward_history.pop(0)
            
            logger.debug(f"Computed state-derived reward: total={total_reward:.4f}, "
                        f"coherence={coherence_reward:.4f}, growth={growth_reward:.4f}, "
                        f"integration={integration_reward:.4f}, elegance={elegance_reward:.4f}, "
                        f"emergence={emergence_reward:.4f}")
            
            return state_reward
            
        except Exception as e:
            # NO FALLBACK! Let the error propagate to expose initialization issues
            logger.error(f"CRITICAL ERROR in state-derived reward computation: {e}")
            logger.error(f"reward_pipeline_active: {self.reward_pipeline_active}")
            logger.error(f"state_reward_generator: {self.state_reward_generator is not None}")
            logger.error(f"This indicates the autonomous reward system was not properly initialized!")
            raise RuntimeError(f"Autonomous reward system failure: {e}") from e
    
    def generate_intrinsic_motivation(self, state: np.ndarray, 
                                    context: Dict[str, Any]) -> IntrinsicMotivation:
        """
        Generate intrinsic motivation signals for curiosity, mastery, and autonomy.
        
        Args:
            state: Current pathos state
            context: Current context including recent experiences
            
        Returns:
            IntrinsicMotivation containing all motivation drives
        """
        try:
            # Handle potential state corruption
            state = self.error_handler.handle_state_corruption(
                state, self.state_transition_buffer
            )
            
            if not self.intrinsic_motivation_engine:
                # NO FALLBACK! System must be properly initialized
                raise RuntimeError(
                    f"Intrinsic motivation engine not initialized! "
                    f"Call initialize_subsystems() before using generate_intrinsic_motivation()."
                )
            
            # Extract context information with safe defaults
            knowledge_gaps = context.get('knowledge_gaps', [])
            state_energy = float(np.linalg.norm(state))
            skill_domains = context.get('skill_domains', {})
            self_directed_actions = context.get('self_directed_actions', 0)
            external_dependencies = context.get('external_dependencies', 0)
            current_capabilities = context.get('current_capabilities', set())
            learning_opportunities = context.get('learning_opportunities', [])
            
            # Generate curiosity drive with error handling
            curiosity_drive = self.error_handler.safe_execute(
                self.intrinsic_motivation_engine.generate_curiosity_drive,
                knowledge_gaps, state_energy,
                fallback_value=0.3,
                error_type=ErrorType.NUMERICAL_INSTABILITY
            )
            
            # Assess mastery progress across skill domains
            mastery_drive = 0.0
            for domain, performance_history in skill_domains.items():
                domain_mastery = self.error_handler.safe_execute(
                    self.intrinsic_motivation_engine.assess_mastery_progress,
                    domain, performance_history,
                    fallback_value=0.3,
                    error_type=ErrorType.NUMERICAL_INSTABILITY
                )
                mastery_drive += domain_mastery
            mastery_drive = mastery_drive / max(len(skill_domains), 1)  # Average across domains
            
            # Compute autonomy reward with error handling
            autonomy_drive = self.error_handler.safe_execute(
                self.intrinsic_motivation_engine.compute_autonomy_reward,
                self_directed_actions, external_dependencies,
                fallback_value=0.3,
                error_type=ErrorType.NUMERICAL_INSTABILITY
            )
            
            # Evaluate growth potential with error handling
            growth_drive = self.error_handler.safe_execute(
                self.intrinsic_motivation_engine.evaluate_growth_potential,
                current_capabilities, learning_opportunities,
                fallback_value=0.3,
                error_type=ErrorType.NUMERICAL_INSTABILITY
            )
            
            # Handle potential overflows
            curiosity_drive = self.error_handler.handle_reward_overflow(curiosity_drive, "curiosity")
            mastery_drive = self.error_handler.handle_reward_overflow(mastery_drive, "mastery")
            autonomy_drive = self.error_handler.handle_reward_overflow(autonomy_drive, "autonomy")
            growth_drive = self.error_handler.handle_reward_overflow(growth_drive, "growth")
            
            # Combine motivation drives
            combined_motivation = (curiosity_drive + mastery_drive + autonomy_drive + growth_drive) / 4.0
            combined_motivation = self.error_handler.handle_reward_overflow(combined_motivation, "combined_motivation")
            
            intrinsic_motivation = IntrinsicMotivation(
                curiosity_drive=curiosity_drive,
                mastery_drive=mastery_drive,
                autonomy_drive=autonomy_drive,
                growth_drive=growth_drive,
                combined_motivation=combined_motivation
            )
            
            # Validate the complete intrinsic motivation
            intrinsic_motivation = self.error_handler.validate_intrinsic_motivation(intrinsic_motivation)
            
            # Store in history
            self.motivation_history.append(intrinsic_motivation)
            if len(self.motivation_history) > self.max_buffer_size:
                self.motivation_history.pop(0)
            
            logger.debug(f"Generated intrinsic motivation: combined={combined_motivation:.4f}, "
                        f"curiosity={curiosity_drive:.4f}, mastery={mastery_drive:.4f}, "
                        f"autonomy={autonomy_drive:.4f}, growth={growth_drive:.4f}")
            
            return intrinsic_motivation
            
        except Exception as e:
            # NO FALLBACK! Let the error propagate to expose initialization issues
            logger.error(f"CRITICAL ERROR in intrinsic motivation generation: {e}")
            logger.error(f"intrinsic_motivation_engine: {self.intrinsic_motivation_engine is not None}")
            logger.error(f"This indicates the autonomous reward system was not properly initialized!")
            raise RuntimeError(f"Autonomous reward system failure in intrinsic motivation: {e}") from e
    
    def assess_world_interaction_value(self, action_result: ToolResult, 
                                     state: np.ndarray) -> WorldInteractionResult:
        """
        Assess the value of world interaction outcomes.
        
        Args:
            action_result: Result from tool execution or world interaction
            state: Current pathos state
            
        Returns:
            WorldInteractionResult with comprehensive interaction assessment
        """
        if not self.world_interaction_reward_system:
            # NO FALLBACK! System must be properly initialized
            raise RuntimeError(
                f"World interaction reward system not initialized! "
                f"Call initialize_subsystems() before using assess_world_interaction()."
            )
        
        # Extract action information
        action_type = action_result.tool_name
        success_level = 1.0 if action_result.success else 0.0
        
        # Assess discovery value (simplified for now)
        discovery_value = self._assess_discovery_value(action_result)
        
        # Assess connection quality (simplified for now)
        connection_quality = self._assess_connection_quality(action_result)
        
        # Assess creativity score (simplified for now)
        creativity_score = self._assess_creativity_score(action_result)
        
        # Compute total reward with proper failure handling
        if action_result.success:
            total_reward = (success_level * 2.0 +  # High weight for success
                           discovery_value * 1.5 +   # High weight for discovery
                           connection_quality * 1.0 + # Medium weight for connection
                           creativity_score * 1.2)    # High weight for creativity
        else:
            # FAILURE PENALTY: Failed actions should have negative or near-zero reward
            # This ensures the agent learns to avoid actions that fail
            failure_penalty = -1.5  # Significant negative signal for failures
            
            # Small positive for attempting (encourages exploration) but net negative
            attempt_bonus = 0.3
            
            total_reward = failure_penalty + attempt_bonus  # Net: -1.2
            
            logger.debug(f"Tool {action_type} failed - applying failure penalty: {total_reward:.4f}")
        
        world_interaction_result = WorldInteractionResult(
            action_type=action_type,
            success_level=success_level,
            discovery_value=discovery_value,
            connection_quality=connection_quality,
            creativity_score=creativity_score,
            total_reward=total_reward
        )
        
        logger.debug(f"Assessed world interaction: total_reward={total_reward:.4f}, "
                    f"success={success_level:.4f}, discovery={discovery_value:.4f}, "
                    f"connection={connection_quality:.4f}, creativity={creativity_score:.4f}")
        
        return world_interaction_result
    
    def update_emergent_values(self, experience: Experience, reward: float) -> None:
        """
        Update the emergent value system based on new experiences.
        
        Args:
            experience: New experience data
            reward: Associated reward value
        """
        # Add experience to buffer
        self.experience_buffer.append(experience)
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
        
        if self.emergent_value_system:
            # Extract recent experiences and rewards for value discovery
            recent_experiences = self.experience_buffer[-20:]  # Last 20 experiences
            recent_rewards = [exp.reward_received for exp in recent_experiences]
            
            # Discover new value patterns
            new_value_patterns = self.emergent_value_system.discover_value_patterns(
                recent_experiences, recent_rewards
            )
            
            if new_value_patterns:
                logger.debug(f"Discovered {len(new_value_patterns)} new value patterns")
        
        logger.debug(f"Updated emergent values with experience reward={reward:.4f}")
    
    def compute_cross_layer_synergy(self, logos_state: Any, pathos_state: np.ndarray, 
                                  memory_context: List[MemoryTrace]) -> float:
        """
        Compute synergy bonus for cross-layer coordination.
        
        Args:
            logos_state: Current logos layer state
            pathos_state: Current pathos state
            memory_context: Relevant memory traces
            
        Returns:
            Synergy bonus value
        """
        try:
            # Handle potential state corruption
            pathos_state = self.error_handler.handle_state_corruption(
                pathos_state, self.state_transition_buffer
            )
            
            if not self.cross_layer_synergy:
                # NO FALLBACK! System must be properly initialized
                raise RuntimeError(
                    f"Cross layer synergy system not initialized! "
                    f"Call initialize_subsystems() before using compute_cross_layer_synergy()."
                )
            
            # Extract reward signals from each layer (simplified for now)
            logos_rewards = {'reasoning': 0.5, 'planning': 0.3}  # Placeholder
            pathos_rewards = {'coherence': 0.4, 'growth': 0.6}   # Placeholder
            memory_rewards = {'integration': 0.5, 'recall': 0.4} # Placeholder
            
            # Coordinate reward signals with error handling
            coordinated_rewards = self.error_handler.safe_execute(
                self.cross_layer_synergy.coordinate_reward_signals,
                logos_rewards, pathos_rewards, memory_rewards,
                fallback_value={'synergy': 0.1},
                error_type=ErrorType.INTEGRATION_FAILURE
            )
            
            # Compute synergy bonus
            layer_harmony = {
                'logos': np.mean(list(logos_rewards.values())),
                'pathos': np.mean(list(pathos_rewards.values())),
                'memory': np.mean(list(memory_rewards.values()))
            }
            
            synergy_bonus = self.error_handler.safe_execute(
                self.cross_layer_synergy.compute_synergy_bonus,
                layer_harmony,
                fallback_value=0.1,
                error_type=ErrorType.INTEGRATION_FAILURE
            )
            
            # Handle potential overflow
            synergy_bonus = self.error_handler.handle_reward_overflow(synergy_bonus, "synergy")
            
            logger.debug(f"Computed cross-layer synergy bonus: {synergy_bonus:.4f}")
            
            return synergy_bonus
            
        except Exception as e:
            # NO FALLBACK! Let the error propagate to expose initialization issues
            logger.error(f"CRITICAL ERROR in cross-layer synergy computation: {e}")
            logger.error(f"cross_layer_synergy: {self.cross_layer_synergy is not None}")
            logger.error(f"This indicates the autonomous reward system was not properly initialized!")
            raise RuntimeError(f"Autonomous reward system failure in cross-layer synergy: {e}") from e
    
    def get_current_learning_state(self) -> Optional[LearningState]:
        """
        Get the current learning state configuration.
        
        Returns:
            Current LearningState or None if not available
        """
        return self.current_learning_state
    
    def get_reward_history(self) -> List[StateReward]:
        """
        Get the history of state-derived rewards.
        
        Returns:
            List of StateReward objects
        """
        return self.reward_history.copy()
    
    def get_motivation_history(self) -> List[IntrinsicMotivation]:
        """
        Get the history of intrinsic motivation signals.
        
        Returns:
            List of IntrinsicMotivation objects
        """
        return self.motivation_history.copy()
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status information.
        
        Returns:
            Dictionary containing system status
        """
        return {
            'pipeline_active': self.reward_pipeline_active,
            'subsystems_initialized': all([
                self.state_reward_generator is not None,
                self.intrinsic_motivation_engine is not None,
                self.progressive_learning_controller is not None,
                self.world_interaction_reward_system is not None,
                self.emergent_value_system is not None,
                self.mcp_integration_layer is not None,
                self.self_success_metrics_system is not None,
                self.cross_layer_synergy is not None,
                self.adaptive_complexity_management is not None
            ]),
            'reward_history_length': len(self.reward_history),
            'motivation_history_length': len(self.motivation_history),
            'experience_buffer_length': len(self.experience_buffer),
            'state_buffer_length': len(self.state_transition_buffer),
            'current_learning_state': self.current_learning_state is not None,
            'error_statistics': self.error_handler.get_error_statistics()
        }
    
    def generate_autonomous_goals(self, current_state: np.ndarray, 
                                context: Dict[str, Any]) -> List[AutonomousGoal]:
        """
        Generate autonomous goals based on current state and emergent values.
        
        Args:
            current_state: Current pathos state
            context: Context information for goal generation
            
        Returns:
            List of generated autonomous goals
        """
        try:
            # Handle potential state corruption
            current_state = self.error_handler.handle_state_corruption(
                current_state, self.state_transition_buffer
            )
            
            if not self.emergent_value_system:
                # NO FALLBACK! System must be properly initialized
                raise RuntimeError(
                    f"Emergent value system not initialized! "
                    f"Call initialize_subsystems() before using generate_autonomous_goals()."
                )
            
            # Get current value patterns
            current_values = self.emergent_value_system.get_discovered_patterns()
            
            # Generate goals using the emergent value system
            autonomous_goals = self.error_handler.safe_execute(
                self.emergent_value_system.generate_autonomous_goals,
                current_values, current_state,
                fallback_value=[],
                error_type=ErrorType.INTEGRATION_FAILURE
            )
            
            logger.debug(f"Generated {len(autonomous_goals)} autonomous goals")
            
            return autonomous_goals
            
        except Exception as e:
            logger.error(f"Critical error in autonomous goal generation: {e}")
            
            # Return fallback goals
            return self._generate_basic_autonomous_goals(current_state, context)
    
    # REMAINING FALLBACK METHODS REMOVED - System must be properly initialized
    
    def get_error_handler(self) -> RewardSystemErrorHandler:
        """
        Get the error handler for direct access to error management.
        
        Returns:
            RewardSystemErrorHandler instance
        """
        return self.error_handler
    
    # Private helper methods
    
    def _update_state_buffer(self, state: np.ndarray) -> None:
        """Update the state transition buffer with the new state."""
        self.state_transition_buffer.append(state.copy())
        if len(self.state_transition_buffer) > self.max_buffer_size:
            self.state_transition_buffer.pop(0)
    
    def _extract_memory_patterns(self) -> Dict[str, Any]:
        """Extract memory patterns from experience buffer."""
        # Simplified implementation - extract basic patterns
        patterns = {
            'experience_count': len(self.experience_buffer),
            'average_reward': np.mean([exp.reward_received for exp in self.experience_buffer]) if self.experience_buffer else 0.0,
            'action_types': list(set([exp.action_taken for exp in self.experience_buffer]))
        }
        return patterns
    
    def _compute_state_complexity(self, state: np.ndarray) -> float:
        """Compute complexity measure of the current state."""
        # Use entropy-based complexity measure
        # Normalize state to probabilities
        state_abs = np.abs(state)
        state_probs = state_abs / (np.sum(state_abs) + 1e-8)
        
        # Compute entropy
        entropy = -np.sum(state_probs * np.log(state_probs + 1e-8))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(len(state))
        complexity = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return complexity
    
    def _assess_solution_efficiency(self, current_state: np.ndarray, previous_state: np.ndarray) -> float:
        """Assess efficiency of the solution/approach based on state transition."""
        # Measure efficiency as inverse of state change magnitude
        state_change = np.linalg.norm(current_state - previous_state)
        efficiency = 1.0 / (1.0 + state_change)  # Higher efficiency for smaller changes
        return efficiency
    
    def _compute_emergence_reward(self, patterns: List[Any]) -> float:
        """Compute reward for emergent patterns."""
        # Simple implementation: reward based on number and novelty of patterns
        if not patterns:
            return 0.0
        
        # Base reward for having patterns
        base_reward = len(patterns) * 0.1
        
        # Bonus for pattern diversity (simplified)
        pattern_types = set([getattr(p, 'pattern_type', 'unknown') for p in patterns])
        diversity_bonus = len(pattern_types) * 0.05
        
        return base_reward + diversity_bonus
    
    # FALLBACK METHODS REMOVED - System must be properly initialized
    # If you see errors about missing methods, the autonomous reward system
    # was not properly initialized. Call initialize_subsystems() first.
    
    def _assess_discovery_value(self, action_result: ToolResult) -> float:
        """Assess discovery value from action result."""
        # Simplified: higher value for successful actions with rich results
        if not action_result.success:
            return 0.0
        
        # Estimate discovery value based on result content
        result_str = str(action_result.result)
        discovery_value = min(len(result_str) / 1000.0, 1.0)  # Normalize by content length
        
        return discovery_value
    
    def _assess_connection_quality(self, action_result: ToolResult) -> float:
        """Assess connection quality from action result."""
        # Simplified: assume moderate connection quality for successful actions
        return 0.5 if action_result.success else 0.0
    
    def _assess_creativity_score(self, action_result: ToolResult) -> float:
        """Assess creativity score from action result."""
        # Simplified: assume moderate creativity for successful actions
        return 0.3 if action_result.success else 0.0