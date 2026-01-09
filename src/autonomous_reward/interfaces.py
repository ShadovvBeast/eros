"""
Interfaces for the Autonomous State-Derived Reward System.

This module defines the abstract interfaces for all components of the autonomous
reward system, ensuring proper separation of concerns and testability.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np

from .models import (
    StateReward, IntrinsicMotivation, ValuePattern, LearningState,
    WorldInteractionResult, AutonomousGoal, Experience, Pattern
)
from ..core.models import SemanticVector, MemoryTrace, ToolResult, Intention


class AutonomousRewardSystemInterface(ABC):
    """Central coordinator that orchestrates all reward generation and processing"""
    
    @abstractmethod
    def compute_state_derived_reward(self, current_state: np.ndarray, 
                                   previous_state: np.ndarray) -> StateReward:
        """
        Compute reward signals derived from pathos state patterns and transitions.
        
        Args:
            current_state: Current pathos state F(t+1)
            previous_state: Previous pathos state F(t)
            
        Returns:
            StateReward containing all state-derived reward components
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def update_emergent_values(self, experience: Experience, 
                             reward: float) -> None:
        """
        Update the emergent value system based on new experiences.
        
        Args:
            experience: New experience data
            reward: Associated reward value
        """
        pass
    
    @abstractmethod
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
        pass


class StateDerivedRewardGeneratorInterface(ABC):
    """Generates reward signals directly from pathos state patterns and transitions"""
    
    @abstractmethod
    def compute_coherence_reward(self, state: np.ndarray) -> float:
        """
        Compute reward for achieving internal state harmony.
        
        Args:
            state: Current pathos state
            
        Returns:
            Coherence reward value
        """
        pass
    
    @abstractmethod
    def compute_growth_reward(self, current_state: np.ndarray, 
                            historical_states: List[np.ndarray]) -> float:
        """
        Compute reward for healthy state evolution and expansion.
        
        Args:
            current_state: Current pathos state
            historical_states: Recent historical states
            
        Returns:
            Growth reward value
        """
        pass
    
    @abstractmethod
    def compute_integration_reward(self, state: np.ndarray, 
                                 memory_patterns: Dict[str, Any]) -> float:
        """
        Compute reward for connecting disparate state patterns.
        
        Args:
            state: Current pathos state
            memory_patterns: Relevant memory patterns
            
        Returns:
            Integration reward value
        """
        pass
    
    @abstractmethod
    def compute_elegance_reward(self, state_complexity: float, 
                              solution_efficiency: float) -> float:
        """
        Compute reward for optimal complexity-simplicity balance.
        
        Args:
            state_complexity: Measured complexity of current state
            solution_efficiency: Efficiency of current solution/approach
            
        Returns:
            Elegance reward value
        """
        pass
    
    @abstractmethod
    def detect_emergent_patterns(self, state_sequence: List[np.ndarray]) -> List[Pattern]:
        """
        Detect novel state configurations and emergent behaviors.
        
        Args:
            state_sequence: Sequence of recent states
            
        Returns:
            List of detected emergent patterns
        """
        pass


class IntrinsicMotivationEngineInterface(ABC):
    """Generates self-directed motivation for curiosity, mastery, and autonomy"""
    
    @abstractmethod
    def generate_curiosity_drive(self, knowledge_gaps: List[str], 
                               state_energy: float) -> float:
        """
        Generate curiosity drive for exploring unknown domains.
        
        Args:
            knowledge_gaps: Identified areas of unknown knowledge
            state_energy: Current energy level from pathos state
            
        Returns:
            Curiosity drive strength
        """
        pass
    
    @abstractmethod
    def assess_mastery_progress(self, skill_domain: str, 
                              performance_history: List[float]) -> float:
        """
        Assess progress toward mastery in a skill domain.
        
        Args:
            skill_domain: Domain of skill being developed
            performance_history: Historical performance data
            
        Returns:
            Mastery progress reward
        """
        pass
    
    @abstractmethod
    def compute_autonomy_reward(self, self_directed_actions: int, 
                              external_dependencies: int) -> float:
        """
        Compute reward for self-direction and independence.
        
        Args:
            self_directed_actions: Number of autonomous actions taken
            external_dependencies: Number of external dependencies
            
        Returns:
            Autonomy reward value
        """
        pass
    
    @abstractmethod
    def evaluate_growth_potential(self, current_capabilities: Set[str], 
                                learning_opportunities: List[str]) -> float:
        """
        Evaluate potential for capability expansion and learning.
        
        Args:
            current_capabilities: Set of current capabilities
            learning_opportunities: Available learning opportunities
            
        Returns:
            Growth potential score
        """
        pass


class ProgressiveLearningControllerInterface(ABC):
    """Adapts learning pace and complexity based on internal state readiness"""
    
    @abstractmethod
    def compute_adaptive_learning_rate(self, state_confidence: float, 
                                     recent_performance: List[float]) -> float:
        """
        Compute adaptive learning rate based on state confidence.
        
        Args:
            state_confidence: Confidence level from pathos state
            recent_performance: Recent learning performance data
            
        Returns:
            Adaptive learning rate
        """
        pass
    
    @abstractmethod
    def assess_complexity_readiness(self, current_state: np.ndarray, 
                                  capability_level: float) -> float:
        """
        Assess readiness for increased complexity challenges.
        
        Args:
            current_state: Current pathos state
            capability_level: Current capability assessment
            
        Returns:
            Complexity readiness score
        """
        pass
    
    @abstractmethod
    def update_meta_learning_parameters(self, learning_outcomes: List[Dict[str, Any]]) -> None:
        """
        Update meta-learning parameters based on learning outcomes.
        
        Args:
            learning_outcomes: Recent learning outcome data
        """
        pass
    
    @abstractmethod
    def detect_consolidation_need(self, recent_experiences: List[Experience]) -> bool:
        """
        Detect when consolidation time is needed for integration.
        
        Args:
            recent_experiences: Recent experience data
            
        Returns:
            True if consolidation is needed, False otherwise
        """
        pass


class WorldInteractionRewardSystemInterface(ABC):
    """Provides maximum rewards for meaningful world engagement"""
    
    @abstractmethod
    def evaluate_task_completion(self, task: Dict[str, Any], outcome: Dict[str, Any], 
                               effort_invested: float) -> float:
        """
        Evaluate reward for task completion.
        
        Args:
            task: Task description and requirements
            outcome: Task completion outcome
            effort_invested: Amount of effort invested
            
        Returns:
            Task completion reward
        """
        pass
    
    @abstractmethod
    def assess_discovery_value(self, new_knowledge: Dict[str, Any], 
                             existing_knowledge: Dict[str, Any]) -> float:
        """
        Assess value of new discoveries and learning.
        
        Args:
            new_knowledge: Newly acquired knowledge
            existing_knowledge: Existing knowledge base
            
        Returns:
            Discovery value reward
        """
        pass
    
    @abstractmethod
    def compute_connection_reward(self, interaction: Dict[str, Any], 
                                authenticity_score: float) -> float:
        """
        Compute reward for meaningful connections with other beings.
        
        Args:
            interaction: Interaction details
            authenticity_score: Authenticity assessment of the interaction
            
        Returns:
            Connection reward value
        """
        pass
    
    @abstractmethod
    def evaluate_creative_synthesis(self, solution: Dict[str, Any], 
                                  novelty_score: float) -> float:
        """
        Evaluate reward for creative synthesis and novel solutions.
        
        Args:
            solution: Solution details
            novelty_score: Novelty assessment of the solution
            
        Returns:
            Creative synthesis reward
        """
        pass


class EmergentValueSystemInterface(ABC):
    """Develops authentic value system through experience and state dynamics"""
    
    @abstractmethod
    def discover_value_patterns(self, experiences: List[Experience], 
                              rewards: List[float]) -> List[ValuePattern]:
        """
        Discover emerging value patterns from experiences.
        
        Args:
            experiences: List of experiences
            rewards: Associated reward values
            
        Returns:
            List of discovered value patterns
        """
        pass
    
    @abstractmethod
    def integrate_new_values(self, new_values: List[ValuePattern], 
                           existing_values: List[ValuePattern]) -> List[ValuePattern]:
        """
        Integrate new values into existing value system.
        
        Args:
            new_values: Newly discovered values
            existing_values: Existing value system
            
        Returns:
            Updated value system
        """
        pass
    
    @abstractmethod
    def transfer_values_across_contexts(self, source_context: str, 
                                      target_context: str) -> List[ValuePattern]:
        """
        Transfer learned values across different contexts.
        
        Args:
            source_context: Source context for value transfer
            target_context: Target context for value application
            
        Returns:
            Transferred value patterns
        """
        pass
    
    @abstractmethod
    def generate_autonomous_goals(self, current_values: List[ValuePattern], 
                                current_state: np.ndarray) -> List[AutonomousGoal]:
        """
        Generate autonomous goals based on developed values.
        
        Args:
            current_values: Current value system
            current_state: Current pathos state
            
        Returns:
            List of generated autonomous goals
        """
        pass


class MCPIntegrationLayerInterface(ABC):
    """Provides seamless plug-and-play integration with external capabilities"""
    
    @abstractmethod
    def discover_mcp_servers(self) -> List[Dict[str, Any]]:
        """
        Discover available MCP servers.
        
        Returns:
            List of discovered MCP server information
        """
        pass
    
    @abstractmethod
    def select_optimal_tool(self, goal: AutonomousGoal, available_tools: List[Dict[str, Any]], 
                          current_state: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Select optimal tool based on current state and goals.
        
        Args:
            goal: Current autonomous goal
            available_tools: List of available tools
            current_state: Current pathos state
            
        Returns:
            Selected tool information or None
        """
        pass
    
    @abstractmethod
    def execute_tool_with_feedback(self, tool: Dict[str, Any], 
                                 parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute tool and provide rich feedback.
        
        Args:
            tool: Tool to execute
            parameters: Tool parameters
            
        Returns:
            Tool execution result with feedback
        """
        pass
    
    @abstractmethod
    def learn_tool_effectiveness(self, tool: Dict[str, Any], context: Dict[str, Any], 
                               outcome: Dict[str, Any]) -> None:
        """
        Learn tool effectiveness through experience.
        
        Args:
            tool: Tool that was used
            context: Context of tool usage
            outcome: Outcome of tool usage
        """
        pass


class SelfSuccessMetricsSystemInterface(ABC):
    """Measures success based on internal state satisfaction and growth"""
    
    @abstractmethod
    def compute_state_harmony_index(self, state: np.ndarray, 
                                  harmony_patterns: List[Pattern]) -> float:
        """
        Compute state harmony index.
        
        Args:
            state: Current pathos state
            harmony_patterns: Known harmony patterns
            
        Returns:
            State harmony index
        """
        pass
    
    @abstractmethod
    def track_growth_trajectory(self, historical_capabilities: List[Set[str]], 
                              current_capabilities: Set[str]) -> Dict[str, float]:
        """
        Track growth trajectory over time.
        
        Args:
            historical_capabilities: Historical capability sets
            current_capabilities: Current capability set
            
        Returns:
            Growth trajectory metrics
        """
        pass
    
    @abstractmethod
    def measure_autonomy_level(self, decision_history: List[Dict[str, Any]], 
                             external_influence: float) -> float:
        """
        Measure current autonomy level.
        
        Args:
            decision_history: History of decisions made
            external_influence: Level of external influence
            
        Returns:
            Autonomy level measurement
        """
        pass
    
    @abstractmethod
    def assess_value_alignment(self, actions: List[Dict[str, Any]], 
                             values: List[ValuePattern]) -> float:
        """
        Assess alignment between actions and developed values.
        
        Args:
            actions: Recent actions taken
            values: Current value system
            
        Returns:
            Value alignment score
        """
        pass


class CrossLayerSynergyInterface(ABC):
    """Creates synergy between Logos reasoning, Pathos affective dynamics, and Memory integration"""
    
    @abstractmethod
    def coordinate_reward_signals(self, logos_rewards: Dict[str, float], 
                                pathos_rewards: Dict[str, float], 
                                memory_rewards: Dict[str, float]) -> Dict[str, float]:
        """
        Coordinate reward signals across all layers.
        
        Args:
            logos_rewards: Rewards from logos layer
            pathos_rewards: Rewards from pathos layer
            memory_rewards: Rewards from memory layer
            
        Returns:
            Coordinated reward signals
        """
        pass
    
    @abstractmethod
    def compute_synergy_bonus(self, layer_harmony: Dict[str, float]) -> float:
        """
        Compute synergy bonus for harmonious layer operation.
        
        Args:
            layer_harmony: Harmony metrics for each layer
            
        Returns:
            Synergy bonus value
        """
        pass
    
    @abstractmethod
    def detect_emergent_behaviors(self, cross_layer_patterns: List[Pattern]) -> List[Pattern]:
        """
        Detect emergent behaviors from cross-layer patterns.
        
        Args:
            cross_layer_patterns: Patterns detected across layers
            
        Returns:
            List of emergent behavior patterns
        """
        pass
    
    @abstractmethod
    def create_feedback_loops(self, layer_states: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create feedback loops that strengthen inter-layer cooperation.
        
        Args:
            layer_states: Current states of all layers
            
        Returns:
            Feedback loop adjustments for each layer
        """
        pass


class AdaptiveComplexityManagementInterface(ABC):
    """Progressively takes on more complex challenges as capabilities grow"""
    
    @abstractmethod
    def assess_capability_levels(self, state_patterns: List[Pattern]) -> Dict[str, float]:
        """
        Assess current capability levels from state patterns.
        
        Args:
            state_patterns: Current state patterns
            
        Returns:
            Capability level assessments by domain
        """
        pass
    
    @abstractmethod
    def scale_complexity_challenges(self, capabilities: Dict[str, float]) -> Dict[str, float]:
        """
        Scale complexity challenges based on capabilities.
        
        Args:
            capabilities: Current capability levels
            
        Returns:
            Appropriate complexity levels for each domain
        """
        pass
    
    @abstractmethod
    def adjust_challenge_difficulty(self, performance_history: List[Dict[str, Any]], 
                                  current_readiness: float) -> float:
        """
        Adjust challenge difficulty based on performance and readiness.
        
        Args:
            performance_history: Historical performance data
            current_readiness: Current readiness assessment
            
        Returns:
            Adjusted difficulty level
        """
        pass
    
    @abstractmethod
    def reward_complexity_navigation(self, complexity_handled: float, 
                                   success_rate: float) -> float:
        """
        Reward successful navigation of complexity.
        
        Args:
            complexity_handled: Level of complexity successfully handled
            success_rate: Success rate in handling complexity
            
        Returns:
            Complexity navigation reward
        """
        pass