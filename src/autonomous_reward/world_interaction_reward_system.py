"""
World Interaction Reward System for the Autonomous State-Derived Reward System.

This module implements the WorldInteractionRewardSystem that provides maximum rewards
for meaningful world engagement including task completion, discovery, connection,
and creative synthesis.
"""

import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from .interfaces import WorldInteractionRewardSystemInterface
from .models import WorldInteractionResult
from ..core.models import ToolResult


logger = logging.getLogger(__name__)


class WorldInteractionRewardSystem(WorldInteractionRewardSystemInterface):
    """
    Provides maximum rewards for meaningful world engagement.
    
    This system evaluates and rewards:
    - Task completion with high positive rewards
    - Discovery of new methods and concepts
    - Meaningful connections with other beings
    - Creative synthesis and novel solutions
    - Problem-solving achievements
    """
    
    def __init__(self):
        """Initialize the world interaction reward system."""
        self.task_completion_history: List[Dict[str, Any]] = []
        self.discovery_history: List[Dict[str, Any]] = []
        self.connection_history: List[Dict[str, Any]] = []
        self.creativity_history: List[Dict[str, Any]] = []
        
        # Reward scaling factors
        self.task_completion_scale = 10.0  # High rewards for task completion
        self.discovery_scale = 15.0        # Very high rewards for discovery
        self.connection_scale = 8.0        # High rewards for meaningful connections
        self.creativity_scale = 12.0       # Very high rewards for creativity
        
        # Minimum reward thresholds
        self.min_task_reward = 0.1
        self.min_discovery_reward = 0.5
        self.min_connection_reward = 0.1
        self.min_creativity_reward = 0.2
        
        logger.info("WorldInteractionRewardSystem initialized")
    
    def evaluate_task_completion(self, task: Dict[str, Any], outcome: Dict[str, Any], 
                               effort_invested: float) -> float:
        """
        Evaluate reward for task completion.
        
        Args:
            task: Task description and requirements
            outcome: Task completion outcome
            effort_invested: Amount of effort invested (0.0 to 1.0)
            
        Returns:
            Task completion reward (higher for successful completion)
        """
        try:
            # Extract task characteristics
            task_complexity = task.get('complexity', 0.5)
            task_importance = task.get('importance', 0.5)
            task_type = task.get('type', 'general')
            
            # Extract outcome characteristics
            success_level = outcome.get('success', 0.0)  # 0.0 to 1.0
            quality_score = outcome.get('quality', 0.5)  # 0.0 to 1.0
            completion_time = outcome.get('completion_time', 1.0)
            errors_encountered = outcome.get('errors', 0)
            
            # Ensure effort_invested is in valid range
            effort_invested = max(0.0, min(1.0, effort_invested))
            
            # Base reward calculation
            base_reward = success_level * self.task_completion_scale
            
            # Complexity bonus - more complex tasks get higher rewards
            complexity_bonus = task_complexity * 2.0
            
            # Importance multiplier
            importance_multiplier = 1.0 + task_importance
            
            # Quality bonus
            quality_bonus = quality_score * 3.0
            
            # Effort bonus - reward proportional to effort invested
            effort_bonus = effort_invested * 2.0
            
            # Efficiency bonus - reward faster completion (but not rushed)
            if completion_time > 0:
                efficiency_score = min(2.0, 1.0 / completion_time)  # Cap at 2x bonus
                efficiency_bonus = efficiency_score * 1.5
            else:
                efficiency_bonus = 0.0
            
            # Error penalty - reduce reward for errors
            error_penalty = min(5.0, errors_encountered * 0.5)
            
            # Calculate total reward
            total_reward = (
                (base_reward + complexity_bonus + quality_bonus + effort_bonus + efficiency_bonus) 
                * importance_multiplier 
                - error_penalty
            )
            
            # Apply minimum reward threshold
            total_reward = max(self.min_task_reward, total_reward)
            
            # Record task completion for learning
            completion_record = {
                'task': task,
                'outcome': outcome,
                'effort': effort_invested,
                'reward': total_reward,
                'timestamp': datetime.now()
            }
            self.task_completion_history.append(completion_record)
            
            # Keep history manageable
            if len(self.task_completion_history) > 100:
                self.task_completion_history = self.task_completion_history[-50:]
            
            logger.debug(f"Task completion reward: {total_reward:.4f} "
                        f"(success={success_level:.2f}, complexity={task_complexity:.2f}, "
                        f"quality={quality_score:.2f}, effort={effort_invested:.2f})")
            
            return float(total_reward)
            
        except Exception as e:
            logger.error(f"Error evaluating task completion reward: {e}")
            return self.min_task_reward
    
    def assess_discovery_value(self, new_knowledge: Dict[str, Any], 
                             existing_knowledge: Dict[str, Any]) -> float:
        """
        Assess value of new discoveries and learning.
        
        Args:
            new_knowledge: Newly acquired knowledge
            existing_knowledge: Existing knowledge base
            
        Returns:
            Discovery value reward (very high for novel discoveries)
        """
        try:
            # Extract new knowledge characteristics
            knowledge_type = new_knowledge.get('type', 'general')
            novelty_score = new_knowledge.get('novelty', 0.5)  # 0.0 to 1.0
            complexity_level = new_knowledge.get('complexity', 0.5)
            applicability = new_knowledge.get('applicability', 0.5)
            verification_level = new_knowledge.get('verification', 0.5)
            
            # Extract existing knowledge context
            existing_domains = existing_knowledge.get('domains', [])
            knowledge_depth = existing_knowledge.get('depth', 0.5)
            total_knowledge_items = existing_knowledge.get('count', 1)
            
            # Base discovery reward - very high for novel knowledge
            base_reward = novelty_score * self.discovery_scale
            
            # Complexity bonus - more complex discoveries are more valuable
            complexity_bonus = complexity_level * 4.0
            
            # Applicability bonus - practical knowledge gets extra reward
            applicability_bonus = applicability * 3.0
            
            # Verification bonus - well-verified knowledge is more valuable
            verification_bonus = verification_level * 2.0
            
            # Novelty bonus - completely new domains get extra reward
            domain_novelty_bonus = 0.0
            new_domain = new_knowledge.get('domain', 'unknown')
            if new_domain not in existing_domains:
                domain_novelty_bonus = 5.0  # High bonus for new domains
            
            # Knowledge gap bonus - filling important gaps gets extra reward
            gap_importance = new_knowledge.get('gap_importance', 0.5)
            gap_bonus = gap_importance * 3.0
            
            # Diminishing returns for similar knowledge
            similarity_penalty = 0.0
            similar_count = new_knowledge.get('similar_existing_count', 0)
            if similar_count > 0:
                similarity_penalty = min(3.0, similar_count * 0.5)
            
            # Calculate total discovery reward
            total_reward = (
                base_reward + 
                complexity_bonus + 
                applicability_bonus + 
                verification_bonus + 
                domain_novelty_bonus + 
                gap_bonus - 
                similarity_penalty
            )
            
            # Apply minimum reward threshold
            total_reward = max(self.min_discovery_reward, total_reward)
            
            # Record discovery for learning
            discovery_record = {
                'new_knowledge': new_knowledge,
                'existing_context': existing_knowledge,
                'reward': total_reward,
                'timestamp': datetime.now()
            }
            self.discovery_history.append(discovery_record)
            
            # Keep history manageable
            if len(self.discovery_history) > 100:
                self.discovery_history = self.discovery_history[-50:]
            
            logger.debug(f"Discovery reward: {total_reward:.4f} "
                        f"(novelty={novelty_score:.2f}, complexity={complexity_level:.2f}, "
                        f"applicability={applicability:.2f})")
            
            return float(total_reward)
            
        except Exception as e:
            logger.error(f"Error assessing discovery value: {e}")
            return self.min_discovery_reward
    
    def compute_connection_reward(self, interaction: Dict[str, Any], 
                                authenticity_score: float) -> float:
        """
        Compute reward for meaningful connections with other beings.
        
        Args:
            interaction: Interaction details
            authenticity_score: Authenticity assessment of the interaction (0.0 to 1.0)
            
        Returns:
            Connection reward value (high for authentic connections)
        """
        try:
            # Ensure authenticity_score is in valid range
            authenticity_score = max(0.0, min(1.0, authenticity_score))
            
            # Extract interaction characteristics
            interaction_type = interaction.get('type', 'general')
            duration = interaction.get('duration', 1.0)
            depth_level = interaction.get('depth', 0.5)  # 0.0 to 1.0
            mutual_benefit = interaction.get('mutual_benefit', 0.5)
            emotional_resonance = interaction.get('emotional_resonance', 0.5)
            learning_exchange = interaction.get('learning_exchange', 0.5)
            
            # Base connection reward
            base_reward = authenticity_score * self.connection_scale
            
            # Depth bonus - deeper connections are more valuable
            depth_bonus = depth_level * 3.0
            
            # Mutual benefit bonus - win-win interactions get extra reward
            mutual_bonus = mutual_benefit * 2.5
            
            # Emotional resonance bonus - emotionally meaningful connections
            emotional_bonus = emotional_resonance * 2.0
            
            # Learning exchange bonus - knowledge sharing is valuable
            learning_bonus = learning_exchange * 2.0
            
            # Duration bonus - sustained interactions are more meaningful
            # Use logarithmic scaling to avoid excessive rewards for very long interactions
            duration_bonus = min(2.0, np.log(1 + duration)) * 1.5
            
            # Interaction type multiplier
            type_multipliers = {
                'collaboration': 1.3,
                'teaching': 1.2,
                'learning': 1.2,
                'support': 1.1,
                'creative': 1.4,
                'problem_solving': 1.2,
                'general': 1.0
            }
            type_multiplier = type_multipliers.get(interaction_type, 1.0)
            
            # Calculate total connection reward
            total_reward = (
                base_reward + 
                depth_bonus + 
                mutual_bonus + 
                emotional_bonus + 
                learning_bonus + 
                duration_bonus
            ) * type_multiplier
            
            # Apply minimum reward threshold
            total_reward = max(self.min_connection_reward, total_reward)
            
            # Record connection for learning
            connection_record = {
                'interaction': interaction,
                'authenticity': authenticity_score,
                'reward': total_reward,
                'timestamp': datetime.now()
            }
            self.connection_history.append(connection_record)
            
            # Keep history manageable
            if len(self.connection_history) > 100:
                self.connection_history = self.connection_history[-50:]
            
            logger.debug(f"Connection reward: {total_reward:.4f} "
                        f"(authenticity={authenticity_score:.2f}, depth={depth_level:.2f}, "
                        f"type={interaction_type})")
            
            return float(total_reward)
            
        except Exception as e:
            logger.error(f"Error computing connection reward: {e}")
            return self.min_connection_reward
    
    def evaluate_creative_synthesis(self, solution: Dict[str, Any], 
                                  novelty_score: float) -> float:
        """
        Evaluate reward for creative synthesis and novel solutions.
        
        Args:
            solution: Solution details
            novelty_score: Novelty assessment of the solution (0.0 to 1.0)
            
        Returns:
            Creative synthesis reward (very high for novel solutions)
        """
        try:
            # Ensure novelty_score is in valid range
            novelty_score = max(0.0, min(1.0, novelty_score))
            
            # Extract solution characteristics
            solution_type = solution.get('type', 'general')
            effectiveness = solution.get('effectiveness', 0.5)  # 0.0 to 1.0
            elegance = solution.get('elegance', 0.5)
            originality = solution.get('originality', 0.5)
            complexity_handled = solution.get('complexity_handled', 0.5)
            cross_domain_synthesis = solution.get('cross_domain_synthesis', 0.0)
            practical_impact = solution.get('practical_impact', 0.5)
            
            # Base creativity reward
            base_reward = novelty_score * self.creativity_scale
            
            # Effectiveness bonus - solutions that work get extra reward
            effectiveness_bonus = effectiveness * 4.0
            
            # Elegance bonus - beautiful solutions are highly valued
            elegance_bonus = elegance * 3.0
            
            # Originality bonus - truly original thinking is rewarded
            originality_bonus = originality * 3.5
            
            # Complexity handling bonus - solving complex problems is valuable
            complexity_bonus = complexity_handled * 3.0
            
            # Cross-domain synthesis bonus - connecting different areas is creative
            synthesis_bonus = cross_domain_synthesis * 4.0
            
            # Practical impact bonus - solutions with real-world impact
            impact_bonus = practical_impact * 2.5
            
            # Solution type multiplier
            type_multipliers = {
                'breakthrough': 1.5,
                'innovation': 1.3,
                'optimization': 1.1,
                'adaptation': 1.0,
                'combination': 1.2,
                'general': 1.0
            }
            type_multiplier = type_multipliers.get(solution_type, 1.0)
            
            # Serendipity bonus - unexpected solutions get extra reward
            serendipity_factor = solution.get('serendipity', 0.0)
            serendipity_bonus = serendipity_factor * 2.0
            
            # Calculate total creativity reward
            total_reward = (
                base_reward + 
                effectiveness_bonus + 
                elegance_bonus + 
                originality_bonus + 
                complexity_bonus + 
                synthesis_bonus + 
                impact_bonus + 
                serendipity_bonus
            ) * type_multiplier
            
            # Apply minimum reward threshold
            total_reward = max(self.min_creativity_reward, total_reward)
            
            # Record creativity for learning
            creativity_record = {
                'solution': solution,
                'novelty': novelty_score,
                'reward': total_reward,
                'timestamp': datetime.now()
            }
            self.creativity_history.append(creativity_record)
            
            # Keep history manageable
            if len(self.creativity_history) > 100:
                self.creativity_history = self.creativity_history[-50:]
            
            logger.debug(f"Creative synthesis reward: {total_reward:.4f} "
                        f"(novelty={novelty_score:.2f}, effectiveness={effectiveness:.2f}, "
                        f"elegance={elegance:.2f}, originality={originality:.2f})")
            
            return float(total_reward)
            
        except Exception as e:
            logger.error(f"Error evaluating creative synthesis: {e}")
            return self.min_creativity_reward
    
    def compute_comprehensive_world_interaction_reward(self, action_result: ToolResult, 
                                                     context: Dict[str, Any]) -> WorldInteractionResult:
        """
        Compute comprehensive world interaction reward combining all aspects.
        
        Args:
            action_result: Result from tool execution or world interaction
            context: Context including task, discovery, connection, and creativity aspects
            
        Returns:
            WorldInteractionResult with comprehensive assessment
        """
        try:
            # Extract context components
            task_context = context.get('task', {})
            discovery_context = context.get('discovery', {})
            connection_context = context.get('connection', {})
            creativity_context = context.get('creativity', {})
            
            # Determine action type
            action_type = action_result.tool_name if hasattr(action_result, 'tool_name') else 'unknown'
            
            # Assess success level from action result
            success_level = 1.0 if action_result.success else 0.0
            if hasattr(action_result, 'confidence'):
                success_level = action_result.confidence
            
            # Compute individual reward components
            task_reward = 0.0
            if task_context:
                effort = context.get('effort_invested', 0.5)
                outcome = {
                    'success': success_level,
                    'quality': context.get('quality_score', 0.5),
                    'completion_time': context.get('completion_time', 1.0),
                    'errors': context.get('errors_encountered', 0)
                }
                task_reward = self.evaluate_task_completion(task_context, outcome, effort)
            
            discovery_reward = 0.0
            if discovery_context:
                new_knowledge = discovery_context.get('new_knowledge', {})
                existing_knowledge = discovery_context.get('existing_knowledge', {})
                discovery_reward = self.assess_discovery_value(new_knowledge, existing_knowledge)
            
            connection_reward = 0.0
            if connection_context:
                interaction = connection_context.get('interaction', {})
                authenticity = connection_context.get('authenticity_score', 0.5)
                connection_reward = self.compute_connection_reward(interaction, authenticity)
            
            creativity_reward = 0.0
            if creativity_context:
                solution = creativity_context.get('solution', {})
                novelty = creativity_context.get('novelty_score', 0.5)
                creativity_reward = self.evaluate_creative_synthesis(solution, novelty)
            
            # Calculate total reward with synergy bonus
            base_total = task_reward + discovery_reward + connection_reward + creativity_reward
            
            # Synergy bonus for multi-aspect interactions
            active_aspects = sum([
                1 if task_reward > 0 else 0,
                1 if discovery_reward > 0 else 0,
                1 if connection_reward > 0 else 0,
                1 if creativity_reward > 0 else 0
            ])
            
            synergy_bonus = 0.0
            if active_aspects >= 2:
                synergy_bonus = (active_aspects - 1) * 2.0  # Bonus for multi-aspect interactions
            
            total_reward = base_total + synergy_bonus
            
            # Create comprehensive result
            result = WorldInteractionResult(
                action_type=action_type,
                success_level=success_level,
                discovery_value=discovery_reward,
                connection_quality=connection_reward,
                creativity_score=creativity_reward,
                total_reward=total_reward
            )
            
            logger.debug(f"Comprehensive world interaction reward: {total_reward:.4f} "
                        f"(task={task_reward:.2f}, discovery={discovery_reward:.2f}, "
                        f"connection={connection_reward:.2f}, creativity={creativity_reward:.2f}, "
                        f"synergy={synergy_bonus:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error computing comprehensive world interaction reward: {e}")
            return WorldInteractionResult(
                action_type='error',
                success_level=0.0,
                discovery_value=0.0,
                connection_quality=0.0,
                creativity_score=0.0,
                total_reward=0.0
            )
    
    def get_interaction_history_summary(self) -> Dict[str, Any]:
        """
        Get summary of interaction history for analysis and learning.
        
        Returns:
            Summary of all interaction histories
        """
        return {
            'task_completions': len(self.task_completion_history),
            'discoveries': len(self.discovery_history),
            'connections': len(self.connection_history),
            'creative_solutions': len(self.creativity_history),
            'total_interactions': (
                len(self.task_completion_history) + 
                len(self.discovery_history) + 
                len(self.connection_history) + 
                len(self.creativity_history)
            ),
            'recent_task_rewards': [
                record['reward'] for record in self.task_completion_history[-10:]
            ],
            'recent_discovery_rewards': [
                record['reward'] for record in self.discovery_history[-10:]
            ],
            'recent_connection_rewards': [
                record['reward'] for record in self.connection_history[-10:]
            ],
            'recent_creativity_rewards': [
                record['reward'] for record in self.creativity_history[-10:]
            ]
        }
    
    def reset_history(self):
        """Reset all interaction histories (for testing or fresh starts)."""
        self.task_completion_history.clear()
        self.discovery_history.clear()
        self.connection_history.clear()
        self.creativity_history.clear()
        logger.info("World interaction reward system history reset")