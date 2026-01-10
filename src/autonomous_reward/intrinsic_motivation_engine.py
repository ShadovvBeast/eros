"""
Intrinsic Motivation Engine for the Autonomous Reward System.

This module implements the IntrinsicMotivationEngine class that generates self-directed
motivation for curiosity, mastery, autonomy, and growth, enabling the agent to develop
authentic intrinsic drives without external constraints.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Set
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import math

from .interfaces import IntrinsicMotivationEngineInterface
from .models import Pattern

logger = logging.getLogger(__name__)


class IntrinsicMotivationEngine(IntrinsicMotivationEngineInterface):
    """
    Generates self-directed motivation for curiosity, mastery, autonomy, and growth.
    
    This class implements the core intrinsic motivation system that drives the agent
    to explore unknown domains, develop mastery, maintain autonomy, and pursue growth
    based purely on internal drives and state dynamics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the intrinsic motivation engine.
        
        Args:
            config: Optional configuration parameters for motivation computation
        """
        self.config = config or {}
        
        # Motivation module weights
        self.curiosity_weight = self.config.get('curiosity_weight', 1.0)
        self.mastery_weight = self.config.get('mastery_weight', 1.0)
        self.autonomy_weight = self.config.get('autonomy_weight', 1.2)
        self.growth_weight = self.config.get('growth_weight', 1.1)
        
        # Curiosity module parameters
        self.curiosity_threshold = self.config.get('curiosity_threshold', 0.3)
        self.novelty_decay_rate = self.config.get('novelty_decay_rate', 0.1)
        self.exploration_bonus = self.config.get('exploration_bonus', 0.2)
        
        # Mastery module parameters
        self.mastery_threshold = self.config.get('mastery_threshold', 0.8)
        self.skill_decay_rate = self.config.get('skill_decay_rate', 0.05)
        self.mastery_plateau_detection = self.config.get('mastery_plateau_detection', 5)
        
        # Autonomy module parameters
        self.autonomy_threshold = self.config.get('autonomy_threshold', 0.6)
        self.dependency_penalty = self.config.get('dependency_penalty', 0.3)
        self.self_direction_bonus = self.config.get('self_direction_bonus', 0.4)
        
        # Growth module parameters
        self.growth_threshold = self.config.get('growth_threshold', 0.5)
        self.capability_expansion_bonus = self.config.get('capability_expansion_bonus', 0.3)
        self.learning_acceleration_factor = self.config.get('learning_acceleration_factor', 1.2)
        
        # Internal state tracking
        self.knowledge_domain_familiarity: Dict[str, float] = defaultdict(float)
        self.skill_performance_history: Dict[str, List[float]] = defaultdict(list)
        self.autonomy_action_history: List[Dict[str, Any]] = []
        self.capability_evolution_history: List[Set[str]] = []
        self.motivation_history: List[Dict[str, float]] = []
        
        # Temporal tracking
        self.last_curiosity_update = datetime.now()
        self.last_mastery_assessment = datetime.now()
        self.last_autonomy_evaluation = datetime.now()
        self.last_growth_evaluation = datetime.now()
        
        # History management
        self.max_history_size = self.config.get('max_history_size', 100)
        
        logger.info("Initialized IntrinsicMotivationEngine")
    
    def generate_curiosity_drive(self, knowledge_gaps: List[str], 
                               state_energy: float) -> float:
        """
        Generate curiosity drive for exploring unknown domains.
        
        Curiosity drive increases when the agent encounters unknown domains or
        identifies knowledge gaps. The drive is modulated by current state energy
        and the novelty of the unknown domains.
        
        Args:
            knowledge_gaps: Identified areas of unknown knowledge
            state_energy: Current energy level from pathos state
            
        Returns:
            Curiosity drive strength (0.0 to 2.0)
        """
        if not knowledge_gaps:
            return 0.0
        
        # Normalize state energy to [0, 1] range
        normalized_energy = min(state_energy / 10.0, 1.0)  # Assume max energy ~10
        
        # Base curiosity from knowledge gaps
        gap_novelty_scores = []
        
        for gap in knowledge_gaps:
            # Check familiarity with this domain
            familiarity = self.knowledge_domain_familiarity.get(gap, 0.0)
            
            # Novelty is inverse of familiarity
            novelty = 1.0 - familiarity
            
            # Apply novelty decay over time
            time_since_update = (datetime.now() - self.last_curiosity_update).total_seconds() / 3600.0  # hours
            novelty_with_decay = novelty * math.exp(-self.novelty_decay_rate * time_since_update)
            
            gap_novelty_scores.append(novelty_with_decay)
            
            # Update familiarity (slight increase from exposure)
            self.knowledge_domain_familiarity[gap] = min(familiarity + 0.01, 1.0)
        
        # Compute average novelty across gaps
        avg_novelty = np.mean(gap_novelty_scores) if gap_novelty_scores else 0.0
        
        # Energy modulation: higher energy enables more curiosity
        energy_modulation = 0.5 + 0.5 * normalized_energy  # Range [0.5, 1.0]
        
        # Exploration bonus for having multiple diverse gaps
        diversity_bonus = 0.0
        if len(knowledge_gaps) > 1:
            # Bonus increases with number of gaps (up to a limit)
            diversity_bonus = min(len(knowledge_gaps) * 0.1, self.exploration_bonus)
        
        # Combine curiosity components
        base_curiosity = avg_novelty * energy_modulation
        curiosity_drive = (base_curiosity + diversity_bonus) * self.curiosity_weight
        
        # Apply curiosity threshold
        if curiosity_drive < self.curiosity_threshold:
            curiosity_drive *= 0.5  # Reduce below threshold
        
        # Cap at maximum value
        curiosity_drive = min(curiosity_drive, 2.0)
        
        # Update temporal tracking
        self.last_curiosity_update = datetime.now()
        
        logger.debug(f"Generated curiosity drive: {curiosity_drive:.4f} "
                    f"(novelty={avg_novelty:.3f}, energy={normalized_energy:.3f}, "
                    f"diversity_bonus={diversity_bonus:.3f})")
        
        return curiosity_drive
    
    def assess_mastery_progress(self, skill_domain: str, 
                              performance_history: List[float]) -> float:
        """
        Assess progress toward mastery in a skill domain.
        
        Mastery progress is measured by improvement trends, consistency,
        and approach toward mastery thresholds. Rewards are higher for
        steady improvement and breakthrough moments.
        
        Args:
            skill_domain: Domain of skill being developed
            performance_history: Historical performance data (0.0 to 1.0)
            
        Returns:
            Mastery progress reward (0.0 to 2.0)
        """
        if not performance_history:
            return 0.0
        
        # Update internal performance history
        self.skill_performance_history[skill_domain].extend(performance_history)
        if len(self.skill_performance_history[skill_domain]) > self.max_history_size:
            self.skill_performance_history[skill_domain] = \
                self.skill_performance_history[skill_domain][-self.max_history_size:]
        
        full_history = self.skill_performance_history[skill_domain]
        
        if len(full_history) < 2:
            return 0.0
        
        # Measure 1: Performance improvement trend
        recent_performance = full_history[-min(10, len(full_history)):]  # Last 10 or all
        
        if len(recent_performance) >= 2:
            # Compute linear trend
            x = np.arange(len(recent_performance))
            trend_slope = np.polyfit(x, recent_performance, 1)[0]
            
            # Positive trend indicates improvement, but weight it based on absolute performance level
            avg_performance = np.mean(recent_performance)
            
            # For high performance levels, small negative trends shouldn't be heavily penalized
            if avg_performance > 0.8 and trend_slope > -0.1:
                # High performance with small decline - still good
                improvement_reward = max(0.3, trend_slope * 3.0 + 0.5)
            elif trend_slope > 0:
                # Positive trend - reward improvement
                improvement_reward = min(trend_slope * 5.0, 1.0)
            else:
                # Negative trend - penalize but not too heavily if performance is still good
                if avg_performance > 0.6:
                    improvement_reward = max(0.1, trend_slope * 2.0 + 0.3)
                else:
                    improvement_reward = max(0.0, trend_slope * 3.0 + 0.2)
            
            improvement_reward = max(0.0, min(improvement_reward, 1.0))  # Clamp to [0, 1]
        else:
            improvement_reward = 0.0
        
        # Measure 2: Consistency in performance
        if len(recent_performance) >= 3:
            performance_std = np.std(recent_performance)
            # Adjust consistency reward to be more forgiving and consider performance level
            base_consistency = 1.0 / (1.0 + performance_std * 1.0)  # Further reduce penalty for variation
            
            # Bonus for consistent high performance (plateaus at high levels are good)
            avg_performance = np.mean(recent_performance)
            if avg_performance > 0.8:  # High performance level
                consistency_reward = base_consistency * (1.0 + avg_performance * 0.5)  # Increase bonus
            elif avg_performance > 0.6:  # Good performance level
                consistency_reward = base_consistency * (1.0 + avg_performance * 0.3)
            else:
                consistency_reward = base_consistency
        else:
            consistency_reward = 0.0
        
        # Measure 3: Mastery level achievement and absolute performance
        current_performance = recent_performance[-1]
        avg_performance = np.mean(recent_performance)
        
        # Base reward for absolute performance level
        absolute_performance_reward = 0.0
        if avg_performance > 0.9:
            absolute_performance_reward = 0.8  # Excellent performance
        elif avg_performance > 0.8:
            absolute_performance_reward = 0.6  # Very good performance
        elif avg_performance > 0.7:
            absolute_performance_reward = 0.4  # Good performance
        elif avg_performance > 0.5:
            absolute_performance_reward = 0.2  # Moderate performance
        
        # Additional mastery threshold bonus
        mastery_threshold_bonus = 0.0
        if current_performance >= self.mastery_threshold:
            # Bonus for achieving mastery level
            mastery_threshold_bonus = 0.3
            
            # Additional bonus for exceeding mastery
            if current_performance > 0.9:
                mastery_threshold_bonus += 0.2
        
        mastery_level_reward = absolute_performance_reward + mastery_threshold_bonus
        
        # Measure 4: Breakthrough detection (significant jumps in performance)
        breakthrough_reward = 0.0
        if len(full_history) >= 5:
            # Look for significant performance jumps
            recent_window = full_history[-5:]
            performance_jumps = np.diff(recent_window)
            
            # Find significant positive jumps
            mean_jump = np.mean(performance_jumps)
            std_jump = np.std(performance_jumps)
            
            for jump in performance_jumps:
                if jump > mean_jump + 2 * std_jump and jump > 0.1:  # Significant positive jump
                    breakthrough_reward += 0.2
            
            breakthrough_reward = min(breakthrough_reward, 0.4)  # Cap breakthrough bonus
        
        # Measure 5: Plateau detection and motivation adjustment
        plateau_penalty = 0.0
        if len(recent_performance) >= self.mastery_plateau_detection:
            # Check if performance has plateaued
            recent_variance = np.var(recent_performance)
            recent_trend = abs(trend_slope) if 'trend_slope' in locals() else 0.0
            avg_performance = np.mean(recent_performance)
            
            # Only penalize plateaus at low performance levels
            # High performance plateaus (mastery maintenance) should not be penalized
            if recent_variance < 0.005 and recent_trend < 0.005:
                if avg_performance < 0.5:  # Low performance plateau
                    plateau_penalty = 0.1
                elif avg_performance < 0.7:  # Medium performance plateau
                    plateau_penalty = 0.05
                # No penalty for high performance plateaus (>= 0.7)
        
        # Apply skill decay over time (reduce decay effect)
        time_since_assessment = (datetime.now() - self.last_mastery_assessment).total_seconds() / 3600.0  # hours
        decay_factor = math.exp(-self.skill_decay_rate * time_since_assessment * 0.5)  # Reduce decay impact
        
        # Combine mastery components (adjust weights to properly balance improvement vs absolute performance)
        raw_mastery = (improvement_reward * 0.3 + consistency_reward * 0.2 + 
                      mastery_level_reward * 0.4 + breakthrough_reward * 0.1 - plateau_penalty)
        
        mastery_progress = raw_mastery * decay_factor * self.mastery_weight
        mastery_progress = max(0.0, min(mastery_progress, 2.0))  # Clamp to [0, 2]
        
        # Update temporal tracking
        self.last_mastery_assessment = datetime.now()
        
        logger.debug(f"Assessed mastery progress for {skill_domain}: {mastery_progress:.4f} "
                    f"(improvement={improvement_reward:.3f}, consistency={consistency_reward:.3f}, "
                    f"mastery_level={mastery_level_reward:.3f}, breakthrough={breakthrough_reward:.3f})")
        
        return mastery_progress
    
    def compute_autonomy_reward(self, self_directed_actions: int, 
                              external_dependencies: int) -> float:
        """
        Compute reward for self-direction and independence.
        
        Autonomy reward increases when the agent makes self-directed decisions
        and decreases when it relies heavily on external guidance or constraints.
        
        Args:
            self_directed_actions: Number of autonomous actions taken
            external_dependencies: Number of external dependencies
            
        Returns:
            Autonomy reward value (0.0 to 2.0)
        """
        total_actions = self_directed_actions + external_dependencies
        
        if total_actions == 0:
            return 0.0
        
        # Record autonomy action for history
        autonomy_record = {
            'timestamp': datetime.now(),
            'self_directed': self_directed_actions,
            'external_dependencies': external_dependencies,
            'autonomy_ratio': self_directed_actions / total_actions
        }
        
        self.autonomy_action_history.append(autonomy_record)
        if len(self.autonomy_action_history) > self.max_history_size:
            self.autonomy_action_history.pop(0)
        
        # Measure 1: Current autonomy ratio
        current_autonomy_ratio = self_directed_actions / total_actions
        
        # Base autonomy reward from ratio
        base_autonomy = current_autonomy_ratio
        
        # Measure 2: Autonomy trend over time
        autonomy_trend_reward = 0.0
        if len(self.autonomy_action_history) >= 3:
            recent_ratios = [record['autonomy_ratio'] for record in self.autonomy_action_history[-5:]]
            
            if len(recent_ratios) >= 2:
                # Compute trend in autonomy
                x = np.arange(len(recent_ratios))
                autonomy_trend = np.polyfit(x, recent_ratios, 1)[0]
                
                # Positive trend (increasing autonomy) is rewarded
                autonomy_trend_reward = max(0.0, autonomy_trend * 2.0)
                autonomy_trend_reward = min(autonomy_trend_reward, 0.5)
        
        # Measure 3: Self-direction bonus
        self_direction_bonus = 0.0
        if current_autonomy_ratio >= self.autonomy_threshold:
            # Bonus for achieving autonomy threshold
            self_direction_bonus = self.self_direction_bonus
            
            # Additional bonus for high autonomy
            if current_autonomy_ratio >= 0.8:
                self_direction_bonus += 0.2
        
        # Measure 4: Dependency penalty
        dependency_penalty = 0.0
        if external_dependencies > self_directed_actions:
            # Penalty for being more dependent than autonomous
            dependency_ratio = external_dependencies / total_actions
            dependency_penalty = dependency_ratio * self.dependency_penalty
        
        # Measure 5: Consistency in autonomy
        consistency_bonus = 0.0
        if len(self.autonomy_action_history) >= 5:
            recent_ratios = [record['autonomy_ratio'] for record in self.autonomy_action_history[-5:]]
            autonomy_std = np.std(recent_ratios)
            
            # Reward consistent high autonomy
            if np.mean(recent_ratios) >= self.autonomy_threshold and autonomy_std < 0.2:
                consistency_bonus = 0.2
        
        # Apply temporal decay
        time_since_evaluation = (datetime.now() - self.last_autonomy_evaluation).total_seconds() / 3600.0  # hours
        decay_factor = math.exp(-0.02 * time_since_evaluation)  # Slow decay for autonomy
        
        # Combine autonomy components
        raw_autonomy = (base_autonomy + autonomy_trend_reward + self_direction_bonus + 
                       consistency_bonus - dependency_penalty)
        
        autonomy_reward = raw_autonomy * decay_factor * self.autonomy_weight
        autonomy_reward = max(0.0, min(autonomy_reward, 2.0))  # Clamp to [0, 2]
        
        # Update temporal tracking
        self.last_autonomy_evaluation = datetime.now()
        
        logger.debug(f"Computed autonomy reward: {autonomy_reward:.4f} "
                    f"(base={base_autonomy:.3f}, trend={autonomy_trend_reward:.3f}, "
                    f"self_direction={self_direction_bonus:.3f}, consistency={consistency_bonus:.3f}, "
                    f"dependency_penalty={dependency_penalty:.3f})")
        
        return autonomy_reward
    
    def evaluate_growth_potential(self, current_capabilities: Set[str], 
                                learning_opportunities: List[str]) -> float:
        """
        Evaluate potential for capability expansion and learning.
        
        Growth potential is assessed based on available learning opportunities,
        capability expansion trends, and the agent's readiness for new challenges.
        
        Args:
            current_capabilities: Set of current capabilities
            learning_opportunities: Available learning opportunities
            
        Returns:
            Growth potential score (0.0 to 2.0)
        """
        # Record capability evolution
        self.capability_evolution_history.append(current_capabilities.copy())
        if len(self.capability_evolution_history) > self.max_history_size:
            self.capability_evolution_history.pop(0)
        
        # Measure 1: Learning opportunity richness
        opportunity_richness = 0.0
        if learning_opportunities:
            # Base score from number of opportunities
            opportunity_count_score = min(len(learning_opportunities) / 10.0, 1.0)  # Normalize to [0,1]
            
            # Diversity bonus for different types of opportunities
            unique_opportunities = set(learning_opportunities)
            diversity_score = min(len(unique_opportunities) / 5.0, 1.0)  # Normalize to [0,1]
            
            opportunity_richness = (opportunity_count_score + diversity_score) / 2.0
        
        # Measure 2: Capability expansion trend
        expansion_trend_reward = 0.0
        if len(self.capability_evolution_history) >= 3:
            # Track capability set sizes over time
            capability_sizes = [len(caps) for caps in self.capability_evolution_history[-5:]]
            
            if len(capability_sizes) >= 2:
                # Compute trend in capability growth
                x = np.arange(len(capability_sizes))
                size_trend = np.polyfit(x, capability_sizes, 1)[0]
                
                # Positive trend indicates capability expansion
                expansion_trend_reward = max(0.0, size_trend * 0.2)
                expansion_trend_reward = min(expansion_trend_reward, 0.5)
        
        # Measure 3: Capability-opportunity alignment
        alignment_bonus = 0.0
        if learning_opportunities and current_capabilities:
            # Check how many opportunities build on existing capabilities
            aligned_opportunities = 0
            
            for opportunity in learning_opportunities:
                # Simple heuristic: check if opportunity relates to existing capabilities
                for capability in current_capabilities:
                    if any(word in opportunity.lower() for word in capability.lower().split()):
                        aligned_opportunities += 1
                        break
            
            if learning_opportunities:
                alignment_ratio = aligned_opportunities / len(learning_opportunities)
                # Moderate alignment is optimal (some building on existing, some novel)
                optimal_alignment = 0.4
                alignment_bonus = 1.0 - abs(alignment_ratio - optimal_alignment)
                alignment_bonus = max(0.0, alignment_bonus) * 0.3
        
        # Measure 4: Growth readiness (based on recent capability additions)
        growth_readiness = 0.0
        if len(self.capability_evolution_history) >= 2:
            recent_capabilities = self.capability_evolution_history[-1]
            previous_capabilities = self.capability_evolution_history[-2]
            
            # Check for recent capability additions
            new_capabilities = recent_capabilities - previous_capabilities
            
            if new_capabilities:
                # Recent growth indicates readiness for more
                growth_readiness = min(len(new_capabilities) * 0.2, 0.4)
            else:
                # No recent growth might indicate readiness for new challenges
                if len(self.capability_evolution_history) >= 5:
                    # Check if there's been stagnation
                    recent_sizes = [len(caps) for caps in self.capability_evolution_history[-3:]]
                    if all(size == recent_sizes[0] for size in recent_sizes):
                        growth_readiness = 0.3  # Stagnation indicates need for growth
        
        # Measure 5: Capability expansion bonus
        expansion_bonus = 0.0
        if len(self.capability_evolution_history) >= 2:
            current_size = len(current_capabilities)
            previous_size = len(self.capability_evolution_history[-2])
            
            if current_size > previous_size:
                # Bonus for actual capability expansion
                expansion_bonus = min((current_size - previous_size) * self.capability_expansion_bonus, 0.5)
        
        # Apply temporal factors
        time_since_evaluation = (datetime.now() - self.last_growth_evaluation).total_seconds() / 3600.0  # hours
        
        # Growth motivation increases over time without growth
        time_factor = min(time_since_evaluation * 0.01, 0.2)  # Gradual increase
        
        # Combine growth components
        raw_growth = (opportunity_richness * 0.3 + expansion_trend_reward * 0.2 + 
                     alignment_bonus * 0.2 + growth_readiness * 0.2 + expansion_bonus * 0.1 + time_factor)
        
        growth_potential = raw_growth * self.growth_weight
        
        # Apply growth threshold
        if growth_potential >= self.growth_threshold:
            # Acceleration bonus for high growth potential
            growth_potential *= self.learning_acceleration_factor
        
        growth_potential = max(0.0, min(growth_potential, 2.0))  # Clamp to [0, 2]
        
        # Update temporal tracking
        self.last_growth_evaluation = datetime.now()
        
        logger.debug(f"Evaluated growth potential: {growth_potential:.4f} "
                    f"(opportunity_richness={opportunity_richness:.3f}, "
                    f"expansion_trend={expansion_trend_reward:.3f}, "
                    f"alignment={alignment_bonus:.3f}, readiness={growth_readiness:.3f}, "
                    f"expansion_bonus={expansion_bonus:.3f})")
        
        return growth_potential
    
    def get_motivation_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of current motivation state.
        
        Returns:
            Dictionary containing motivation state information
        """
        return {
            'knowledge_domains': len(self.knowledge_domain_familiarity),
            'tracked_skills': len(self.skill_performance_history),
            'autonomy_history_length': len(self.autonomy_action_history),
            'capability_evolution_length': len(self.capability_evolution_history),
            'last_updates': {
                'curiosity': self.last_curiosity_update.isoformat(),
                'mastery': self.last_mastery_assessment.isoformat(),
                'autonomy': self.last_autonomy_evaluation.isoformat(),
                'growth': self.last_growth_evaluation.isoformat()
            },
            'domain_familiarity': dict(self.knowledge_domain_familiarity),
            'current_capabilities': list(self.capability_evolution_history[-1]) if self.capability_evolution_history else []
        }
    
    def reset_motivation_state(self) -> None:
        """Reset all motivation tracking state for fresh start."""
        self.knowledge_domain_familiarity.clear()
        self.skill_performance_history.clear()
        self.autonomy_action_history.clear()
        self.capability_evolution_history.clear()
        self.motivation_history.clear()
        
        # Reset temporal tracking
        now = datetime.now()
        self.last_curiosity_update = now
        self.last_mastery_assessment = now
        self.last_autonomy_evaluation = now
        self.last_growth_evaluation = now
        
        logger.info("Reset IntrinsicMotivationEngine state")
    
    def update_domain_familiarity(self, domain: str, familiarity_increase: float) -> None:
        """
        Update familiarity with a knowledge domain.
        
        Args:
            domain: Knowledge domain name
            familiarity_increase: Amount to increase familiarity (0.0 to 1.0)
        """
        current_familiarity = self.knowledge_domain_familiarity[domain]
        new_familiarity = min(current_familiarity + familiarity_increase, 1.0)
        self.knowledge_domain_familiarity[domain] = new_familiarity
        
        logger.debug(f"Updated domain familiarity for {domain}: {new_familiarity:.3f}")
    
    def get_skill_performance_trend(self, skill_domain: str) -> Optional[float]:
        """
        Get the performance trend for a specific skill domain.
        
        Args:
            skill_domain: Skill domain to analyze
            
        Returns:
            Performance trend slope or None if insufficient data
        """
        if skill_domain not in self.skill_performance_history:
            return None
        
        history = self.skill_performance_history[skill_domain]
        if len(history) < 2:
            return None
        
        x = np.arange(len(history))
        trend_slope = np.polyfit(x, history, 1)[0]
        
        return trend_slope
    
    def get_autonomy_trend(self) -> Optional[float]:
        """
        Get the autonomy trend over recent history.
        
        Returns:
            Autonomy trend slope or None if insufficient data
        """
        if len(self.autonomy_action_history) < 2:
            return None
        
        ratios = [record['autonomy_ratio'] for record in self.autonomy_action_history[-10:]]
        
        if len(ratios) < 2:
            return None
        
        x = np.arange(len(ratios))
        trend_slope = np.polyfit(x, ratios, 1)[0]
        
        return trend_slope