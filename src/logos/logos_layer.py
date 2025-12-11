"""
Concrete implementation of the Logos Layer reasoning engine.
"""

import time
import logging
from typing import List, Optional, Dict, Any
import numpy as np
from datetime import datetime

from .interfaces import LogosLayer as LogosLayerInterface
from ..core.models import SemanticVector, Intention, MemoryTrace, ToolCall
from ..core.config import LogosConfig
from ..core.math_utils import cosine_similarity, normalize_vector
from ..core.logging_config import instrumentation, logger


class LogosLayer(LogosLayerInterface):
    """
    Concrete implementation of the Logos cognitive reasoning engine.
    
    Handles intention generation, semantic interpretation, and preference learning
    based on current affective state and memory patterns.
    """
    
    def __init__(self, config: LogosConfig):
        super().__init__(config)
        self.preference_weights: Dict[str, float] = {}
        self.historical_patterns: Dict[str, List[float]] = {}
        self.tool_effectiveness: Dict[str, List[float]] = {}
        self.semantic_categories = [
            'exploration', 'analysis', 'communication', 'learning', 
            'planning', 'reflection', 'creativity', 'problem_solving'
        ]
        
        # Initialize preference weights for semantic categories
        for category in self.semantic_categories:
            self.preference_weights[category] = 0.5  # Neutral starting point
            self.historical_patterns[category] = []
    
    def interpret_state(self, pathos_state: np.ndarray, memories: List[MemoryTrace]) -> SemanticVector:
        """
        Interpret current pathos state and memories to create semantic representation.
        
        Creates a semantic vector by analyzing the affective state patterns and
        incorporating insights from relevant memories.
        """
        logger.debug(f"Interpreting state with pathos dimension: {len(pathos_state)}, memories: {len(memories)}")
        
        # Analyze affective state to determine dominant patterns
        state_magnitude = np.linalg.norm(pathos_state)
        state_direction = normalize_vector(pathos_state, zero_vector_strategy='return_zero')
        
        # Extract semantic insights from memories
        memory_themes = self._extract_memory_themes(memories)
        
        # Determine semantic category based on state and memory patterns
        semantic_category = self._determine_semantic_category(pathos_state, memory_themes)
        
        # Generate semantic embedding
        embedding = self._create_semantic_embedding(pathos_state, memory_themes, semantic_category)
        
        # Create intention text based on interpretation
        intention_text = self._generate_intention_text(semantic_category, state_magnitude, memory_themes)
        
        # Log the reasoning process
        logger.debug("Logos - Semantic interpretation", 
                    semantic_category=semantic_category,
                    state_magnitude=f"{state_magnitude:.3f}",
                    memory_themes=str(list(memory_themes.keys())[:3]),
                    intention_preview=intention_text[:60])
        
        return SemanticVector(
            embedding=embedding,
            intention_text=intention_text,
            semantic_category=semantic_category
        )
    
    def generate_intention(self, semantic_vector: SemanticVector, pathos_state: np.ndarray) -> Intention:
        """
        Generate intention based on semantic interpretation and current state.
        
        Creates a structured intention with priority and tool candidates based on
        current preferences and historical success patterns.
        """
        logger.debug("Logos - Generating intention", 
                    semantic_category=semantic_vector.semantic_category,
                    base_intention=semantic_vector.intention_text[:50])
        
        # Compute priority based on preference weights and state energy
        priority = self._compute_intention_priority(semantic_vector, pathos_state)
        
        # Select tool candidates based on semantic category and historical effectiveness
        tool_candidates = self._select_tool_candidates(semantic_vector.semantic_category)
        
        # Enhance description with self-reflective elements
        enhanced_description = self._enhance_intention_description(
            semantic_vector.intention_text, pathos_state, priority
        )
        
        logger.debug("Logos - Intention details", 
                    priority=f"{priority:.3f}",
                    tool_candidates=str(tool_candidates[:3]),
                    enhanced_description=enhanced_description[:80])
        
        # Create base intention
        base_intention = Intention(
            description=enhanced_description,
            semantic_vector=semantic_vector,
            priority=priority,
            tool_candidates=tool_candidates
        )
        
        # Integrate historical patterns to bias toward successful approaches
        final_intention = self.integrate_historical_context(base_intention)
        
        return final_intention
    
    def evaluate_tool_usage(self, intention: Intention, available_tools: List[str]) -> Optional[ToolCall]:
        """
        Evaluate whether tool usage is needed for the current intention.
        
        Determines if any available tools would be beneficial for achieving
        the current intention based on historical effectiveness.
        """
        logger.debug(f"Evaluating tool usage for intention: {intention.description[:50]}...")
        
        # Filter available tools by candidates
        viable_tools = [tool for tool in available_tools if tool in intention.tool_candidates]
        
        if not viable_tools:
            return None
        
        # Select best tool based on effectiveness history
        best_tool = self._select_best_tool(viable_tools, intention.semantic_vector.semantic_category)
        
        if best_tool:
            # Generate appropriate arguments for the tool
            arguments = self._generate_tool_arguments(best_tool, intention)
            
            return ToolCall(
                tool_name=best_tool,
                arguments=arguments,
                timestamp=int(datetime.now().timestamp())
            )
        
        return None
    
    def compute_interest_signal(self, semantic_input: SemanticVector) -> float:
        """
        Compute interest signal for Pathos integration.
        
        Interest is based on novelty, preference alignment, and potential for learning.
        """
        category = semantic_input.semantic_category
        
        # Base interest from preference weights
        preference_interest = self.preference_weights.get(category, 0.5)
        
        # Novelty interest (higher for less explored categories)
        novelty_interest = 1.0 - min(len(self.historical_patterns.get(category, [])) / 100.0, 1.0)
        
        # Semantic complexity interest
        complexity_interest = min(np.linalg.norm(semantic_input.embedding) / 10.0, 1.0)
        
        # Combine interest signals
        total_interest = (
            0.4 * preference_interest +
            0.3 * novelty_interest +
            0.3 * complexity_interest
        )
        
        return np.clip(total_interest, 0.0, 1.0)
    
    def update_preferences(self, reward: float, semantic_vector: SemanticVector) -> None:
        """
        Update preference weights based on reward feedback.
        
        Implements preference learning by adjusting weights for semantic categories
        based on experienced rewards.
        """
        category = semantic_vector.semantic_category
        
        # Update preference weight with learning rate
        learning_rate = 0.1
        current_weight = self.preference_weights.get(category, 0.5)
        
        # Reward-based update (positive rewards increase preference)
        reward_signal = np.tanh(reward)  # Normalize reward
        new_weight = current_weight + learning_rate * reward_signal
        
        # Keep weights in reasonable bounds
        new_weight_clipped = np.clip(new_weight, 0.0, 1.0)
        self.preference_weights[category] = new_weight_clipped
        
        # Record preference change for monitoring
        instrumentation.record_preference_change(category, current_weight, new_weight_clipped)
        
        # Update historical patterns
        if category not in self.historical_patterns:
            self.historical_patterns[category] = []
        
        self.historical_patterns[category].append(reward)
        
        # Keep only recent history (last 100 experiences)
        if len(self.historical_patterns[category]) > 100:
            self.historical_patterns[category] = self.historical_patterns[category][-100:]
        
        logger.debug(f"Updated preference for {category}: {self.preference_weights[category]:.3f}")
    
    def update_tool_effectiveness(self, tool_name: str, reward: float, category: str) -> None:
        """
        Update tool effectiveness tracking based on outcomes.
        
        Args:
            tool_name: Name of the tool that was used
            reward: Reward received from using the tool
            category: Semantic category context
        """
        key = f"{tool_name}_{category}"
        
        if key not in self.tool_effectiveness:
            self.tool_effectiveness[key] = []
        
        self.tool_effectiveness[key].append(reward)
        
        # Keep only recent history
        if len(self.tool_effectiveness[key]) > 50:
            self.tool_effectiveness[key] = self.tool_effectiveness[key][-50:]
    
    def analyze_historical_patterns(self) -> Dict[str, Any]:
        """
        Analyze historical reward patterns to identify successful themes and methods.
        
        Returns:
            Dictionary containing pattern analysis results
        """
        analysis = {
            'successful_categories': {},
            'effective_tools': {},
            'preference_trends': {},
            'learning_progress': {}
        }
        
        # Analyze successful semantic categories
        for category, rewards in self.historical_patterns.items():
            if rewards:
                avg_reward = np.mean(rewards)
                recent_avg = np.mean(rewards[-10:]) if len(rewards) >= 10 else avg_reward
                trend = recent_avg - avg_reward if len(rewards) > 1 else 0.0
                
                analysis['successful_categories'][category] = {
                    'average_reward': avg_reward,
                    'recent_average': recent_avg,
                    'trend': trend,
                    'experience_count': len(rewards),
                    'success_rate': len([r for r in rewards if r > 0]) / len(rewards)
                }
        
        # Analyze effective tools
        for tool_key, rewards in self.tool_effectiveness.items():
            if rewards:
                tool_name, category = tool_key.rsplit('_', 1)
                avg_effectiveness = np.mean(rewards)
                
                if tool_name not in analysis['effective_tools']:
                    analysis['effective_tools'][tool_name] = {}
                
                analysis['effective_tools'][tool_name][category] = {
                    'average_effectiveness': avg_effectiveness,
                    'usage_count': len(rewards),
                    'success_rate': len([r for r in rewards if r > 0]) / len(rewards)
                }
        
        # Analyze preference trends
        for category, weight in self.preference_weights.items():
            if category in self.historical_patterns:
                rewards = self.historical_patterns[category]
                if len(rewards) >= 5:
                    early_avg = np.mean(rewards[:5])
                    recent_avg = np.mean(rewards[-5:])
                    learning_rate = (recent_avg - early_avg) / len(rewards)
                    
                    analysis['preference_trends'][category] = {
                        'current_weight': weight,
                        'early_performance': early_avg,
                        'recent_performance': recent_avg,
                        'learning_rate': learning_rate
                    }
        
        # Overall learning progress
        all_rewards = []
        for rewards in self.historical_patterns.values():
            all_rewards.extend(rewards)
        
        if all_rewards:
            analysis['learning_progress'] = {
                'total_experiences': len(all_rewards),
                'overall_average': np.mean(all_rewards),
                'recent_performance': np.mean(all_rewards[-20:]) if len(all_rewards) >= 20 else np.mean(all_rewards),
                'improvement_trend': self._calculate_improvement_trend(all_rewards)
            }
        
        return analysis
    
    def get_bias_toward_successful_patterns(self, semantic_category: str) -> float:
        """
        Calculate bias factor toward successful themes, tools, and methods.
        
        Args:
            semantic_category: The semantic category to evaluate
            
        Returns:
            Bias factor (0.0 to 2.0, where 1.0 is neutral)
        """
        if semantic_category not in self.historical_patterns:
            return 1.0  # Neutral bias for unexplored categories
        
        rewards = self.historical_patterns[semantic_category]
        if not rewards:
            return 1.0
        
        # Calculate success metrics
        avg_reward = np.mean(rewards)
        success_rate = len([r for r in rewards if r > 0]) / len(rewards)
        recent_performance = np.mean(rewards[-5:]) if len(rewards) >= 5 else avg_reward
        
        # Combine metrics to create bias factor
        # Higher average reward and success rate increase bias
        reward_bias = np.clip(avg_reward + 1.0, 0.5, 1.5)  # Normalize around 1.0
        success_bias = 0.5 + success_rate  # 0.5 to 1.5 range
        recency_bias = np.clip(recent_performance + 1.0, 0.5, 1.5)
        
        # Weight the factors
        total_bias = (
            0.4 * reward_bias +
            0.3 * success_bias +
            0.3 * recency_bias
        )
        
        return np.clip(total_bias, 0.5, 2.0)
    
    def integrate_historical_context(self, current_intention: Intention) -> Intention:
        """
        Integrate historical patterns into current intention to bias toward successful approaches.
        
        Args:
            current_intention: The base intention to enhance
            
        Returns:
            Enhanced intention with historical bias applied
        """
        category = current_intention.semantic_vector.semantic_category
        
        # Get bias factor for this category
        bias_factor = self.get_bias_toward_successful_patterns(category)
        
        # Adjust priority based on historical success
        enhanced_priority = current_intention.priority * bias_factor
        enhanced_priority = np.clip(enhanced_priority, 0.0, 1.0)
        
        # Enhance tool candidates with historically effective tools
        enhanced_candidates = current_intention.tool_candidates.copy()
        
        # Add historically effective tools for this category
        for tool_key, rewards in self.tool_effectiveness.items():
            if tool_key.endswith(f"_{category}") and rewards:
                tool_name = tool_key.split('_')[0]
                avg_effectiveness = np.mean(rewards)
                
                if avg_effectiveness > 0.5 and tool_name not in enhanced_candidates:
                    enhanced_candidates.append(tool_name)
        
        # Sort tool candidates by effectiveness
        def tool_effectiveness_score(tool_name):
            key = f"{tool_name}_{category}"
            if key in self.tool_effectiveness and self.tool_effectiveness[key]:
                return np.mean(self.tool_effectiveness[key])
            return 0.5  # Default score for unknown tools
        
        enhanced_candidates.sort(key=tool_effectiveness_score, reverse=True)
        
        # Enhance description with historical context
        enhanced_description = current_intention.description
        
        if bias_factor > 1.2:
            enhanced_description += f" (Building on previous success in {category})"
        elif bias_factor < 0.8:
            enhanced_description += f" (Exploring new approaches in {category})"
        
        return Intention(
            description=enhanced_description,
            semantic_vector=current_intention.semantic_vector,
            priority=enhanced_priority,
            tool_candidates=enhanced_candidates[:5]  # Limit to top 5
        )
    
    def _calculate_improvement_trend(self, rewards: List[float]) -> float:
        """
        Calculate improvement trend from a series of rewards.
        
        Args:
            rewards: List of reward values over time
            
        Returns:
            Trend value (positive = improving, negative = declining)
        """
        if len(rewards) < 3:
            return 0.0
        
        # Use linear regression to find trend
        x = np.arange(len(rewards))
        y = np.array(rewards)
        
        # Simple linear regression
        n = len(rewards)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        
        # Calculate slope (trend)
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        return slope
    
    def evaluate_internal_state(self, pathos_state: np.ndarray, recent_memories: List[MemoryTrace]) -> Dict[str, Any]:
        """
        Evaluate current internal state for self-reflective planning.
        
        Args:
            pathos_state: Current affective state vector
            recent_memories: Recent memory traces for context
            
        Returns:
            Dictionary containing internal state evaluation
        """
        evaluation = {
            'affective_analysis': {},
            'memory_patterns': {},
            'preference_status': {},
            'learning_assessment': {},
            'emergent_drives': []
        }
        
        # Analyze affective state
        state_magnitude = np.linalg.norm(pathos_state)
        state_stability = self._assess_state_stability(pathos_state)
        dominant_dimensions = self._identify_dominant_dimensions(pathos_state)
        
        evaluation['affective_analysis'] = {
            'energy_level': min(state_magnitude / 10.0, 1.0),  # Normalize to 0-1
            'stability': state_stability,
            'dominant_dimensions': dominant_dimensions,
            'complexity': self._compute_state_complexity(pathos_state)
        }
        
        # Analyze memory patterns
        if recent_memories:
            memory_themes = self._extract_memory_themes(recent_memories)
            memory_coherence = self._assess_memory_coherence(recent_memories)
            
            evaluation['memory_patterns'] = {
                'recent_themes': memory_themes,
                'coherence': memory_coherence,
                'reward_trend': self._analyze_recent_reward_trend(recent_memories),
                'diversity': self._assess_memory_diversity(recent_memories)
            }
        
        # Analyze preference status
        preference_analysis = self.analyze_historical_patterns()
        evaluation['preference_status'] = {
            'strongest_preferences': self._identify_strongest_preferences(),
            'emerging_interests': self._identify_emerging_interests(),
            'preference_stability': self._assess_preference_stability(),
            'learning_progress': preference_analysis.get('learning_progress', {})
        }
        
        # Assess learning and adaptation
        evaluation['learning_assessment'] = {
            'exploration_vs_exploitation': self._assess_exploration_exploitation_balance(),
            'adaptation_rate': self._compute_adaptation_rate(),
            'knowledge_gaps': self._identify_knowledge_gaps(),
            'skill_development': self._assess_skill_development()
        }
        
        # Detect emergent drives from layer interactions
        evaluation['emergent_drives'] = self._detect_emergent_drives(
            pathos_state, recent_memories, evaluation
        )
        
        return evaluation
    
    def create_self_reflective_plan(self, internal_evaluation: Dict[str, Any], 
                                  current_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a planning strategy that incorporates affect state and memory history.
        
        Args:
            internal_evaluation: Results from evaluate_internal_state
            current_context: Current situational context
            
        Returns:
            Self-reflective planning strategy
        """
        plan = {
            'strategic_focus': {},
            'tactical_adjustments': {},
            'learning_objectives': [],
            'risk_assessments': {},
            'adaptation_strategies': []
        }
        
        # Determine strategic focus based on internal state
        affective_analysis = internal_evaluation.get('affective_analysis', {})
        energy_level = affective_analysis.get('energy_level', 0.5)
        stability = affective_analysis.get('stability', 0.5)
        
        if energy_level > 0.7 and stability > 0.6:
            strategic_focus = 'exploration_and_growth'
        elif energy_level < 0.3 or stability < 0.4:
            strategic_focus = 'stabilization_and_reflection'
        else:
            strategic_focus = 'balanced_development'
        
        plan['strategic_focus'] = {
            'primary_mode': strategic_focus,
            'rationale': self._explain_strategic_choice(energy_level, stability),
            'duration_estimate': self._estimate_focus_duration(internal_evaluation),
            'success_metrics': self._define_success_metrics(strategic_focus)
        }
        
        # Tactical adjustments based on preferences and memory patterns
        preference_status = internal_evaluation.get('preference_status', {})
        memory_patterns = internal_evaluation.get('memory_patterns', {})
        
        plan['tactical_adjustments'] = {
            'preference_refinement': self._plan_preference_refinement(preference_status),
            'memory_integration': self._plan_memory_integration(memory_patterns),
            'tool_usage_optimization': self._plan_tool_optimization(),
            'attention_allocation': self._plan_attention_allocation(internal_evaluation)
        }
        
        # Learning objectives based on gaps and opportunities
        learning_assessment = internal_evaluation.get('learning_assessment', {})
        plan['learning_objectives'] = self._formulate_learning_objectives(
            learning_assessment, internal_evaluation
        )
        
        # Risk assessments for potential issues
        plan['risk_assessments'] = {
            'preference_drift': self._assess_preference_drift_risk(preference_status),
            'exploration_stagnation': self._assess_exploration_risk(learning_assessment),
            'memory_overload': self._assess_memory_overload_risk(memory_patterns),
            'adaptation_failure': self._assess_adaptation_risk(internal_evaluation)
        }
        
        # Adaptation strategies for different scenarios
        plan['adaptation_strategies'] = self._develop_adaptation_strategies(
            internal_evaluation, plan['risk_assessments']
        )
        
        return plan
    
    def detect_emergent_drives(self, pathos_state: np.ndarray, memories: List[MemoryTrace], 
                             preferences: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Detect emergent drives from the interaction of all layers.
        
        Args:
            pathos_state: Current affective state
            memories: Recent memory traces
            preferences: Current preference weights
            
        Returns:
            List of detected emergent drives
        """
        drives = []
        
        # Analyze patterns in the interaction between affect, memory, and preferences
        
        # Drive 1: Curiosity drive (high energy + low familiarity)
        energy = min(np.linalg.norm(pathos_state) / 10.0, 1.0)  # Normalize energy to 0-1
        if energy > 0.6:
            unfamiliar_categories = [cat for cat, weight in preferences.items() if weight < 0.4]
            if unfamiliar_categories:
                strength = energy * (1.0 - np.mean([preferences[cat] for cat in unfamiliar_categories]))
                drives.append({
                    'type': 'curiosity',
                    'strength': np.clip(strength, 0.0, 2.0),
                    'target_categories': unfamiliar_categories,
                    'description': f"Strong drive to explore {', '.join(unfamiliar_categories[:2])}"
                })
        
        # Drive 2: Mastery drive (high preference + recent success)
        for category, weight in preferences.items():
            if weight > 0.7 and category in self.historical_patterns:
                recent_rewards = self.historical_patterns[category][-5:]
                if recent_rewards and np.mean(recent_rewards) > 0.5:
                    strength = weight * np.mean(recent_rewards)
                    drives.append({
                        'type': 'mastery',
                        'strength': np.clip(strength, 0.0, 2.0),
                        'target_categories': [category],
                        'description': f"Drive to deepen expertise in {category}"
                    })
        
        # Drive 3: Balance drive (detect imbalances in experience)
        category_experience = {cat: len(patterns) for cat, patterns in self.historical_patterns.items()}
        if category_experience:
            max_exp = max(category_experience.values())
            min_exp = min(category_experience.values())
            if max_exp > min_exp * 3:  # Significant imbalance
                underexplored = [cat for cat, exp in category_experience.items() if exp == min_exp]
                strength = (max_exp - min_exp) / max_exp
                drives.append({
                    'type': 'balance',
                    'strength': np.clip(strength, 0.0, 2.0),
                    'target_categories': underexplored,
                    'description': f"Drive to balance experience across categories"
                })
        
        # Drive 4: Integration drive (connect disparate experiences)
        if len(memories) > 3:
            memory_categories = [m.semantic_vector.semantic_category for m in memories]
            unique_categories = set(memory_categories)
            if len(unique_categories) > 2:
                strength = len(unique_categories) / len(self.semantic_categories)
                drives.append({
                    'type': 'integration',
                    'strength': np.clip(strength, 0.0, 2.0),
                    'target_categories': list(unique_categories),
                    'description': f"Drive to integrate insights across {len(unique_categories)} domains"
                })
        
        # Sort drives by strength
        drives.sort(key=lambda d: d['strength'], reverse=True)
        
        return drives[:3]  # Return top 3 drives
    
    def _assess_state_stability(self, pathos_state: np.ndarray) -> float:
        """Assess the stability of the affective state."""
        # Use variance as a proxy for stability (lower variance = more stable)
        variance = np.var(pathos_state)
        # Normalize to 0-1 range (higher = more stable)
        stability = 1.0 / (1.0 + variance)
        return np.clip(stability, 0.0, 1.0)
    
    def _identify_dominant_dimensions(self, pathos_state: np.ndarray, top_k: int = 5) -> List[int]:
        """Identify the most active dimensions in the affective state."""
        abs_values = np.abs(pathos_state)
        top_indices = np.argsort(abs_values)[-top_k:]
        return top_indices.tolist()
    
    def _compute_state_complexity(self, pathos_state: np.ndarray) -> float:
        """Compute complexity measure of the affective state."""
        # Use entropy-like measure
        abs_values = np.abs(pathos_state)
        if np.sum(abs_values) == 0:
            return 0.0
        
        normalized = abs_values / np.sum(abs_values)
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        entropy = -np.sum(normalized * np.log(normalized + epsilon))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(len(pathos_state))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _assess_memory_coherence(self, memories: List[MemoryTrace]) -> float:
        """Assess coherence of memory patterns."""
        if len(memories) < 2:
            return 1.0
        
        # Measure similarity between consecutive memories
        similarities = []
        for i in range(len(memories) - 1):
            sim = cosine_similarity(
                memories[i].semantic_vector.embedding,
                memories[i + 1].semantic_vector.embedding
            )
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _analyze_recent_reward_trend(self, memories: List[MemoryTrace]) -> float:
        """Analyze trend in recent rewards."""
        if len(memories) < 2:
            return 0.0
        
        rewards = [m.reward for m in memories]
        return self._calculate_improvement_trend(rewards)
    
    def _assess_memory_diversity(self, memories: List[MemoryTrace]) -> float:
        """Assess diversity of memory categories."""
        if not memories:
            return 0.0
        
        categories = [m.semantic_vector.semantic_category for m in memories]
        unique_categories = set(categories)
        
        return len(unique_categories) / len(self.semantic_categories)
    
    def _identify_strongest_preferences(self, top_k: int = 3) -> List[Dict[str, Any]]:
        """Identify strongest current preferences."""
        sorted_prefs = sorted(
            self.preference_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {'category': cat, 'weight': weight, 'rank': i + 1}
            for i, (cat, weight) in enumerate(sorted_prefs[:top_k])
        ]
    
    def _identify_emerging_interests(self) -> List[Dict[str, Any]]:
        """Identify categories with growing interest."""
        emerging = []
        
        for category, patterns in self.historical_patterns.items():
            if len(patterns) >= 5:
                recent_avg = np.mean(patterns[-3:])
                earlier_avg = np.mean(patterns[:3])
                
                if recent_avg > earlier_avg + 0.2:  # Significant improvement
                    emerging.append({
                        'category': category,
                        'growth_rate': recent_avg - earlier_avg,
                        'current_performance': recent_avg
                    })
        
        return sorted(emerging, key=lambda x: x['growth_rate'], reverse=True)
    
    def _assess_preference_stability(self) -> float:
        """Assess how stable preferences are over time."""
        # This is a simplified measure - in a full implementation,
        # we'd track preference changes over time
        if not self.preference_weights:
            return 0.5
        
        # Use variance of preference weights as stability measure
        weights = list(self.preference_weights.values())
        variance = np.var(weights)
        
        # Lower variance = more stable
        stability = 1.0 / (1.0 + variance * 10)
        return np.clip(stability, 0.0, 1.0)
    
    def _assess_exploration_exploitation_balance(self) -> Dict[str, float]:
        """Assess balance between exploration and exploitation."""
        total_experiences = sum(len(patterns) for patterns in self.historical_patterns.values())
        
        if total_experiences == 0:
            return {'exploration': 1.0, 'exploitation': 0.0, 'balance_score': 0.0}
        
        # Categories with few experiences = exploration
        # Categories with many experiences = exploitation
        exploration_count = sum(1 for patterns in self.historical_patterns.values() if len(patterns) < 5)
        exploitation_count = len(self.historical_patterns) - exploration_count
        
        total_categories = len(self.historical_patterns)
        exploration_ratio = exploration_count / total_categories if total_categories > 0 else 0.0
        exploitation_ratio = exploitation_count / total_categories if total_categories > 0 else 0.0
        
        # Ideal balance is around 0.3 exploration, 0.7 exploitation
        ideal_exploration = 0.3
        balance_score = 1.0 - abs(exploration_ratio - ideal_exploration)
        
        return {
            'exploration': exploration_ratio,
            'exploitation': exploitation_ratio,
            'balance_score': balance_score
        }
    
    def _compute_adaptation_rate(self) -> float:
        """Compute how quickly the system adapts to new information."""
        # Measure how quickly preferences change in response to rewards
        if not self.historical_patterns:
            return 0.5
        
        adaptation_scores = []
        for category, patterns in self.historical_patterns.items():
            if len(patterns) >= 3:
                # Look at how preference changed relative to reward changes
                current_pref = self.preference_weights.get(category, 0.5)
                # This is simplified - in practice we'd track preference changes over time
                adaptation_scores.append(min(abs(current_pref - 0.5) * 2, 1.0))
        
        return np.mean(adaptation_scores) if adaptation_scores else 0.5
    
    def _identify_knowledge_gaps(self) -> List[str]:
        """Identify areas with limited experience."""
        gaps = []
        
        for category in self.semantic_categories:
            experience_count = len(self.historical_patterns.get(category, []))
            if experience_count < 3:
                gaps.append(category)
        
        return gaps
    
    def _assess_skill_development(self) -> Dict[str, float]:
        """Assess skill development in different categories."""
        skills = {}
        
        for category, patterns in self.historical_patterns.items():
            if len(patterns) >= 3:
                # Measure improvement over time
                early_performance = np.mean(patterns[:len(patterns)//2])
                recent_performance = np.mean(patterns[len(patterns)//2:])
                
                improvement = recent_performance - early_performance
                skills[category] = np.clip(improvement + 0.5, 0.0, 1.0)  # Normalize to 0-1
        
        return skills
    
    def _detect_emergent_drives(self, pathos_state: np.ndarray, memories: List[MemoryTrace], 
                              evaluation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect emergent drives from layer interactions."""
        return self.detect_emergent_drives(pathos_state, memories, self.preference_weights)
    
    # Additional helper methods for planning
    def _explain_strategic_choice(self, energy_level: float, stability: float) -> str:
        """Explain the rationale for strategic focus choice."""
        if energy_level > 0.7 and stability > 0.6:
            return "High energy and stability enable ambitious exploration and growth"
        elif energy_level < 0.3 or stability < 0.4:
            return "Low energy or instability requires focus on stabilization and reflection"
        else:
            return "Moderate state allows for balanced development across multiple areas"
    
    def _estimate_focus_duration(self, evaluation: Dict[str, Any]) -> str:
        """Estimate how long to maintain current strategic focus."""
        stability = evaluation.get('affective_analysis', {}).get('stability', 0.5)
        
        if stability > 0.8:
            return "extended_period"  # Can maintain focus longer
        elif stability > 0.5:
            return "moderate_period"
        else:
            return "short_period"  # Need frequent reassessment
    
    def _define_success_metrics(self, strategic_focus: str) -> List[str]:
        """Define success metrics for the strategic focus."""
        metrics = {
            'exploration_and_growth': [
                'New categories explored',
                'Diversity of experiences',
                'Learning rate improvement'
            ],
            'stabilization_and_reflection': [
                'Affective state stability',
                'Memory coherence improvement',
                'Preference clarity'
            ],
            'balanced_development': [
                'Balanced category experience',
                'Consistent performance',
                'Adaptive flexibility'
            ]
        }
        
        return metrics.get(strategic_focus, ['General progress'])
    
    def _plan_preference_refinement(self, preference_status: Dict[str, Any]) -> Dict[str, Any]:
        """Plan how to refine preferences based on current status."""
        return {
            'strengthen_successful': preference_status.get('strongest_preferences', [])[:2],
            'explore_emerging': preference_status.get('emerging_interests', [])[:2],
            'rebalance_if_needed': len(preference_status.get('strongest_preferences', [])) > 5
        }
    
    def _plan_memory_integration(self, memory_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Plan how to better integrate memory patterns."""
        coherence = memory_patterns.get('coherence', 0.5)
        
        return {
            'improve_coherence': coherence < 0.6,
            'diversify_themes': memory_patterns.get('diversity', 0) < 0.4,
            'strengthen_connections': True  # Always beneficial
        }
    
    def _plan_tool_optimization(self) -> Dict[str, Any]:
        """Plan tool usage optimization."""
        return {
            'evaluate_effectiveness': True,
            'explore_new_tools': len(self.tool_effectiveness) < 10,
            'optimize_selection': True
        }
    
    def _plan_attention_allocation(self, evaluation: Dict[str, Any]) -> Dict[str, str]:
        """Plan how to allocate attention across different areas."""
        drives = evaluation.get('emergent_drives', [])
        
        if drives:
            primary_drive = drives[0]
            return {
                'primary_focus': primary_drive.get('type', 'exploration'),
                'secondary_focus': drives[1].get('type', 'learning') if len(drives) > 1 else 'reflection',
                'attention_split': '70/30'  # Primary/Secondary
            }
        
        return {
            'primary_focus': 'balanced_exploration',
            'secondary_focus': 'skill_development',
            'attention_split': '60/40'
        }
    
    def _formulate_learning_objectives(self, learning_assessment: Dict[str, Any], 
                                     evaluation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Formulate specific learning objectives."""
        objectives = []
        
        # Address knowledge gaps
        gaps = learning_assessment.get('knowledge_gaps', [])
        for gap in gaps[:2]:  # Top 2 gaps
            objectives.append({
                'type': 'knowledge_gap',
                'target': gap,
                'priority': 'high',
                'description': f"Gain experience in {gap}"
            })
        
        # Improve weak skills
        skills = learning_assessment.get('skill_development', {})
        weak_skills = [cat for cat, score in skills.items() if score < 0.4]
        for skill in weak_skills[:2]:
            objectives.append({
                'type': 'skill_improvement',
                'target': skill,
                'priority': 'medium',
                'description': f"Improve performance in {skill}"
            })
        
        # Leverage emergent drives
        drives = evaluation.get('emergent_drives', [])
        if drives:
            primary_drive = drives[0]
            objectives.append({
                'type': 'drive_fulfillment',
                'target': primary_drive.get('type'),
                'priority': 'high',
                'description': primary_drive.get('description', 'Fulfill emergent drive')
            })
        
        return objectives
    
    def _assess_preference_drift_risk(self, preference_status: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk of unhealthy preference drift."""
        stability = preference_status.get('preference_stability', 0.5)
        
        return {
            'risk_level': 'low' if stability > 0.7 else 'medium' if stability > 0.4 else 'high',
            'indicators': ['rapid_preference_changes'] if stability < 0.4 else [],
            'mitigation': 'monitor_stability' if stability < 0.6 else 'none_needed'
        }
    
    def _assess_exploration_risk(self, learning_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk of exploration stagnation."""
        balance = learning_assessment.get('exploration_vs_exploitation', {})
        exploration_ratio = balance.get('exploration', 0.5)
        
        return {
            'risk_level': 'high' if exploration_ratio < 0.1 else 'medium' if exploration_ratio < 0.2 else 'low',
            'indicators': ['low_exploration'] if exploration_ratio < 0.2 else [],
            'mitigation': 'force_exploration' if exploration_ratio < 0.1 else 'encourage_exploration'
        }
    
    def _assess_memory_overload_risk(self, memory_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk of memory system overload."""
        # This is simplified - in practice we'd check actual memory usage
        return {
            'risk_level': 'low',  # Placeholder
            'indicators': [],
            'mitigation': 'none_needed'
        }
    
    def _assess_adaptation_risk(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk of adaptation failure."""
        learning_assessment = evaluation.get('learning_assessment', {})
        adaptation_rate = learning_assessment.get('adaptation_rate', 0.5)
        
        return {
            'risk_level': 'high' if adaptation_rate < 0.2 else 'medium' if adaptation_rate < 0.4 else 'low',
            'indicators': ['slow_adaptation'] if adaptation_rate < 0.4 else [],
            'mitigation': 'increase_learning_rate' if adaptation_rate < 0.3 else 'monitor'
        }
    
    def _develop_adaptation_strategies(self, evaluation: Dict[str, Any], 
                                    risks: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Develop strategies for different adaptation scenarios."""
        strategies = []
        
        # Strategy for high energy periods
        energy = evaluation.get('affective_analysis', {}).get('energy_level', 0.5)
        if energy > 0.7:
            strategies.append({
                'scenario': 'high_energy',
                'strategy': 'aggressive_exploration',
                'actions': ['explore_new_categories', 'try_novel_approaches', 'take_calculated_risks']
            })
        
        # Strategy for low energy periods
        if energy < 0.3:
            strategies.append({
                'scenario': 'low_energy',
                'strategy': 'consolidation_and_reflection',
                'actions': ['review_recent_experiences', 'strengthen_existing_skills', 'rest_and_recover']
            })
        
        # Strategy for high-risk situations
        high_risk_areas = [area for area, risk in risks.items() if risk.get('risk_level') == 'high']
        if high_risk_areas:
            strategies.append({
                'scenario': 'high_risk',
                'strategy': 'risk_mitigation',
                'actions': [f"address_{area}_risk" for area in high_risk_areas]
            })
        
        return strategies
    
    def get_preference_summary(self) -> Dict[str, float]:
        """Get current preference weights for monitoring."""
        return self.preference_weights.copy()
    
    def _extract_memory_themes(self, memories: List[MemoryTrace]) -> Dict[str, float]:
        """Extract thematic patterns from memory traces."""
        themes = {}
        
        for memory in memories:
            category = memory.semantic_vector.semantic_category
            reward = memory.reward
            
            if category not in themes:
                themes[category] = []
            themes[category].append(reward)
        
        # Compute average reward per theme
        theme_scores = {}
        for category, rewards in themes.items():
            theme_scores[category] = np.mean(rewards) if rewards else 0.0
        
        return theme_scores
    
    def _determine_semantic_category(self, pathos_state: np.ndarray, memory_themes: Dict[str, float]) -> str:
        """Determine the most appropriate semantic category for current state."""
        # Use state energy and memory themes to select category
        state_energy = np.linalg.norm(pathos_state)
        
        # High energy states favor exploration and creativity
        if state_energy > 0.8:
            candidates = ['exploration', 'creativity', 'problem_solving']
        # Medium energy favors analysis and learning
        elif state_energy > 0.4:
            candidates = ['analysis', 'learning', 'planning']
        # Low energy favors reflection and communication
        else:
            candidates = ['reflection', 'communication']
        
        # Select based on preferences and recent memory themes
        best_category = candidates[0]
        best_score = -float('inf')
        
        for category in candidates:
            # Combine preference weight with recent memory success
            preference_score = self.preference_weights.get(category, 0.5)
            memory_score = memory_themes.get(category, 0.0)
            
            total_score = 0.7 * preference_score + 0.3 * memory_score
            
            if total_score > best_score:
                best_score = total_score
                best_category = category
        
        return best_category
    
    def _create_semantic_embedding(self, pathos_state: np.ndarray, memory_themes: Dict[str, float], 
                                 category: str) -> np.ndarray:
        """Create semantic embedding vector from state and themes."""
        # Start with normalized pathos state as base
        base_embedding = normalize_vector(pathos_state[:self.config.semantic_dimension], 
                                        zero_vector_strategy='return_zero')
        
        # If pathos state is smaller than semantic dimension, pad with zeros
        if len(base_embedding) < self.config.semantic_dimension:
            padding = np.zeros(self.config.semantic_dimension - len(base_embedding))
            base_embedding = np.concatenate([base_embedding, padding])
        
        # Add category-specific bias
        category_index = self.semantic_categories.index(category) if category in self.semantic_categories else 0
        category_bias = np.zeros(self.config.semantic_dimension)
        
        # Create category signature in embedding
        if category_index < self.config.semantic_dimension:
            category_bias[category_index] = 0.5
        
        # Combine base embedding with category bias
        semantic_embedding = 0.8 * base_embedding + 0.2 * category_bias
        
        return normalize_vector(semantic_embedding, zero_vector_strategy='return_zero')
    
    def _generate_intention_text(self, category: str, state_magnitude: float, 
                               memory_themes: Dict[str, float]) -> str:
        """Generate human-readable intention text."""
        # Base intentions by category
        base_intentions = {
            'exploration': "Explore new possibilities and gather information",
            'analysis': "Analyze current situation and identify patterns",
            'communication': "Communicate insights and share understanding",
            'learning': "Learn from experience and update knowledge",
            'planning': "Plan future actions and set goals",
            'reflection': "Reflect on past experiences and internal state",
            'creativity': "Generate creative solutions and novel ideas",
            'problem_solving': "Solve current challenges and overcome obstacles"
        }
        
        base_text = base_intentions.get(category, "Engage in meaningful activity")
        
        # Add state-dependent modifiers
        if state_magnitude > 0.8:
            modifier = "with high energy and focus"
        elif state_magnitude > 0.4:
            modifier = "with balanced attention"
        else:
            modifier = "with calm consideration"
        
        return f"{base_text} {modifier}"
    
    def _compute_intention_priority(self, semantic_vector: SemanticVector, pathos_state: np.ndarray) -> float:
        """Compute priority score for the intention."""
        category = semantic_vector.semantic_category
        
        # Base priority from preference
        preference_priority = self.preference_weights.get(category, 0.5)
        
        # State energy contribution
        state_energy = np.linalg.norm(pathos_state)
        energy_priority = min(state_energy, 1.0)
        
        # Historical success contribution
        if category in self.historical_patterns and self.historical_patterns[category]:
            recent_rewards = self.historical_patterns[category][-10:]  # Last 10 experiences
            success_priority = np.clip(np.mean(recent_rewards) + 0.5, 0.0, 1.0)
        else:
            success_priority = 0.5  # Neutral for unexplored categories
        
        # Combine priorities
        total_priority = (
            0.4 * preference_priority +
            0.3 * energy_priority +
            0.3 * success_priority
        )
        
        return np.clip(total_priority, 0.0, 1.0)
    
    def _select_tool_candidates(self, category: str) -> List[str]:
        """Select appropriate tool candidates for the semantic category."""
        # Tool mappings by category - using actually registered tools
        category_tools = {
            'exploration': ['communication_search', 'analysis_data_processor', 'communication_echo'],
            'analysis': ['analysis_data_processor', 'analysis_advanced_analyzer', 'system_optimizer'],
            'communication': ['communication_echo', 'communication_search', 'creative_problem_solver'],
            'learning': ['learning_synthesizer', 'analysis_data_processor', 'communication_search'],
            'planning': ['system_optimizer', 'creative_problem_solver', 'analysis_data_processor'],
            'reflection': ['analysis_data_processor', 'learning_synthesizer', 'communication_echo'],
            'creativity': ['creative_problem_solver', 'learning_synthesizer', 'development_tool_creator'],
            'problem_solving': ['creative_problem_solver', 'system_optimizer', 'analysis_data_processor']
        }
        
        candidates = category_tools.get(category, ['general_tool'])
        
        # Add tools with high effectiveness for this category
        effective_tools = []
        for tool_key, rewards in self.tool_effectiveness.items():
            if tool_key.endswith(f"_{category}") and rewards:
                avg_reward = np.mean(rewards)
                if avg_reward > 0.5:  # Threshold for effectiveness
                    tool_name = tool_key.split('_')[0]
                    effective_tools.append(tool_name)
        
        # Combine and deduplicate
        all_candidates = list(set(candidates + effective_tools))
        
        return all_candidates[:5]  # Limit to top 5 candidates
    
    def _select_best_tool(self, viable_tools: List[str], category: str) -> Optional[str]:
        """Select the best tool from viable options based on effectiveness."""
        if not viable_tools:
            return None
        
        best_tool = None
        best_score = -float('inf')
        
        for tool in viable_tools:
            key = f"{tool}_{category}"
            
            if key in self.tool_effectiveness and self.tool_effectiveness[key]:
                # Use average effectiveness
                avg_effectiveness = np.mean(self.tool_effectiveness[key])
            else:
                # Default score for untested tools
                avg_effectiveness = 0.5
            
            if avg_effectiveness > best_score:
                best_score = avg_effectiveness
                best_tool = tool
        
        # Only return tool if it meets minimum effectiveness threshold
        return best_tool if best_score > 0.3 else None
    
    def _generate_tool_arguments(self, tool_name: str, intention: Intention) -> Dict[str, Any]:
        """Generate appropriate arguments for tool execution."""
        # Basic argument templates by tool type
        base_args = {
            'query': intention.semantic_vector.intention_text,
            'category': intention.semantic_vector.semantic_category,
            'priority': intention.priority
        }
        
        # Tool-specific argument generation
        if 'search' in tool_name.lower():
            base_args['search_terms'] = self._extract_search_terms(intention.description)
        elif 'analyzer' in tool_name.lower():
            base_args['analysis_type'] = intention.semantic_vector.semantic_category
        elif 'generator' in tool_name.lower():
            base_args['generation_prompt'] = intention.description
        
        return base_args
    
    def _extract_search_terms(self, description: str) -> List[str]:
        """Extract search terms from intention description."""
        # Simple keyword extraction (could be enhanced with NLP)
        words = description.lower().split()
        
        # Filter out common words and keep meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:5]  # Limit to 5 keywords
    
    def _enhance_intention_description(self, base_description: str, pathos_state: np.ndarray, 
                                     priority: float) -> str:
        """Enhance intention description with self-reflective elements."""
        # Add state awareness
        state_energy = np.linalg.norm(pathos_state)
        
        if state_energy > 0.8:
            energy_desc = "I feel energized and ready for challenging tasks."
        elif state_energy > 0.4:
            energy_desc = "I'm in a balanced state, suitable for focused work."
        else:
            energy_desc = "I'm in a calm state, good for reflection and careful analysis."
        
        # Add priority awareness
        if priority > 0.8:
            priority_desc = "This feels very important to me right now."
        elif priority > 0.5:
            priority_desc = "This seems like a worthwhile pursuit."
        else:
            priority_desc = "This might be worth exploring when I have time."
        
        return f"{base_description}. {energy_desc} {priority_desc}"