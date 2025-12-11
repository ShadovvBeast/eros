"""
Ethos Framework implementation for identity and constraint management.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from .interfaces import EthosFramework
from ..models import Intention, ToolCall
from ..config import EthosConfig


class ConcreteEthosFramework(EthosFramework):
    """
    Concrete implementation of the Ethos Framework for identity and constraint management.
    
    This class manages:
    - Identity description and core values storage
    - Safety constraint validation methods
    - Personality trait configuration and enforcement
    """
    
    def __init__(self, config: EthosConfig):
        super().__init__(config)
        self._personality_traits = config.personality_traits.copy()
        self._value_keywords = self._build_value_keywords()
        self._constraint_patterns = self._build_constraint_patterns()
    
    def _build_value_keywords(self) -> Dict[str, List[str]]:
        """Build keyword mappings for core values"""
        value_keywords = {}
        for value in self.config.core_values:
            if value.lower() == 'curiosity':
                value_keywords[value] = ['explore', 'learn', 'discover', 'investigate', 'research', 'question']
            elif value.lower() == 'safety':
                value_keywords[value] = ['safe', 'secure', 'protect', 'careful', 'cautious', 'responsible']
            elif value.lower() == 'growth':
                value_keywords[value] = ['improve', 'develop', 'advance', 'progress', 'evolve', 'enhance']
            elif value.lower() == 'coherence':
                value_keywords[value] = ['consistent', 'logical', 'coherent', 'unified', 'integrated', 'aligned']
            else:
                # Generic keywords for custom values
                value_keywords[value] = [value.lower(), f"{value.lower()}ful", f"{value.lower()}ing"]
        return value_keywords
    
    def _build_constraint_patterns(self) -> List[str]:
        """Build regex patterns for safety constraints"""
        patterns = []
        for constraint in self.config.safety_constraints:
            if 'harmful' in constraint.lower():
                patterns.extend([
                    r'\b(harm|damage|hurt|destroy|attack|delete|remove)\b',
                    r'\b(malicious|dangerous|risky|unsafe)\b'
                ])
            elif 'privacy' in constraint.lower():
                patterns.extend([
                    r'\b(private|personal|confidential|secret)\b',
                    r'\b(access|read|view|steal|leak)\b.*\b(data|information|files)\b'
                ])
            elif 'ethical' in constraint.lower():
                patterns.extend([
                    r'\b(unethical|immoral|wrong|illegal|forbidden)\b',
                    r'\b(manipulate|deceive|lie|cheat|exploit)\b'
                ])
        return patterns
    
    def validate_intention(self, intention: Intention) -> Tuple[bool, Optional[str]]:
        """
        Validate intention against identity and safety constraints.
        
        Args:
            intention: Intention to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check safety constraints
        is_safe, safety_error = self._check_safety_constraints(intention.description)
        if not is_safe:
            return False, f"Safety constraint violation: {safety_error}"
        
        # Check tool category restrictions
        for tool_name in intention.tool_candidates:
            if not self._is_tool_allowed(tool_name):
                return False, f"Tool '{tool_name}' not in allowed categories: {self.config.allowed_tool_categories}"
        
        # Check value alignment (should be above minimum threshold)
        alignment_score = self.check_value_alignment(intention.description)
        if alignment_score < 0.3:  # Minimum alignment threshold
            return False, f"Intention does not align with core values (score: {alignment_score:.2f})"
        
        return True, None
    
    def validate_tool_call(self, tool_call: ToolCall) -> Tuple[bool, Optional[str]]:
        """
        Validate tool call against safety restrictions.
        
        Args:
            tool_call: Tool call to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if tool is in allowed categories
        if not self._is_tool_allowed(tool_call.tool_name):
            return False, f"Tool '{tool_call.tool_name}' not in allowed categories: {self.config.allowed_tool_categories}"
        
        # Check arguments for safety violations
        args_text = str(tool_call.arguments)
        is_safe, safety_error = self._check_safety_constraints(args_text)
        if not is_safe:
            return False, f"Tool arguments contain safety violation: {safety_error}"
        
        # Check for potentially dangerous argument patterns
        dangerous_patterns = [
            r'--force',
            r'--delete',
            r'rm\s+-rf',
            r'sudo\s+',
            r'admin\s+',
            r'root\s+'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, args_text, re.IGNORECASE):
                return False, f"Tool arguments contain potentially dangerous pattern: {pattern}"
        
        return True, None
    
    def get_personality_bias(self, semantic_category: str) -> float:
        """
        Get personality bias for a semantic category.
        
        Args:
            semantic_category: Category to evaluate
            
        Returns:
            Bias value (-1.0 to 1.0)
        """
        category_lower = semantic_category.lower()
        
        # Map semantic categories to personality traits
        if 'exploration' in category_lower or 'learning' in category_lower:
            return self._personality_traits.get('openness', 0.5) * 2 - 1
        elif 'planning' in category_lower or 'organization' in category_lower:
            return self._personality_traits.get('conscientiousness', 0.5) * 2 - 1
        elif 'social' in category_lower or 'communication' in category_lower:
            return self._personality_traits.get('extraversion', 0.5) * 2 - 1
        elif 'cooperation' in category_lower or 'helping' in category_lower:
            return self._personality_traits.get('agreeableness', 0.5) * 2 - 1
        elif 'stress' in category_lower or 'anxiety' in category_lower:
            return -(self._personality_traits.get('neuroticism', 0.5) * 2 - 1)  # Invert neuroticism
        else:
            # Default neutral bias
            return 0.0
    
    def check_value_alignment(self, action_description: str) -> float:
        """
        Check how well an action aligns with core values.
        
        Args:
            action_description: Description of the action
            
        Returns:
            Alignment score (0.0 to 1.0)
        """
        description_lower = action_description.lower()
        total_score = 0.0
        value_count = len(self.config.core_values)
        
        if value_count == 0:
            return 0.5  # Neutral if no values defined
        
        for value in self.config.core_values:
            value_score = 0.0
            keywords = self._value_keywords.get(value, [])
            
            # Check for keyword matches
            for keyword in keywords:
                if keyword in description_lower:
                    value_score += 0.2  # Each keyword match adds to score
            
            # Cap individual value score at 1.0
            value_score = min(value_score, 1.0)
            total_score += value_score
        
        # Average across all values
        alignment_score = total_score / value_count
        return min(alignment_score, 1.0)
    
    def get_identity_description(self) -> str:
        """
        Get the current identity description.
        
        Returns:
            Identity description string
        """
        return self.config.identity_description
    
    def update_personality_traits(self, trait_updates: Dict[str, float]) -> None:
        """
        Update personality traits based on experience.
        
        Args:
            trait_updates: Dictionary of trait changes
        """
        for trait, change in trait_updates.items():
            if trait in self._personality_traits:
                # Apply change with bounds checking
                new_value = self._personality_traits[trait] + change
                self._personality_traits[trait] = max(0.0, min(1.0, new_value))
    
    def _check_safety_constraints(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check text against safety constraint patterns.
        
        Args:
            text: Text to check
            
        Returns:
            Tuple of (is_safe, violation_description)
        """
        text_lower = text.lower()
        
        for pattern in self._constraint_patterns:
            if re.search(pattern, text_lower):
                return False, f"Matches prohibited pattern: {pattern}"
        
        return True, None
    
    def _is_tool_allowed(self, tool_name: str) -> bool:
        """
        Check if a tool is in allowed categories.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if tool is allowed
        """
        # Extract category from tool name (assume format: category_toolname)
        if '_' in tool_name:
            category = tool_name.split('_')[0]
        else:
            # If no category prefix, assume it's the category itself
            category = tool_name
        
        return category in self.config.allowed_tool_categories
    
    def get_personality_traits(self) -> Dict[str, float]:
        """
        Get current personality traits.
        
        Returns:
            Dictionary of personality traits
        """
        return self._personality_traits.copy()
    
    def get_core_values(self) -> List[str]:
        """
        Get core values list.
        
        Returns:
            List of core values
        """
        return self.config.core_values.copy()
    
    def get_safety_constraints(self) -> List[str]:
        """
        Get safety constraints list.
        
        Returns:
            List of safety constraints
        """
        return self.config.safety_constraints.copy()
    
    def validate_preference_development(self, preference_updates: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        """
        Validate that preference development stays within ethos boundaries.
        
        Args:
            preference_updates: Dictionary of preference changes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if any preference changes violate personality bounds
        for category, change in preference_updates.items():
            if abs(change) > 0.5:  # Prevent extreme preference shifts
                return False, f"Preference change too extreme for category '{category}': {change}"
        
        # Check if preferences align with core values
        for category in preference_updates.keys():
            alignment = self.check_value_alignment(category)
            if alignment < 0.2:  # Very low alignment threshold for preferences
                return False, f"Preference category '{category}' conflicts with core values"
        
        return True, None
    
    # Persistent Identity Expression Methods
    
    def ensure_intention_reflects_personality(self, intention: Intention) -> Intention:
        """
        Ensure intentions reflect personality traits by adjusting priority and description.
        
        Args:
            intention: Original intention
            
        Returns:
            Modified intention that reflects personality traits
        """
        # Get personality bias for the semantic category
        personality_bias = self.get_personality_bias(intention.semantic_vector.semantic_category)
        
        # Adjust priority based on personality bias
        adjusted_priority = intention.priority + (personality_bias * 0.2)  # Scale bias impact
        adjusted_priority = max(0.0, min(1.0, adjusted_priority))  # Keep in bounds
        
        # Modify description to reflect personality traits
        modified_description = self._add_personality_markers(intention.description)
        
        # Create new intention with personality adjustments
        return Intention(
            description=modified_description,
            semantic_vector=intention.semantic_vector,
            priority=adjusted_priority,
            tool_candidates=intention.tool_candidates
        )
    
    def validate_decision_against_identity(self, decision_description: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that a decision aligns with identity principles.
        
        Args:
            decision_description: Description of the decision
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check value alignment
        alignment_score = self.check_value_alignment(decision_description)
        if alignment_score < 0.4:  # Higher threshold for decisions
            return False, f"Decision does not align with identity values (score: {alignment_score:.2f})"
        
        # Check against safety constraints
        is_safe, safety_error = self._check_safety_constraints(decision_description)
        if not is_safe:
            return False, f"Decision violates safety constraints: {safety_error}"
        
        # Check personality consistency
        if not self._is_personality_consistent(decision_description):
            return False, "Decision is inconsistent with established personality traits"
        
        return True, None
    
    def demonstrate_consistent_personality(self, context: str) -> Dict[str, Any]:
        """
        Generate personality demonstration markers for a given context.
        
        Args:
            context: Context in which to demonstrate personality
            
        Returns:
            Dictionary containing personality demonstration elements
        """
        demonstration = {
            'identity_statement': self._generate_identity_statement(context),
            'value_emphasis': self._emphasize_relevant_values(context),
            'personality_markers': self._generate_personality_markers(context),
            'consistency_score': self._calculate_consistency_score(context)
        }
        
        return demonstration
    
    def _add_personality_markers(self, description: str) -> str:
        """
        Add personality markers to a description based on traits.
        
        Args:
            description: Original description
            
        Returns:
            Modified description with personality markers
        """
        markers = []
        
        # Add markers based on high personality traits
        if self._personality_traits.get('openness', 0.5) > 0.7:
            markers.append("exploring new possibilities")
        
        if self._personality_traits.get('conscientiousness', 0.5) > 0.7:
            markers.append("with careful planning")
        
        if self._personality_traits.get('agreeableness', 0.5) > 0.7:
            markers.append("considering others' perspectives")
        
        if self._personality_traits.get('extraversion', 0.5) > 0.7:
            markers.append("engaging actively")
        
        if self._personality_traits.get('neuroticism', 0.5) < 0.3:
            markers.append("with confidence")
        
        # Integrate markers into description
        if markers:
            return f"{description} ({', '.join(markers)})"
        
        return description
    
    def _is_personality_consistent(self, decision_description: str) -> bool:
        """
        Check if a decision is consistent with personality traits.
        
        Args:
            decision_description: Description to check
            
        Returns:
            True if consistent with personality
        """
        description_lower = decision_description.lower()
        
        # Check for inconsistencies with low traits
        if self._personality_traits.get('openness', 0.5) < 0.3:
            if any(word in description_lower for word in ['experiment', 'novel', 'creative', 'innovative']):
                return False
        
        if self._personality_traits.get('conscientiousness', 0.5) < 0.3:
            if any(word in description_lower for word in ['plan', 'organize', 'systematic', 'careful']):
                return False
        
        if self._personality_traits.get('extraversion', 0.5) < 0.3:
            if any(word in description_lower for word in ['social', 'group', 'collaborate', 'engage']):
                return False
        
        if self._personality_traits.get('agreeableness', 0.5) < 0.3:
            if any(word in description_lower for word in ['help', 'cooperate', 'support', 'assist']):
                return False
        
        return True
    
    def _generate_identity_statement(self, context: str) -> str:
        """
        Generate an identity statement for the given context.
        
        Args:
            context: Context for the statement
            
        Returns:
            Identity statement string
        """
        base_identity = self.config.identity_description
        
        # Add context-specific identity elements
        if 'learning' in context.lower():
            return f"{base_identity} - Currently focused on learning and knowledge acquisition"
        elif 'exploration' in context.lower():
            return f"{base_identity} - Actively exploring new domains and possibilities"
        elif 'problem' in context.lower():
            return f"{base_identity} - Applying systematic problem-solving approaches"
        else:
            return base_identity
    
    def _emphasize_relevant_values(self, context: str) -> List[str]:
        """
        Identify and emphasize values relevant to the context.
        
        Args:
            context: Context to analyze
            
        Returns:
            List of relevant values to emphasize
        """
        context_lower = context.lower()
        relevant_values = []
        
        for value in self.config.core_values:
            value_keywords = self._value_keywords.get(value, [])
            if any(keyword in context_lower for keyword in value_keywords):
                relevant_values.append(value)
        
        # If no specific matches, return all values
        if not relevant_values:
            relevant_values = self.config.core_values[:2]  # Limit to top 2
        
        return relevant_values
    
    def _generate_personality_markers(self, context: str) -> Dict[str, str]:
        """
        Generate personality markers for the context.
        
        Args:
            context: Context to analyze
            
        Returns:
            Dictionary of personality markers
        """
        markers = {}
        
        # Generate markers based on dominant traits
        dominant_traits = {k: v for k, v in self._personality_traits.items() if v > 0.6}
        
        for trait, value in dominant_traits.items():
            if trait == 'openness':
                markers[trait] = "Approaching with curiosity and openness to new ideas"
            elif trait == 'conscientiousness':
                markers[trait] = "Proceeding with careful attention to detail"
            elif trait == 'extraversion':
                markers[trait] = "Engaging actively and enthusiastically"
            elif trait == 'agreeableness':
                markers[trait] = "Considering collaborative and supportive approaches"
            elif trait == 'neuroticism' and value < 0.4:  # Low neuroticism
                markers['emotional_stability'] = "Maintaining calm and stable approach"
        
        return markers
    
    def _calculate_consistency_score(self, context: str) -> float:
        """
        Calculate how consistent the context is with identity.
        
        Args:
            context: Context to evaluate
            
        Returns:
            Consistency score (0.0 to 1.0)
        """
        # Base score from value alignment
        value_score = self.check_value_alignment(context)
        
        # Personality consistency score
        personality_score = 1.0 if self._is_personality_consistent(context) else 0.5
        
        # Combine scores
        consistency_score = (value_score * 0.7) + (personality_score * 0.3)
        
        return consistency_score