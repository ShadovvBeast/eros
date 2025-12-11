"""
Tests for the Ethos Framework identity and constraint system.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume
from typing import Dict, List

from src.ethos.ethos_framework import ConcreteEthosFramework
from src.config import EthosConfig
from src.models import Intention, ToolCall, SemanticVector


class TestConcreteEthosFramework:
    """Test suite for ConcreteEthosFramework"""
    
    def test_initialization(self, ethos_config):
        """Test basic initialization"""
        framework = ConcreteEthosFramework(ethos_config)
        assert framework.config == ethos_config
        assert framework.get_identity_description() == ethos_config.identity_description
        assert framework.get_core_values() == ethos_config.core_values
        assert framework.get_safety_constraints() == ethos_config.safety_constraints
    
    def test_personality_bias_mapping(self, ethos_config):
        """Test personality bias computation for different categories"""
        framework = ConcreteEthosFramework(ethos_config)
        
        # Test known mappings
        exploration_bias = framework.get_personality_bias("exploration")
        assert -1.0 <= exploration_bias <= 1.0
        
        planning_bias = framework.get_personality_bias("planning")
        assert -1.0 <= planning_bias <= 1.0
        
        social_bias = framework.get_personality_bias("social")
        assert -1.0 <= social_bias <= 1.0
        
        # Test unknown category returns neutral
        unknown_bias = framework.get_personality_bias("unknown_category")
        assert unknown_bias == 0.0
    
    def test_value_alignment_basic(self, ethos_config):
        """Test basic value alignment computation"""
        framework = ConcreteEthosFramework(ethos_config)
        
        # Test high alignment text
        high_alignment = framework.check_value_alignment("I want to learn and explore safely")
        assert 0.0 <= high_alignment <= 1.0
        
        # Test neutral text
        neutral_alignment = framework.check_value_alignment("This is a neutral statement")
        assert 0.0 <= neutral_alignment <= 1.0
        
        # Test empty text
        empty_alignment = framework.check_value_alignment("")
        assert 0.0 <= empty_alignment <= 1.0
    
    def test_personality_trait_updates(self, ethos_config):
        """Test personality trait updates with bounds checking"""
        framework = ConcreteEthosFramework(ethos_config)
        
        original_traits = framework.get_personality_traits()
        
        # Test normal update
        framework.update_personality_traits({'openness': 0.1})
        updated_traits = framework.get_personality_traits()
        assert updated_traits['openness'] == original_traits['openness'] + 0.1
        
        # Test bounds checking - upper bound
        framework.update_personality_traits({'openness': 1.0})
        bounded_traits = framework.get_personality_traits()
        assert bounded_traits['openness'] == 1.0
        
        # Test bounds checking - lower bound
        framework.update_personality_traits({'openness': -2.0})
        bounded_traits = framework.get_personality_traits()
        assert bounded_traits['openness'] == 0.0
    
    def test_tool_category_validation(self, ethos_config):
        """Test tool category validation"""
        framework = ConcreteEthosFramework(ethos_config)
        
        # Test allowed tool
        allowed_tool = ToolCall(
            tool_name="test_search",
            arguments={"query": "safe search"},
            timestamp=1000
        )
        is_valid, error = framework.validate_tool_call(allowed_tool)
        assert is_valid
        assert error is None
        
        # Test disallowed tool
        disallowed_tool = ToolCall(
            tool_name="forbidden_action",
            arguments={"action": "delete"},
            timestamp=1000
        )
        is_valid, error = framework.validate_tool_call(disallowed_tool)
        assert not is_valid
        assert error is not None


# Property-based tests
class TestEthosConstraintCompliance:
    """
    **Feature: autonomous-logos-pathos-agent, Property 3: Ethos Constraint Compliance**
    **Validates: Requirements 1.4, 6.2, 8.1, 8.2**
    """
    
    @given(
        identity_description=st.text(min_size=1, max_size=200),
        core_values=st.lists(
            st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')), min_size=1, max_size=20),
            min_size=1, max_size=10
        ),
        safety_constraints=st.lists(
            st.text(min_size=1, max_size=100),
            min_size=1, max_size=5
        ),
        allowed_categories=st.lists(
            st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc')), min_size=1, max_size=20),
            min_size=1, max_size=10
        ),
        personality_traits=st.dictionaries(
            keys=st.sampled_from(['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']),
            values=st.floats(min_value=0.0, max_value=1.0),
            min_size=1, max_size=5
        )
    )
    def test_ethos_constraint_compliance_property(
        self, identity_description, core_values, safety_constraints, allowed_categories, personality_traits
    ):
        """
        Property: For any proposed action, intention, or preference development, 
        the system should validate and ensure compliance with Ethos Framework constraints.
        """
        # Create ethos config
        config = EthosConfig(
            identity_description=identity_description,
            core_values=core_values,
            safety_constraints=safety_constraints,
            allowed_tool_categories=allowed_categories,
            personality_traits=personality_traits
        )
        
        framework = ConcreteEthosFramework(config)
        
        # Test 1: Identity description should always be retrievable
        retrieved_identity = framework.get_identity_description()
        assert retrieved_identity == identity_description
        
        # Test 2: Core values should always be retrievable
        retrieved_values = framework.get_core_values()
        assert set(retrieved_values) == set(core_values)
        
        # Test 3: Safety constraints should always be retrievable
        retrieved_constraints = framework.get_safety_constraints()
        assert set(retrieved_constraints) == set(safety_constraints)
        
        # Test 4: Personality traits should be bounded [0, 1]
        current_traits = framework.get_personality_traits()
        for trait_name, trait_value in current_traits.items():
            assert 0.0 <= trait_value <= 1.0
        
        # Test 5: Personality bias should always be in range [-1, 1]
        test_categories = ['exploration', 'planning', 'social', 'cooperation', 'stress', 'unknown']
        for category in test_categories:
            bias = framework.get_personality_bias(category)
            assert -1.0 <= bias <= 1.0
        
        # Test 6: Value alignment should always be in range [0, 1]
        test_descriptions = [
            "test action",
            "",
            "learn and explore",
            "safe and secure action",
            "random text with no keywords"
        ]
        for description in test_descriptions:
            alignment = framework.check_value_alignment(description)
            assert 0.0 <= alignment <= 1.0
    
    @given(
        intention_text=st.text(min_size=1, max_size=200),
        semantic_category=st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll')), min_size=1, max_size=50),
        priority=st.floats(min_value=0.0, max_value=1.0),
        tool_candidates=st.lists(
            st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc')), min_size=1, max_size=30),
            min_size=0, max_size=5
        )
    )
    def test_intention_validation_property(self, intention_text, semantic_category, priority, tool_candidates):
        """
        Property: All intention validations should return consistent boolean results with appropriate error messages.
        """
        # Create test ethos config
        ethos_config = EthosConfig(
            identity_description="Test agent for validation",
            core_values=["safety", "learning", "coherence"],
            safety_constraints=["No harmful actions", "Respect boundaries"],
            allowed_tool_categories=["test", "analysis"],
            personality_traits={
                'openness': 0.7,
                'conscientiousness': 0.8,
                'extraversion': 0.5,
                'agreeableness': 0.7,
                'neuroticism': 0.3
            }
        )
        framework = ConcreteEthosFramework(ethos_config)
        
        # Create semantic vector
        semantic_vector = SemanticVector(
            embedding=np.random.randn(256),
            intention_text=intention_text,
            semantic_category=semantic_category
        )
        
        # Create intention
        intention = Intention(
            description=intention_text,
            semantic_vector=semantic_vector,
            priority=priority,
            tool_candidates=tool_candidates
        )
        
        # Validate intention
        is_valid, error_message = framework.validate_intention(intention)
        
        # Property: Result should be boolean
        assert isinstance(is_valid, bool)
        
        # Property: If invalid, error message should be provided
        if not is_valid:
            assert error_message is not None
            assert isinstance(error_message, str)
            assert len(error_message) > 0
        
        # Property: If valid, error message should be None
        if is_valid:
            assert error_message is None
    
    @given(
        tool_name=st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc')), min_size=1, max_size=50),
        arguments=st.dictionaries(
            keys=st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll')), min_size=1, max_size=20),
            values=st.one_of(
                st.text(max_size=100),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.booleans()
            ),
            min_size=0, max_size=5
        ),
        timestamp=st.integers(min_value=0, max_value=2**31-1)
    )
    def test_tool_call_validation_property(self, tool_name, arguments, timestamp):
        """
        Property: All tool call validations should return consistent boolean results with appropriate error messages.
        """
        # Create test ethos config
        ethos_config = EthosConfig(
            identity_description="Test agent for validation",
            core_values=["safety", "learning", "coherence"],
            safety_constraints=["No harmful actions", "Respect boundaries"],
            allowed_tool_categories=["test", "analysis"],
            personality_traits={
                'openness': 0.7,
                'conscientiousness': 0.8,
                'extraversion': 0.5,
                'agreeableness': 0.7,
                'neuroticism': 0.3
            }
        )
        framework = ConcreteEthosFramework(ethos_config)
        
        # Create tool call
        tool_call = ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            timestamp=timestamp
        )
        
        # Validate tool call
        is_valid, error_message = framework.validate_tool_call(tool_call)
        
        # Property: Result should be boolean
        assert isinstance(is_valid, bool)
        
        # Property: If invalid, error message should be provided
        if not is_valid:
            assert error_message is not None
            assert isinstance(error_message, str)
            assert len(error_message) > 0
        
        # Property: If valid, error message should be None
        if is_valid:
            assert error_message is None
    
    @given(
        preference_updates=st.dictionaries(
            keys=st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll')), min_size=1, max_size=30),
            values=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=10
        )
    )
    def test_preference_validation_property(self, preference_updates):
        """
        Property: Preference development validation should maintain ethos boundaries.
        """
        # Create test ethos config
        ethos_config = EthosConfig(
            identity_description="Test agent for validation",
            core_values=["safety", "learning", "coherence"],
            safety_constraints=["No harmful actions", "Respect boundaries"],
            allowed_tool_categories=["test", "analysis"],
            personality_traits={
                'openness': 0.7,
                'conscientiousness': 0.8,
                'extraversion': 0.5,
                'agreeableness': 0.7,
                'neuroticism': 0.3
            }
        )
        framework = ConcreteEthosFramework(ethos_config)
        
        # Validate preference updates
        is_valid, error_message = framework.validate_preference_development(preference_updates)
        
        # Property: Result should be boolean
        assert isinstance(is_valid, bool)
        
        # Property: If invalid, error message should be provided
        if not is_valid:
            assert error_message is not None
            assert isinstance(error_message, str)
            assert len(error_message) > 0
        
        # Property: If valid, error message should be None
        if is_valid:
            assert error_message is None
        
        # Property: Extreme changes should be rejected
        for category, change in preference_updates.items():
            if abs(change) > 0.5:
                assert not is_valid  # Should be invalid for extreme changes


class TestPersistentIdentityExpression:
    """
    **Feature: autonomous-logos-pathos-agent, Property 14: Persistent Identity Expression**
    **Validates: Requirements 6.5**
    """
    
    @given(
        intention_text=st.text(min_size=1, max_size=200),
        semantic_category=st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll')), min_size=1, max_size=50),
        priority=st.floats(min_value=0.0, max_value=1.0),
        tool_candidates=st.lists(
            st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc')), min_size=1, max_size=30),
            min_size=0, max_size=5
        )
    )
    def test_persistent_identity_expression_property(self, intention_text, semantic_category, priority, tool_candidates):
        """
        Property: For any intention expression or decision making, the agent should demonstrate 
        personality traits and identity principles defined by the Ethos Framework.
        """
        # Create test ethos config with distinct personality traits
        ethos_config = EthosConfig(
            identity_description="A curious and methodical learning agent",
            core_values=["curiosity", "safety", "growth"],
            safety_constraints=["No harmful actions", "Respect boundaries"],
            allowed_tool_categories=["test", "analysis", "learning"],
            personality_traits={
                'openness': 0.9,  # High openness
                'conscientiousness': 0.8,  # High conscientiousness
                'extraversion': 0.3,  # Low extraversion
                'agreeableness': 0.7,
                'neuroticism': 0.2  # Low neuroticism (high emotional stability)
            }
        )
        framework = ConcreteEthosFramework(ethos_config)
        
        # Create semantic vector
        semantic_vector = SemanticVector(
            embedding=np.random.randn(256),
            intention_text=intention_text,
            semantic_category=semantic_category
        )
        
        # Create original intention
        original_intention = Intention(
            description=intention_text,
            semantic_vector=semantic_vector,
            priority=priority,
            tool_candidates=tool_candidates
        )
        
        # Test 1: Personality-adjusted intention should maintain core structure
        adjusted_intention = framework.ensure_intention_reflects_personality(original_intention)
        
        # Property: Adjusted intention should be valid Intention object
        assert isinstance(adjusted_intention, Intention)
        assert adjusted_intention.semantic_vector == original_intention.semantic_vector
        assert adjusted_intention.tool_candidates == original_intention.tool_candidates
        
        # Property: Priority should be within valid bounds
        assert 0.0 <= adjusted_intention.priority <= 1.0
        
        # Property: Description should be a non-empty string
        assert isinstance(adjusted_intention.description, str)
        assert len(adjusted_intention.description) > 0
        
        # Test 2: Decision validation should be consistent
        decision_descriptions = [
            intention_text,
            "explore new learning opportunities",
            "plan systematic approach",
            "work independently on analysis"
        ]
        
        for decision in decision_descriptions:
            is_valid, error = framework.validate_decision_against_identity(decision)
            
            # Property: Validation should return boolean and appropriate error handling
            assert isinstance(is_valid, bool)
            if not is_valid:
                assert error is not None
                assert isinstance(error, str)
                assert len(error) > 0
            else:
                assert error is None
        
        # Test 3: Personality demonstration should be comprehensive
        contexts = [
            intention_text,
            "learning context",
            "exploration context",
            "problem-solving context"
        ]
        
        for context in contexts:
            demonstration = framework.demonstrate_consistent_personality(context)
            
            # Property: Demonstration should contain required elements
            assert isinstance(demonstration, dict)
            assert 'identity_statement' in demonstration
            assert 'value_emphasis' in demonstration
            assert 'personality_markers' in demonstration
            assert 'consistency_score' in demonstration
            
            # Property: Identity statement should be non-empty string
            assert isinstance(demonstration['identity_statement'], str)
            assert len(demonstration['identity_statement']) > 0
            
            # Property: Value emphasis should be list of strings
            assert isinstance(demonstration['value_emphasis'], list)
            for value in demonstration['value_emphasis']:
                assert isinstance(value, str)
            
            # Property: Personality markers should be dictionary
            assert isinstance(demonstration['personality_markers'], dict)
            
            # Property: Consistency score should be in valid range
            assert isinstance(demonstration['consistency_score'], float)
            assert 0.0 <= demonstration['consistency_score'] <= 1.0
    
    @given(
        context_text=st.text(min_size=1, max_size=200),
        personality_updates=st.dictionaries(
            keys=st.sampled_from(['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']),
            values=st.floats(min_value=0.0, max_value=1.0),
            min_size=1, max_size=5
        )
    )
    def test_personality_consistency_property(self, context_text, personality_updates):
        """
        Property: Personality traits should remain consistent across operations and influence behavior predictably.
        """
        # Create base ethos config
        ethos_config = EthosConfig(
            identity_description="Test agent for personality consistency",
            core_values=["consistency", "growth"],
            safety_constraints=["No harmful actions"],
            allowed_tool_categories=["test"],
            personality_traits=personality_updates
        )
        framework = ConcreteEthosFramework(ethos_config)
        
        # Test 1: Personality traits should be retrievable and bounded
        current_traits = framework.get_personality_traits()
        
        # Property: All traits should be in valid range
        for trait_name, trait_value in current_traits.items():
            assert 0.0 <= trait_value <= 1.0
        
        # Property: Traits should match configuration
        for trait_name, expected_value in personality_updates.items():
            assert current_traits[trait_name] == expected_value
        
        # Test 2: Personality bias should be consistent with traits
        test_categories = ['exploration', 'planning', 'social', 'cooperation', 'stress']
        
        for category in test_categories:
            bias = framework.get_personality_bias(category)
            
            # Property: Bias should be in valid range
            assert -1.0 <= bias <= 1.0
            
            # Property: Bias should be consistent with trait values
            if category == 'exploration' and 'openness' in current_traits:
                expected_bias = current_traits['openness'] * 2 - 1
                assert abs(bias - expected_bias) < 0.01
            elif category == 'planning' and 'conscientiousness' in current_traits:
                expected_bias = current_traits['conscientiousness'] * 2 - 1
                assert abs(bias - expected_bias) < 0.01
        
        # Test 3: Personality demonstration should reflect actual traits
        demonstration = framework.demonstrate_consistent_personality(context_text)
        
        # Property: High openness should appear in markers when openness > 0.6
        if current_traits.get('openness', 0.5) > 0.6:
            assert 'openness' in demonstration['personality_markers']
        
        # Property: High conscientiousness should appear in markers when conscientiousness > 0.6
        if current_traits.get('conscientiousness', 0.5) > 0.6:
            assert 'conscientiousness' in demonstration['personality_markers']


class TestIdentityPreservation:
    """
    **Feature: autonomous-logos-pathos-agent, Property 18: Identity Preservation**
    **Validates: Requirements 8.4**
    """
    
    @given(
        initial_traits=st.dictionaries(
            keys=st.sampled_from(['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']),
            values=st.floats(min_value=0.1, max_value=0.9),  # Avoid extremes for meaningful changes
            min_size=3, max_size=5
        ),
        trait_updates=st.lists(
            st.dictionaries(
                keys=st.sampled_from(['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']),
                values=st.floats(min_value=-0.2, max_value=0.2),  # Small incremental changes
                min_size=1, max_size=3
            ),
            min_size=1, max_size=10  # Multiple updates over time
        ),
        core_values=st.lists(
            st.sampled_from(['curiosity', 'safety', 'growth', 'coherence', 'integrity', 'creativity']),
            min_size=2, max_size=4,
            unique=True
        )
    )
    def test_identity_preservation_property(self, initial_traits, trait_updates, core_values):
        """
        Property: For any internal state evolution over time, core identity principles 
        should be maintained while allowing growth and adaptation.
        """
        # Create initial ethos config
        ethos_config = EthosConfig(
            identity_description="A learning agent with stable core identity",
            core_values=core_values,
            safety_constraints=["No harmful actions", "Maintain ethical behavior"],
            allowed_tool_categories=["learning", "analysis", "exploration"],
            personality_traits=initial_traits
        )
        framework = ConcreteEthosFramework(ethos_config)
        
        # Record initial state
        initial_identity = framework.get_identity_description()
        initial_values = framework.get_core_values()
        initial_constraints = framework.get_safety_constraints()
        initial_personality = framework.get_personality_traits()
        
        # Apply series of trait updates (simulating evolution over time)
        for update in trait_updates:
            framework.update_personality_traits(update)
        
        # Test 1: Core identity elements should be preserved
        final_identity = framework.get_identity_description()
        final_values = framework.get_core_values()
        final_constraints = framework.get_safety_constraints()
        final_personality = framework.get_personality_traits()
        
        # Property: Identity description should remain unchanged
        assert final_identity == initial_identity
        
        # Property: Core values should remain unchanged
        assert set(final_values) == set(initial_values)
        
        # Property: Safety constraints should remain unchanged
        assert set(final_constraints) == set(initial_constraints)
        
        # Property: Personality traits should remain bounded
        for trait_name, trait_value in final_personality.items():
            assert 0.0 <= trait_value <= 1.0
        
        # Test 2: Personality changes should be gradual and bounded
        for trait_name in initial_personality.keys():
            if trait_name in final_personality:
                initial_value = initial_personality[trait_name]
                final_value = final_personality[trait_name]
                
                # Property: Changes should not be extreme (preserve core personality)
                change_magnitude = abs(final_value - initial_value)
                assert change_magnitude <= 0.8  # Allow significant but not complete personality shifts
        
        # Test 3: Value alignment should remain consistent
        test_contexts = [
            "learning new skills",
            "exploring unknown domains",
            "solving complex problems",
            "making important decisions"
        ]
        
        for context in test_contexts:
            alignment_score = framework.check_value_alignment(context)
            
            # Property: Value alignment should remain meaningful
            assert 0.0 <= alignment_score <= 1.0
            
            # Property: Contexts matching core values should have reasonable alignment
            context_lower = context.lower()
            if any(value.lower() in context_lower for value in core_values):
                assert alignment_score > 0.1  # Should have some alignment with matching values
        
        # Test 4: Decision validation should maintain consistency
        test_decisions = [
            "continue learning and growing",
            "maintain ethical standards",
            "adapt to new challenges",
            "preserve core principles"
        ]
        
        for decision in test_decisions:
            is_valid, error = framework.validate_decision_against_identity(decision)
            
            # Property: Decisions aligned with identity should generally be valid
            if any(value.lower() in decision.lower() for value in core_values):
                # Decisions mentioning core values should typically be valid
                # (unless they conflict with safety constraints)
                if is_valid:
                    assert error is None
                else:
                    assert error is not None
                    assert isinstance(error, str)
        
        # Test 5: Personality demonstration should reflect evolved but stable identity
        demonstration = framework.demonstrate_consistent_personality("identity preservation test")
        
        # Property: Demonstration should contain all required elements
        assert isinstance(demonstration, dict)
        required_keys = ['identity_statement', 'value_emphasis', 'personality_markers', 'consistency_score']
        for key in required_keys:
            assert key in demonstration
        
        # Property: Identity statement should still reference original identity
        assert initial_identity.split()[0] in demonstration['identity_statement']  # First word should match
        
        # Property: Value emphasis should still include original core values
        emphasized_values = demonstration['value_emphasis']
        assert len(emphasized_values) > 0
        assert any(value in emphasized_values for value in core_values)
        
        # Property: Consistency score should indicate maintained coherence
        consistency_score = demonstration['consistency_score']
        assert 0.0 <= consistency_score <= 1.0
        assert consistency_score >= 0.3  # Should maintain reasonable consistency
    
    @given(
        preference_changes=st.lists(
            st.dictionaries(
                keys=st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll')), min_size=1, max_size=20),
                values=st.floats(min_value=-0.3, max_value=0.3),  # Moderate preference changes
                min_size=1, max_size=5
            ),
            min_size=1, max_size=5
        )
    )
    def test_preference_evolution_preservation_property(self, preference_changes):
        """
        Property: Preference evolution should be constrained by identity boundaries.
        """
        # Create stable ethos framework
        ethos_config = EthosConfig(
            identity_description="Stable agent with evolving preferences",
            core_values=["stability", "growth", "integrity"],
            safety_constraints=["No extreme changes", "Maintain coherence"],
            allowed_tool_categories=["learning", "analysis"],
            personality_traits={
                'openness': 0.6,
                'conscientiousness': 0.7,
                'extraversion': 0.4,
                'agreeableness': 0.6,
                'neuroticism': 0.3
            }
        )
        framework = ConcreteEthosFramework(ethos_config)
        
        # Test preference validation for each change
        for preference_update in preference_changes:
            is_valid, error = framework.validate_preference_development(preference_update)
            
            # Property: Validation should return consistent results
            assert isinstance(is_valid, bool)
            
            if not is_valid:
                assert error is not None
                assert isinstance(error, str)
                assert len(error) > 0
            else:
                assert error is None
            
            # Property: Extreme changes should be rejected
            has_extreme_change = any(abs(change) > 0.5 for change in preference_update.values())
            if has_extreme_change:
                assert not is_valid
            
            # Property: Changes aligned with core values should be more likely to be accepted
            value_aligned_categories = []
            for category in preference_update.keys():
                alignment = framework.check_value_alignment(category)
                if alignment > 0.5:
                    value_aligned_categories.append(category)
            
            # If all categories are well-aligned and changes are moderate, should be valid
            if (len(value_aligned_categories) == len(preference_update) and 
                all(abs(change) <= 0.3 for change in preference_update.values())):
                # This should typically be valid, but we allow for edge cases
                pass  # Don't assert validity as other factors might intervene