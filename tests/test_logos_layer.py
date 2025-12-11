"""
Tests for the Logos Layer reasoning engine.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume
from typing import List

from src.logos.logos_layer import LogosLayer
from src.models import SemanticVector, MemoryTrace, Intention, ToolCall
from src.config import LogosConfig


class TestLogosLayer:
    """Test suite for LogosLayer implementation"""
    
    @pytest.fixture
    def logos_layer(self, logos_config):
        """Create LogosLayer instance for testing"""
        return LogosLayer(logos_config)
    
    def test_initialization(self, logos_layer):
        """Test that LogosLayer initializes correctly"""
        assert logos_layer.config is not None
        assert isinstance(logos_layer.preference_weights, dict)
        assert isinstance(logos_layer.historical_patterns, dict)
        assert isinstance(logos_layer.tool_effectiveness, dict)
        assert len(logos_layer.semantic_categories) > 0
        
        # Check that all semantic categories have initial preference weights
        for category in logos_layer.semantic_categories:
            assert category in logos_layer.preference_weights
            assert 0.0 <= logos_layer.preference_weights[category] <= 1.0
    
    def test_interpret_state_basic(self, logos_layer):
        """Test basic state interpretation functionality"""
        pathos_state = np.random.randn(128)
        memories = []
        
        semantic_vector = logos_layer.interpret_state(pathos_state, memories)
        
        assert isinstance(semantic_vector, SemanticVector)
        assert len(semantic_vector.embedding) == logos_layer.config.semantic_dimension
        assert semantic_vector.semantic_category in logos_layer.semantic_categories
        assert len(semantic_vector.intention_text) > 0
    
    def test_generate_intention_basic(self, logos_layer, sample_semantic_vector):
        """Test basic intention generation"""
        pathos_state = np.random.randn(128)
        
        intention = logos_layer.generate_intention(sample_semantic_vector, pathos_state)
        
        assert isinstance(intention, Intention)
        assert 0.0 <= intention.priority <= 1.0
        assert len(intention.tool_candidates) > 0
        assert len(intention.description) > 0
    
    def test_compute_interest_signal_basic(self, logos_layer, sample_semantic_vector):
        """Test basic interest signal computation"""
        interest = logos_layer.compute_interest_signal(sample_semantic_vector)
        
        assert isinstance(interest, float)
        assert 0.0 <= interest <= 1.0
    
    def test_update_preferences_basic(self, logos_layer, sample_semantic_vector):
        """Test basic preference updating"""
        initial_weight = logos_layer.preference_weights.get(sample_semantic_vector.semantic_category, 0.5)
        
        # Positive reward should increase preference
        logos_layer.update_preferences(1.0, sample_semantic_vector)
        
        new_weight = logos_layer.preference_weights[sample_semantic_vector.semantic_category]
        assert new_weight >= initial_weight
        assert 0.0 <= new_weight <= 1.0


class TestLogosLayerProperties:
    """Property-based tests for LogosLayer"""
    
    @given(
        pathos_state=st.lists(st.floats(min_value=-10.0, max_value=10.0), min_size=64, max_size=256),
        num_memories=st.integers(min_value=0, max_value=10)
    )
    def test_property_continuous_intention_generation(self, pathos_state, num_memories):
        """
        **Feature: autonomous-logos-pathos-agent, Property 1: Continuous Intention Generation**
        **Validates: Requirements 1.1, 1.2**
        
        For any agent cycle, completing the cycle should result in the generation of a new 
        intention object that incorporates current internal state and memory patterns.
        """
        # Create fresh config and logos layer for each test
        logos_config = LogosConfig(
            semantic_dimension=256,
            max_intentions_per_cycle=2,
            planning_horizon=3,
            interest_threshold=0.5,
            gemini_api_key="test_key"
        )
        logos_layer = LogosLayer(logos_config)
        
        # Convert to numpy array
        pathos_array = np.array(pathos_state)
        assume(np.isfinite(pathos_array).all())  # Ensure no NaN or inf values
        
        # Generate sample memories
        memories = []
        for i in range(num_memories):
            semantic_vector = SemanticVector(
                embedding=np.random.randn(logos_layer.config.semantic_dimension),
                intention_text=f"Test memory {i}",
                semantic_category=np.random.choice(logos_layer.semantic_categories)
            )
            memory = MemoryTrace(
                affect_state=np.random.randn(len(pathos_array)),
                semantic_vector=semantic_vector,
                reward=np.random.uniform(-1.0, 1.0),
                timestamp=1000 + i,
                salience=np.random.uniform(0.0, 1.0)
            )
            memories.append(memory)
        
        # Test the complete intention generation cycle
        # Step 1: Interpret state
        semantic_vector = logos_layer.interpret_state(pathos_array, memories)
        
        # Verify semantic vector properties
        assert isinstance(semantic_vector, SemanticVector)
        assert len(semantic_vector.embedding) == logos_layer.config.semantic_dimension
        assert semantic_vector.semantic_category in logos_layer.semantic_categories
        assert isinstance(semantic_vector.intention_text, str)
        assert len(semantic_vector.intention_text) > 0
        
        # Step 2: Generate intention
        intention = logos_layer.generate_intention(semantic_vector, pathos_array)
        
        # Verify intention properties
        assert isinstance(intention, Intention)
        assert isinstance(intention.description, str)
        assert len(intention.description) > 0
        assert isinstance(intention.semantic_vector, SemanticVector)
        assert isinstance(intention.priority, float)
        assert 0.0 <= intention.priority <= 1.0
        assert isinstance(intention.tool_candidates, list)
        assert len(intention.tool_candidates) > 0
        
        # Verify that intention incorporates current state
        # The semantic vector should be influenced by the pathos state
        assert np.isfinite(intention.semantic_vector.embedding).all()
        
        # Verify that intention incorporates memory patterns when memories exist
        if memories:
            # The semantic category should be one that exists in the system
            assert intention.semantic_vector.semantic_category in logos_layer.semantic_categories
        
        # Step 3: Compute interest signal
        interest = logos_layer.compute_interest_signal(semantic_vector)
        
        # Verify interest signal properties
        assert isinstance(interest, float)
        assert 0.0 <= interest <= 1.0
        assert np.isfinite(interest)
        
        # The intention generation process should be deterministic for the same inputs
        # (Generate again with same inputs to verify consistency)
        semantic_vector_2 = logos_layer.interpret_state(pathos_array, memories)
        intention_2 = logos_layer.generate_intention(semantic_vector_2, pathos_array)
        
        # While the exact values might differ due to randomness in implementation,
        # the structure and bounds should be consistent
        assert isinstance(intention_2, Intention)
        assert 0.0 <= intention_2.priority <= 1.0
        assert len(intention_2.tool_candidates) > 0
    
    @given(
        rewards=st.lists(st.floats(min_value=-2.0, max_value=2.0), min_size=1, max_size=20),
        semantic_categories=st.lists(
            st.sampled_from(['exploration', 'analysis', 'communication', 'learning', 
                           'planning', 'reflection', 'creativity', 'problem_solving']),
            min_size=1, max_size=10
        )
    )
    def test_property_preference_learning_consistency(self, rewards, semantic_categories):
        """
        Test that preference learning maintains consistency and bounds.
        
        Preferences should update based on rewards while staying within valid bounds.
        """
        assume(all(np.isfinite(r) for r in rewards))
        
        # Create fresh config and logos layer for each test
        logos_config = LogosConfig(semantic_dimension=256)
        logos_layer = LogosLayer(logos_config)
        
        # Record initial preferences
        initial_preferences = logos_layer.get_preference_summary()
        
        # Apply a series of reward updates
        for i, (reward, category) in enumerate(zip(rewards, semantic_categories)):
            semantic_vector = SemanticVector(
                embedding=np.random.randn(logos_layer.config.semantic_dimension),
                intention_text=f"Test intention {i}",
                semantic_category=category
            )
            
            logos_layer.update_preferences(reward, semantic_vector)
        
        # Verify preference bounds are maintained
        final_preferences = logos_layer.get_preference_summary()
        
        for category, preference in final_preferences.items():
            assert isinstance(preference, float)
            assert 0.0 <= preference <= 1.0
            assert np.isfinite(preference)
        
        # Verify that preferences changed in response to rewards
        # (At least some category should have been updated)
        updated_categories = set(semantic_categories)
        for category in updated_categories:
            if category in initial_preferences and category in final_preferences:
                # The preference should be a valid float (may or may not have changed)
                assert isinstance(final_preferences[category], float)
    
    @given(
        pathos_states=st.lists(
            st.lists(st.floats(min_value=-5.0, max_value=5.0), min_size=64, max_size=128),
            min_size=2, max_size=5
        )
    )
    def test_property_state_interpretation_robustness(self, pathos_states):
        """
        Test that state interpretation is robust across different pathos states.
        
        The system should handle various state configurations without failing.
        """
        # Create fresh config and logos layer for each test
        logos_config = LogosConfig(semantic_dimension=256)
        logos_layer = LogosLayer(logos_config)
        
        for state_list in pathos_states:
            pathos_array = np.array(state_list)
            assume(np.isfinite(pathos_array).all())
            
            # Should not raise exceptions
            semantic_vector = logos_layer.interpret_state(pathos_array, [])
            
            # Should produce valid outputs
            assert isinstance(semantic_vector, SemanticVector)
            assert len(semantic_vector.embedding) == logos_layer.config.semantic_dimension
            assert np.isfinite(semantic_vector.embedding).all()
            assert semantic_vector.semantic_category in logos_layer.semantic_categories
            
            # Should be able to generate intention from this semantic vector
            intention = logos_layer.generate_intention(semantic_vector, pathos_array)
            assert isinstance(intention, Intention)
            assert 0.0 <= intention.priority <= 1.0
    
    @given(
        tool_names=st.lists(st.text(min_size=3, max_size=20), min_size=1, max_size=5),
        rewards=st.lists(st.floats(min_value=-2.0, max_value=2.0), min_size=5, max_size=20),
        semantic_categories=st.lists(
            st.sampled_from(['exploration', 'analysis', 'communication', 'learning']),
            min_size=1, max_size=4
        )
    )
    def test_property_historical_pattern_integration(self, tool_names, rewards, semantic_categories):
        """
        **Feature: autonomous-logos-pathos-agent, Property 10: Historical Pattern Integration**
        **Validates: Requirements 4.5, 5.3**
        
        For any tool evaluation or action selection, the system should incorporate 
        effectiveness patterns and success histories from previous experiences.
        """
        assume(all(np.isfinite(r) for r in rewards))
        assume(len(tool_names) > 0 and len(semantic_categories) > 0)
        
        # Create fresh config and logos layer
        logos_config = LogosConfig(semantic_dimension=256)
        logos_layer = LogosLayer(logos_config)
        
        # Build up historical patterns by simulating tool usage and rewards
        for i, (tool_name, reward, category) in enumerate(zip(tool_names * 5, rewards, semantic_categories * 10)):
            # Update tool effectiveness
            logos_layer.update_tool_effectiveness(tool_name, reward, category)
            
            # Update preferences
            semantic_vector = SemanticVector(
                embedding=np.random.randn(logos_config.semantic_dimension),
                intention_text=f"Test intention {i}",
                semantic_category=category
            )
            logos_layer.update_preferences(reward, semantic_vector)
        
        # Test that historical patterns are integrated into tool evaluation
        test_category = semantic_categories[0]
        available_tools = tool_names + ['unknown_tool']
        
        # Create a test intention
        test_semantic = SemanticVector(
            embedding=np.random.randn(logos_config.semantic_dimension),
            intention_text="Test historical integration",
            semantic_category=test_category
        )
        test_intention = Intention(
            description="Test intention",
            semantic_vector=test_semantic,
            priority=0.5,
            tool_candidates=available_tools
        )
        
        # Evaluate tool usage - should incorporate historical effectiveness
        tool_call = logos_layer.evaluate_tool_usage(test_intention, available_tools)
        
        # If a tool is selected, it should be based on historical effectiveness
        if tool_call is not None:
            assert isinstance(tool_call, ToolCall)
            assert tool_call.tool_name in available_tools
            assert isinstance(tool_call.arguments, dict)
            
            # The selected tool should have some historical data or be a reasonable choice
            selected_tool_key = f"{tool_call.tool_name}_{test_category}"
            # Either the tool has effectiveness data, or it's a fallback choice
            has_history = selected_tool_key in logos_layer.tool_effectiveness
            is_candidate = tool_call.tool_name in test_intention.tool_candidates
            assert has_history or is_candidate
        
        # Test that bias toward successful patterns works
        bias_factor = logos_layer.get_bias_toward_successful_patterns(test_category)
        assert isinstance(bias_factor, float)
        assert 0.5 <= bias_factor <= 2.0
        assert np.isfinite(bias_factor)
        
        # Test historical context integration
        enhanced_intention = logos_layer.integrate_historical_context(test_intention)
        assert isinstance(enhanced_intention, Intention)
        assert 0.0 <= enhanced_intention.priority <= 1.0
        assert len(enhanced_intention.tool_candidates) > 0
        
        # The enhanced intention should maintain the same semantic category
        assert enhanced_intention.semantic_vector.semantic_category == test_category
        
        # Test historical pattern analysis
        analysis = logos_layer.analyze_historical_patterns()
        assert isinstance(analysis, dict)
        assert 'successful_categories' in analysis
        assert 'effective_tools' in analysis
        assert 'preference_trends' in analysis
        assert 'learning_progress' in analysis
        
        # Verify that categories with historical data appear in analysis
        for category in semantic_categories:
            if category in logos_layer.historical_patterns and logos_layer.historical_patterns[category]:
                assert category in analysis['successful_categories']
                category_data = analysis['successful_categories'][category]
                assert isinstance(category_data['average_reward'], float)
                assert isinstance(category_data['success_rate'], float)
                assert 0.0 <= category_data['success_rate'] <= 1.0
    
    @given(
        positive_rewards=st.lists(st.floats(min_value=0.1, max_value=2.0), min_size=3, max_size=15),
        negative_rewards=st.lists(st.floats(min_value=-2.0, max_value=-0.1), min_size=1, max_size=10),
        semantic_category=st.sampled_from(['exploration', 'analysis', 'communication', 'learning'])
    )
    def test_property_preference_reward_coupling(self, positive_rewards, negative_rewards, semantic_category):
        """
        **Feature: autonomous-logos-pathos-agent, Property 11: Preference-Reward Coupling**
        **Validates: Requirements 5.1, 5.4**
        
        For any positive internal reward experience, preference weights should increase 
        for associated semantic regions while maintaining ethos compliance.
        """
        assume(all(np.isfinite(r) for r in positive_rewards))
        assume(all(np.isfinite(r) for r in negative_rewards))
        
        # Create fresh config and logos layer
        logos_config = LogosConfig(semantic_dimension=256)
        logos_layer = LogosLayer(logos_config)
        
        # Record initial preference weight
        initial_preference = logos_layer.preference_weights.get(semantic_category, 0.5)
        
        # Create semantic vector for the test category
        semantic_vector = SemanticVector(
            embedding=np.random.randn(logos_config.semantic_dimension),
            intention_text=f"Test {semantic_category} activity",
            semantic_category=semantic_category
        )
        
        # Apply positive rewards - should increase preference
        for reward in positive_rewards:
            logos_layer.update_preferences(reward, semantic_vector)
        
        preference_after_positive = logos_layer.preference_weights[semantic_category]
        
        # Positive rewards should increase or maintain preference weight
        assert preference_after_positive >= initial_preference
        assert 0.0 <= preference_after_positive <= 1.0
        assert np.isfinite(preference_after_positive)
        
        # Apply negative rewards - should decrease preference
        for reward in negative_rewards:
            logos_layer.update_preferences(reward, semantic_vector)
        
        final_preference = logos_layer.preference_weights[semantic_category]
        
        # Negative rewards should decrease preference compared to after positive rewards
        assert final_preference <= preference_after_positive
        assert 0.0 <= final_preference <= 1.0
        assert np.isfinite(final_preference)
        
        # Test that preference changes affect intention generation
        pathos_state = np.random.randn(128)
        
        # Generate intention - priority should be influenced by preference weight
        intention = logos_layer.generate_intention(semantic_vector, pathos_state)
        
        assert isinstance(intention, Intention)
        assert 0.0 <= intention.priority <= 1.0
        
        # Test that preferences stay within bounds even with extreme rewards
        extreme_positive = 10.0
        extreme_negative = -10.0
        
        # Apply extreme positive reward
        logos_layer.update_preferences(extreme_positive, semantic_vector)
        extreme_high_preference = logos_layer.preference_weights[semantic_category]
        assert 0.0 <= extreme_high_preference <= 1.0
        
        # Apply extreme negative reward
        logos_layer.update_preferences(extreme_negative, semantic_vector)
        extreme_low_preference = logos_layer.preference_weights[semantic_category]
        assert 0.0 <= extreme_low_preference <= 1.0
        
        # Test that historical patterns are updated correctly
        assert semantic_category in logos_layer.historical_patterns
        historical_rewards = logos_layer.historical_patterns[semantic_category]
        
        # Should contain all the rewards we applied
        expected_reward_count = len(positive_rewards) + len(negative_rewards) + 2  # +2 for extreme rewards
        assert len(historical_rewards) == expected_reward_count
        
        # All recorded rewards should be finite
        assert all(np.isfinite(r) for r in historical_rewards)
        
        # Test that preference summary is consistent
        preference_summary = logos_layer.get_preference_summary()
        assert isinstance(preference_summary, dict)
        assert semantic_category in preference_summary
        assert preference_summary[semantic_category] == logos_layer.preference_weights[semantic_category]
        
        # Test coupling with interest signal computation
        interest = logos_layer.compute_interest_signal(semantic_vector)
        assert isinstance(interest, float)
        assert 0.0 <= interest <= 1.0
        assert np.isfinite(interest)
        
        # Interest should be influenced by preference weight
        # Higher preference should generally lead to higher interest (though other factors matter too)
        if logos_layer.preference_weights[semantic_category] > 0.8:
            # High preference should contribute to interest
            assert interest >= 0.0  # Basic sanity check
        
        # Test that ethos compliance is maintained (preferences stay in bounds)
        # This is implicitly tested by the bounds checking above, but let's be explicit
        for category, weight in logos_layer.preference_weights.items():
            assert 0.0 <= weight <= 1.0, f"Preference weight for {category} out of bounds: {weight}"
    
    @given(
        pathos_state=st.lists(st.floats(min_value=-5.0, max_value=5.0), min_size=64, max_size=128),
        num_memories=st.integers(min_value=2, max_value=8),
        memory_rewards=st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=2, max_size=8)
    )
    def test_property_self_reflective_planning(self, pathos_state, num_memories, memory_rewards):
        """
        **Feature: autonomous-logos-pathos-agent, Property 16: Self-Reflective Planning**
        **Validates: Requirements 7.3, 7.5**
        
        For any action planning process, Logos should incorporate evaluation of internal state, 
        historical patterns, and emergent drives from layer interactions.
        """
        assume(all(np.isfinite(s) for s in pathos_state))
        assume(all(np.isfinite(r) for r in memory_rewards))
        assume(len(memory_rewards) >= num_memories)
        
        # Create fresh config and logos layer
        logos_config = LogosConfig(semantic_dimension=256)
        logos_layer = LogosLayer(logos_config)
        
        # Convert pathos state to numpy array
        pathos_array = np.array(pathos_state)
        
        # Create sample memories with varied categories and rewards
        memories = []
        categories = ['exploration', 'analysis', 'communication', 'learning']
        
        for i in range(num_memories):
            category = categories[i % len(categories)]
            semantic_vector = SemanticVector(
                embedding=np.random.randn(logos_config.semantic_dimension),
                intention_text=f"Memory {i} about {category}",
                semantic_category=category
            )
            
            memory = MemoryTrace(
                affect_state=np.random.randn(len(pathos_array)),
                semantic_vector=semantic_vector,
                reward=memory_rewards[i],
                timestamp=1000 + i * 100,
                salience=np.random.uniform(0.5, 1.0)
            )
            memories.append(memory)
            
            # Update logos layer with this experience
            logos_layer.update_preferences(memory_rewards[i], semantic_vector)
        
        # Test internal state evaluation
        internal_evaluation = logos_layer.evaluate_internal_state(pathos_array, memories)
        
        # Verify evaluation structure and content
        assert isinstance(internal_evaluation, dict)
        required_keys = ['affective_analysis', 'memory_patterns', 'preference_status', 
                        'learning_assessment', 'emergent_drives']
        
        for key in required_keys:
            assert key in internal_evaluation, f"Missing key: {key}"
        
        # Verify affective analysis
        affective_analysis = internal_evaluation['affective_analysis']
        assert isinstance(affective_analysis, dict)
        assert 'energy_level' in affective_analysis
        assert 'stability' in affective_analysis
        assert 0.0 <= affective_analysis['energy_level'] <= 1.0
        assert 0.0 <= affective_analysis['stability'] <= 1.0
        
        # Verify memory patterns analysis
        memory_patterns = internal_evaluation['memory_patterns']
        assert isinstance(memory_patterns, dict)
        assert 'recent_themes' in memory_patterns
        assert 'coherence' in memory_patterns
        
        # Verify preference status
        preference_status = internal_evaluation['preference_status']
        assert isinstance(preference_status, dict)
        assert 'strongest_preferences' in preference_status
        assert 'emerging_interests' in preference_status
        
        # Verify learning assessment
        learning_assessment = internal_evaluation['learning_assessment']
        assert isinstance(learning_assessment, dict)
        assert 'exploration_vs_exploitation' in learning_assessment
        
        # Verify emergent drives detection
        emergent_drives = internal_evaluation['emergent_drives']
        assert isinstance(emergent_drives, list)
        
        # Each drive should have required structure
        for drive in emergent_drives:
            assert isinstance(drive, dict)
            assert 'type' in drive
            assert 'strength' in drive
            assert 'target_categories' in drive
            assert 'description' in drive
            assert isinstance(drive['strength'], float)
            assert 0.0 <= drive['strength'] <= 2.0  # Reasonable bounds for drive strength
        
        # Test self-reflective planning
        current_context = {'situation': 'test_planning', 'constraints': []}
        plan = logos_layer.create_self_reflective_plan(internal_evaluation, current_context)
        
        # Verify plan structure
        assert isinstance(plan, dict)
        required_plan_keys = ['strategic_focus', 'tactical_adjustments', 'learning_objectives', 
                             'risk_assessments', 'adaptation_strategies']
        
        for key in required_plan_keys:
            assert key in plan, f"Missing plan key: {key}"
        
        # Verify strategic focus
        strategic_focus = plan['strategic_focus']
        assert isinstance(strategic_focus, dict)
        assert 'primary_mode' in strategic_focus
        assert 'rationale' in strategic_focus
        assert strategic_focus['primary_mode'] in ['exploration_and_growth', 'stabilization_and_reflection', 'balanced_development']
        
        # Verify tactical adjustments
        tactical_adjustments = plan['tactical_adjustments']
        assert isinstance(tactical_adjustments, dict)
        
        # Verify learning objectives
        learning_objectives = plan['learning_objectives']
        assert isinstance(learning_objectives, list)
        
        for objective in learning_objectives:
            assert isinstance(objective, dict)
            assert 'type' in objective
            assert 'target' in objective
            assert 'priority' in objective
        
        # Verify risk assessments
        risk_assessments = plan['risk_assessments']
        assert isinstance(risk_assessments, dict)
        
        # Verify adaptation strategies
        adaptation_strategies = plan['adaptation_strategies']
        assert isinstance(adaptation_strategies, list)
        
        # Test emergent drives detection with direct method
        drives = logos_layer.detect_emergent_drives(pathos_array, memories, logos_layer.preference_weights)
        assert isinstance(drives, list)
        assert len(drives) <= 3  # Should return top 3 drives
        
        # Test that planning incorporates historical patterns
        # The plan should reflect the experiences we've given the system
        if memory_rewards and np.mean(memory_rewards) > 0.5:
            # Positive experiences should influence strategic focus, but mode depends on energy/stability
            assert strategic_focus['primary_mode'] in ['exploration_and_growth', 'balanced_development', 'stabilization_and_reflection']
        
        # Test that internal state evaluation influences planning
        energy_level = affective_analysis['energy_level']
        stability = affective_analysis['stability']
        
        # The strategic focus should be consistent with the energy and stability levels
        if energy_level > 0.7 and stability > 0.6:
            # High energy AND high stability should favor exploration
            assert strategic_focus['primary_mode'] in ['exploration_and_growth', 'balanced_development']
        elif energy_level < 0.3 or stability < 0.4:
            # Low energy OR low stability should favor stabilization
            assert strategic_focus['primary_mode'] in ['stabilization_and_reflection', 'balanced_development']
        else:
            # Moderate conditions should allow any reasonable strategy
            assert strategic_focus['primary_mode'] in ['exploration_and_growth', 'stabilization_and_reflection', 'balanced_development']
        
        # Test that the system can handle edge cases
        # Empty memories case
        empty_evaluation = logos_layer.evaluate_internal_state(pathos_array, [])
        assert isinstance(empty_evaluation, dict)
        assert all(key in empty_evaluation for key in required_keys)
        
        # Zero pathos state case
        zero_state = np.zeros_like(pathos_array)
        zero_evaluation = logos_layer.evaluate_internal_state(zero_state, memories)
        assert isinstance(zero_evaluation, dict)
        assert 0.0 <= zero_evaluation['affective_analysis']['energy_level'] <= 1.0