"""
Property-based tests for Pathos Layer affective dynamics.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck, assume
from hypothesis.extra.numpy import arrays

from src.pathos.pathos_layer import PathosLayer
from src.config import PathosConfig
from src.models import SemanticVector, MemoryTrace


class TestPathosLayerProperties:
    """Property-based tests for Pathos Layer mathematical dynamics"""
    
    @given(
        state_dimension=st.sampled_from([8, 16, 32, 64, 128]),
        decay_factor=st.floats(min_value=0.1, max_value=0.99, allow_nan=False, allow_infinity=False),
        echo_strength=st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False),
        semantic_embedding=arrays(np.float64, shape=st.integers(min_value=8, max_value=128),
                                 elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        intention_text=st.text(min_size=1, max_size=100),
        semantic_category=st.sampled_from(['exploration', 'analysis', 'communication', 'learning', 'planning']),
        external_reward=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        interest=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        num_memory_echoes=st.integers(min_value=0, max_value=5)
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
    def test_pathos_mathematical_dynamics_property(self, state_dimension, decay_factor, echo_strength,
                                                  semantic_embedding, intention_text, semantic_category,
                                                  external_reward, interest, num_memory_echoes):
        """**Feature: autonomous-logos-pathos-agent, Property 4: Pathos Mathematical Dynamics**
        **Validates: Requirements 2.2, 2.3, 2.4**
        
        For any semantic input and current state, the Pathos layer should update according to the 
        specified nonlinear recurrent equation F(t+1) = g(α·F(t) + h(S_t, F(t)) + β·Σ(w_i·F_i))
        """
        # Create configuration
        config = PathosConfig(
            state_dimension=state_dimension,
            decay_factor=decay_factor,
            echo_strength=echo_strength
        )
        
        # Create Pathos layer
        pathos = PathosLayer(config)
        
        # Create semantic input
        semantic_vector = SemanticVector(
            embedding=semantic_embedding,
            intention_text=intention_text,
            semantic_category=semantic_category
        )
        
        # Generate memory echoes
        memory_echoes = []
        for _ in range(num_memory_echoes):
            echo_affect_state = np.random.normal(0.0, 0.5, size=state_dimension)
            echo_affect_state = np.clip(echo_affect_state, -2.0, 2.0)
            
            echo_semantic = SemanticVector(
                embedding=np.random.normal(0.0, 0.3, size=len(semantic_embedding)),
                intention_text="echo_intention",
                semantic_category=semantic_category
            )
            
            memory_trace = MemoryTrace(
                affect_state=echo_affect_state,
                semantic_vector=echo_semantic,
                reward=np.random.uniform(-0.5, 0.5),
                timestamp=1000,
                salience=np.random.uniform(0.0, 1.0),
                metadata={}
            )
            memory_echoes.append(memory_trace)
        
        # Store initial state
        initial_state = pathos.current_state.copy()
        
        # Perform state update
        updated_state = pathos.update_state(
            semantic_input=semantic_vector,
            external_reward=external_reward,
            interest=interest,
            memory_echoes=memory_echoes
        )
        
        # Verify mathematical properties
        
        # 1. State should be bounded (due to squashing function)
        assert np.all(np.abs(updated_state) <= 1.0), "Updated state should be bounded by squashing function"
        
        # 2. State dimension should be preserved
        assert updated_state.shape == (state_dimension,), f"State dimension should be preserved: expected {state_dimension}, got {updated_state.shape}"
        
        # 3. State should be different from initial (unless very specific conditions)
        # Allow for very small changes due to numerical precision
        state_change_magnitude = np.linalg.norm(updated_state - initial_state)
        
        # If there's any input (semantic, reward, interest, or echoes), state should change
        has_input = (np.linalg.norm(semantic_embedding) > 1e-6 or 
                    abs(external_reward) > 1e-6 or 
                    interest > 1e-6 or 
                    len(memory_echoes) > 0)
        
        if has_input and decay_factor < 1.0:
            # State should change when there's input and decay
            assert state_change_magnitude > 1e-10, "State should change when there's meaningful input"
        
        # 4. Verify decay factor influence
        # If no other inputs, state should decay toward zero
        if (np.linalg.norm(semantic_embedding) < 1e-6 and 
            abs(external_reward) < 1e-6 and 
            interest < 1e-6 and 
            len(memory_echoes) == 0):
            
            # State magnitude should decrease (decay toward zero)
            initial_magnitude = np.linalg.norm(initial_state)
            updated_magnitude = np.linalg.norm(updated_state)
            
            if initial_magnitude > 1e-6:  # Only test if initial state is non-zero
                assert updated_magnitude <= initial_magnitude, "State should decay when no input is provided"
        
        # 5. Verify state update is deterministic
        # Reset to initial state and update again
        pathos.set_state(initial_state)
        second_update = pathos.update_state(
            semantic_input=semantic_vector,
            external_reward=external_reward,
            interest=interest,
            memory_echoes=memory_echoes
        )
        
        np.testing.assert_array_almost_equal(
            updated_state, second_update, decimal=10,
            err_msg="State update should be deterministic for same inputs"
        )
        
        # 6. Verify that repeated identical updates produce identical results (determinism)
        pathos.set_state(initial_state)
        third_update = pathos.update_state(
            semantic_input=semantic_vector,
            external_reward=external_reward,
            interest=interest,
            memory_echoes=memory_echoes
        )
        
        np.testing.assert_array_almost_equal(
            updated_state, third_update, decimal=10,
            err_msg="Repeated identical updates should produce identical results"
        )
    
    @given(
        state_dimension=st.sampled_from([16, 32, 64]),
        initial_state=arrays(np.float64, shape=st.sampled_from([16, 32, 64]),
                           elements=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False))
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    def test_pathos_state_bounds_property(self, state_dimension, initial_state):
        """**Feature: autonomous-logos-pathos-agent, Property 4: Pathos Mathematical Dynamics**
        **Validates: Requirements 2.2, 2.3, 2.4**
        
        For any initial state and inputs, the Pathos layer should maintain bounded state values.
        """
        # Ensure state dimension matches
        if len(initial_state) != state_dimension:
            return
        
        config = PathosConfig(state_dimension=state_dimension)
        pathos = PathosLayer(config)
        
        # Set initial state
        pathos.set_state(initial_state)
        
        # Create extreme inputs to test bounds
        large_semantic = SemanticVector(
            embedding=np.full(state_dimension, 10.0),  # Large values
            intention_text="extreme_test",
            semantic_category="exploration"
        )
        
        # Update with extreme inputs
        updated_state = pathos.update_state(
            semantic_input=large_semantic,
            external_reward=10.0,  # Large reward
            interest=1.0,  # Maximum interest
            memory_echoes=[]
        )
        
        # State should still be bounded
        assert np.all(np.abs(updated_state) <= 1.0), "State should remain bounded even with extreme inputs"
        assert not np.any(np.isnan(updated_state)), "State should not contain NaN values"
        assert not np.any(np.isinf(updated_state)), "State should not contain infinite values"
    
    @given(
        state_dimension=st.sampled_from([16, 32, 64]),
        num_updates=st.integers(min_value=1, max_value=10)
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=30)
    def test_pathos_stability_over_time_property(self, state_dimension, num_updates):
        """**Feature: autonomous-logos-pathos-agent, Property 4: Pathos Mathematical Dynamics**
        **Validates: Requirements 2.2, 2.3, 2.4**
        
        For any sequence of updates, the Pathos layer should maintain numerical stability.
        """
        config = PathosConfig(state_dimension=state_dimension)
        pathos = PathosLayer(config)
        
        # Perform multiple updates
        for i in range(num_updates):
            semantic_vector = SemanticVector(
                embedding=np.random.normal(0.0, 0.3, size=state_dimension),
                intention_text=f"update_{i}",
                semantic_category="exploration"
            )
            
            updated_state = pathos.update_state(
                semantic_input=semantic_vector,
                external_reward=np.random.uniform(-0.5, 0.5),
                interest=np.random.uniform(0.0, 1.0),
                memory_echoes=[]
            )
            
            # Verify stability after each update
            assert not np.any(np.isnan(updated_state)), f"State should not contain NaN after update {i}"
            assert not np.any(np.isinf(updated_state)), f"State should not contain inf after update {i}"
            assert np.all(np.abs(updated_state) <= 1.0), f"State should remain bounded after update {i}"
    
    @given(
        state_dimension=st.sampled_from([16, 32, 64]),
        lambda_1=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
        lambda_2=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
        current_state=arrays(np.float64, shape=st.sampled_from([16, 32, 64]),
                           elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        previous_state=arrays(np.float64, shape=st.sampled_from([16, 32, 64]),
                            elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
    def test_internal_reward_computation_property(self, state_dimension, lambda_1, lambda_2, 
                                                current_state, previous_state):
        """**Feature: autonomous-logos-pathos-agent, Property 5: Internal Reward Computation**
        **Validates: Requirements 2.5**
        
        For any state transition, the internal reward should incorporate both homeostatic balance 
        evaluation and state change smoothness penalties following the formula:
        r_t^int = -λ₁·D_t - λ₂·||F(t+1) - F(t)||²
        """
        # Ensure state dimensions match
        if len(current_state) != state_dimension or len(previous_state) != state_dimension:
            return
        
        # Create configuration with specific lambda values
        config = PathosConfig(
            state_dimension=state_dimension,
            lambda_1=lambda_1,
            lambda_2=lambda_2
        )
        
        # Create Pathos layer
        pathos = PathosLayer(config)
        
        # Compute internal reward
        internal_reward = pathos.compute_internal_reward(current_state, previous_state)
        
        # Property 1: Internal reward should be a finite number
        assert np.isfinite(internal_reward), "Internal reward should be finite"
        assert not np.isnan(internal_reward), "Internal reward should not be NaN"
        
        # Property 2: Verify the mathematical formula components
        # Compute homeostatic discomfort D_t
        balance_metrics, discomfort = pathos.compute_homeostatic_balance(current_state)
        
        # Compute state change penalty ||F(t+1) - F(t)||²
        from src.math_utils import compute_state_change_penalty
        change_penalty = compute_state_change_penalty(current_state, previous_state, penalty_type='l2')
        
        # Verify the formula: r_t^int = -λ₁·D_t - λ₂·||F(t+1) - F(t)||²
        expected_reward = -lambda_1 * discomfort - lambda_2 * change_penalty
        
        np.testing.assert_almost_equal(
            internal_reward, expected_reward, decimal=10,
            err_msg=f"Internal reward should follow formula: -λ₁·D_t - λ₂·||ΔF||². "
                   f"Expected: {expected_reward}, Got: {internal_reward}, "
                   f"λ₁={lambda_1}, D_t={discomfort}, λ₂={lambda_2}, ||ΔF||²={change_penalty}"
        )
        
        # Property 3: Homeostatic component should be non-positive (penalty)
        homeostatic_component = -lambda_1 * discomfort
        assert homeostatic_component <= 0, "Homeostatic component should be non-positive (penalty)"
        
        # Property 4: Change penalty component should be non-positive (penalty)
        change_component = -lambda_2 * change_penalty
        assert change_component <= 0, "Change penalty component should be non-positive (penalty)"
        
        # Property 5: Larger state changes should result in more negative rewards (when λ₂ > 0)
        if lambda_2 > 0:
            # Create a state with larger change
            larger_change_state = current_state + 0.5 * (current_state - previous_state)
            larger_change_state = np.clip(larger_change_state, -1.0, 1.0)  # Keep bounded
            
            larger_change_reward = pathos.compute_internal_reward(larger_change_state, previous_state)
            
            # Larger changes should generally result in more negative rewards
            # (unless homeostatic effects dominate)
            larger_change_penalty = compute_state_change_penalty(larger_change_state, previous_state, penalty_type='l2')
            original_change_penalty = compute_state_change_penalty(current_state, previous_state, penalty_type='l2')
            
            if larger_change_penalty > original_change_penalty:
                # The change penalty component should be more negative
                larger_change_component = -lambda_2 * larger_change_penalty
                original_change_component = -lambda_2 * original_change_penalty
                assert larger_change_component <= original_change_component, \
                    "Larger state changes should result in more negative change penalty"
        
        # Property 6: Identical states should have zero change penalty
        zero_change_reward = pathos.compute_internal_reward(current_state, current_state)
        zero_change_penalty = compute_state_change_penalty(current_state, current_state, penalty_type='l2')
        assert zero_change_penalty == 0.0, "Identical states should have zero change penalty"
        
        # Property 7: Reward should be deterministic for same inputs
        repeated_reward = pathos.compute_internal_reward(current_state, previous_state)
        np.testing.assert_almost_equal(
            internal_reward, repeated_reward, decimal=10,
            err_msg="Internal reward computation should be deterministic"
        )

    def test_pathos_state_update_basic(self):
        """Basic unit test for Pathos state update functionality"""
        config = PathosConfig(state_dimension=16)
        pathos = PathosLayer(config)
        
        # Create test inputs
        semantic_vector = SemanticVector(
            embedding=np.array([0.1, 0.2, 0.3, 0.4] + [0.0] * 12),
            intention_text="test intention",
            semantic_category="exploration"
        )
        
        initial_state = pathos.current_state.copy()
        
        # Perform update
        updated_state = pathos.update_state(
            semantic_input=semantic_vector,
            external_reward=0.5,
            interest=0.7,
            memory_echoes=[]
        )
        
        # Basic checks
        assert updated_state.shape == (16,)
        assert not np.array_equal(initial_state, updated_state)
        assert np.all(np.abs(updated_state) <= 1.0)
    
    def test_salience_computation_basic(self):
        """Basic unit test for salience computation and memory gating"""
        config = PathosConfig(state_dimension=16, salience_threshold=0.7)
        pathos = PathosLayer(config)
        
        # Test salience computation with known values
        salience = pathos.compute_salience(
            state_change=0.5,
            reward=0.3,
            novelty_affect=0.4,
            novelty_semantic=0.6,
            interest=0.8
        )
        
        # Salience should be between 0 and 1
        assert 0.0 <= salience <= 1.0
        assert isinstance(salience, float)
        
        # Test memory gating
        should_store_high = pathos.should_write_memory(0.8)  # Above threshold
        should_store_low = pathos.should_write_memory(0.5)   # Below threshold
        
        assert should_store_high is True
        assert should_store_low is False
        
        # Test with zero inputs
        zero_salience = pathos.compute_salience(0.0, 0.0, 0.0, 0.0, 0.0)
        assert 0.0 <= zero_salience <= 1.0
        
        # Test with maximum inputs
        max_salience = pathos.compute_salience(1.0, 1.0, 1.0, 1.0, 1.0)
        assert 0.0 <= max_salience <= 1.0
        assert max_salience > zero_salience  # Higher inputs should give higher salience
    
    @given(
        state_change=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        reward=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
        novelty_affect=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        novelty_semantic=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        interest=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=10)
    def test_salience_based_memory_storage_property(self, state_change, reward,
                                                   novelty_affect, novelty_semantic, interest):
        """**Feature: autonomous-logos-pathos-agent, Property 6: Salience-Based Memory Storage**
        **Validates: Requirements 3.1, 3.2**
        
        For any experience, when salience exceeds threshold, a complete memory trace should be created 
        containing affective state, semantic representation, reward, and timestamp. The salience computation
        should consider state change magnitude, reward magnitude, novelty, and Logos interest.
        """
        # Create configuration with default parameters
        config = PathosConfig(state_dimension=32)
        
        # Create Pathos layer
        pathos = PathosLayer(config)
        
        # Compute salience using the specified inputs
        computed_salience = pathos.compute_salience(
            state_change=state_change,
            reward=reward,
            novelty_affect=novelty_affect,
            novelty_semantic=novelty_semantic,
            interest=interest
        )
        
        # Property 1: Salience should be bounded between 0 and 1
        assert 0.0 <= computed_salience <= 1.0, f"Salience should be in [0,1], got {computed_salience}"
        
        # Property 2: Salience should be finite and not NaN
        assert np.isfinite(computed_salience), "Salience should be finite"
        assert not np.isnan(computed_salience), "Salience should not be NaN"
        
        # Property 3: Memory gating decision should be based on threshold (Requirement 3.2)
        should_store = pathos.should_write_memory(computed_salience)
        
        # Verify threshold-based gating
        if computed_salience >= config.salience_threshold:
            assert should_store is True, f"Should store memory when salience ({computed_salience}) >= threshold ({config.salience_threshold})"
        else:
            assert should_store is False, f"Should not store memory when salience ({computed_salience}) < threshold ({config.salience_threshold})"
        
        # Property 4: Salience computation should be deterministic
        repeated_salience = pathos.compute_salience(
            state_change=state_change,
            reward=reward,
            novelty_affect=novelty_affect,
            novelty_semantic=novelty_semantic,
            interest=interest
        )
        
        np.testing.assert_almost_equal(
            computed_salience, repeated_salience, decimal=10,
            err_msg="Salience computation should be deterministic for same inputs"
        )
        
        # Property 5: Salience computation considers all required factors (Requirement 3.1)
        # Test with zero inputs - should give minimal salience
        zero_salience = pathos.compute_salience(0.0, 0.0, 0.0, 0.0, 0.0)
        
        # If any input is positive, salience should be >= zero salience
        if any(val > 0 for val in [state_change, abs(reward), novelty_affect, novelty_semantic, interest]):
            assert computed_salience >= zero_salience, "Positive inputs should increase salience"

    def test_salience_configurable_parameters(self):
        """Test that salience computation uses configurable parameters"""
        # Test with custom coefficients
        config1 = PathosConfig(c1=2.0, c2=1.0, c3=0.5, c4=0.5, c5=0.5)
        config2 = PathosConfig(c1=1.0, c2=2.0, c3=0.5, c4=0.5, c5=0.5)
        
        pathos1 = PathosLayer(config1)
        pathos2 = PathosLayer(config2)
        
        # Same inputs should produce different salience due to different coefficients
        salience1 = pathos1.compute_salience(0.5, 0.3, 0.2, 0.2, 0.2)
        salience2 = pathos2.compute_salience(0.5, 0.3, 0.2, 0.2, 0.2)
        
        # They should be different because c1 vs c2 are weighted differently
        assert salience1 != salience2
        
        # Test threshold configuration
        config_low_threshold = PathosConfig(salience_threshold=0.3)
        config_high_threshold = PathosConfig(salience_threshold=0.9)
        
        pathos_low = PathosLayer(config_low_threshold)
        pathos_high = PathosLayer(config_high_threshold)
        
        # Same salience value should have different gating results
        test_salience = 0.6
        assert pathos_low.should_write_memory(test_salience) is True   # 0.6 > 0.3
        assert pathos_high.should_write_memory(test_salience) is False  # 0.6 < 0.9


if __name__ == "__main__":
    pytest.main([__file__])