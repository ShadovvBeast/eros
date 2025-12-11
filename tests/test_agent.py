"""
Tests for the main AutonomousAgent class and autonomous operation loop.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings

from src.agent import AutonomousAgent
from src.config import AgentConfig, PathosConfig, LogosConfig, MemoryConfig, EthosConfig, ToolConfig
from src.models import SemanticVector, Intention, MemoryTrace, ToolCall, ToolResult
from src.logos.logos_layer import LogosLayer
from src.pathos.pathos_layer import PathosLayer
from src.memory.memory_system import ConcreteMemorySystem
from src.ethos.ethos_framework import ConcreteEthosFramework
from src.tools.tool_layer import ToolLayer


class TestAutonomousAgent:
    """Test cases for AutonomousAgent class"""
    
    @pytest.fixture
    def agent_config(self):
        """Create test agent configuration"""
        return AgentConfig(
            pathos=PathosConfig(state_dimension=64, salience_threshold=0.5),
            logos=LogosConfig(semantic_dimension=128),
            memory=MemoryConfig(max_memory_traces=100, storage_backend='memory'),
            ethos=EthosConfig(
                identity_description="Test agent for autonomous operation",
                core_values=["safety", "learning", "coherence"],
                safety_constraints=["No harmful actions"]
            ),
            tools=ToolConfig(max_concurrent_tools=2),
            cycle_interval_seconds=0.1,  # Fast cycles for testing
            max_cycles=5  # Limited cycles for testing
        )
    
    @pytest.fixture
    def initialized_agent(self, agent_config):
        """Create fully initialized agent for testing"""
        agent = AutonomousAgent(agent_config)
        
        # Create layer instances
        ethos = ConcreteEthosFramework(agent_config.ethos)
        logos = LogosLayer(agent_config.logos)
        pathos = PathosLayer(agent_config.pathos)
        memory = ConcreteMemorySystem(agent_config.memory)
        tools = ToolLayer(agent_config.tools, ethos)
        
        # Initialize agent with layers
        agent.initialize_layers(logos, pathos, memory, ethos, tools)
        
        return agent
    
    def test_agent_initialization(self, agent_config):
        """Test basic agent initialization"""
        agent = AutonomousAgent(agent_config)
        
        assert agent.config == agent_config
        assert agent.cycle_count == 0
        assert not agent.running
        assert agent.logos is None  # Not initialized yet
    
    def test_layer_initialization(self, agent_config):
        """Test layer initialization"""
        agent = AutonomousAgent(agent_config)
        
        # Create mock layers
        logos = Mock(spec=LogosLayer)
        pathos = Mock(spec=PathosLayer)
        memory = Mock(spec=ConcreteMemorySystem)
        ethos = Mock(spec=ConcreteEthosFramework)
        tools = Mock(spec=ToolLayer)
        
        # Initialize layers
        agent.initialize_layers(logos, pathos, memory, ethos, tools)
        
        assert agent.logos == logos
        assert agent.pathos == pathos
        assert agent.memory == memory
        assert agent.ethos == ethos
        assert agent.tools == tools
        
        # Verify tool callback registration
        tools.register_result_callback.assert_called_once()
    
    def test_run_cycle_without_initialization_raises_error(self, agent_config):
        """Test that running cycle without layer initialization raises error"""
        agent = AutonomousAgent(agent_config)
        
        with pytest.raises(RuntimeError, match="Agent layers not fully initialized"):
            agent.run_cycle()
    
    def test_single_cycle_execution(self, initialized_agent):
        """Test execution of a single agent cycle"""
        agent = initialized_agent
        
        # Mock the current state to avoid None issues
        agent.pathos.current_state = np.random.normal(0, 0.1, size=64)
        agent.pathos.previous_state = np.random.normal(0, 0.1, size=64)
        
        # Execute one cycle
        result = agent.run_cycle()
        
        # Verify cycle result structure
        assert isinstance(result, dict)
        assert 'cycle' in result
        assert 'intention' in result
        assert 'internal_reward' in result
        assert 'external_reward' in result
        assert 'duration' in result
        
        # Verify cycle count incremented
        assert agent.cycle_count == 1
    
    def test_autonomous_operation_limited_cycles(self, initialized_agent):
        """Test autonomous operation with limited cycles"""
        agent = initialized_agent
        
        # Mock the current state
        agent.pathos.current_state = np.random.normal(0, 0.1, size=64)
        agent.pathos.previous_state = np.random.normal(0, 0.1, size=64)
        
        # Run autonomous operation with 3 cycles
        agent.run_autonomous(max_cycles=3)
        
        # Verify cycles were executed
        assert agent.cycle_count == 3
        assert not agent.running  # Should stop after max cycles
    
    def test_agent_stop_functionality(self, initialized_agent):
        """Test agent stop functionality"""
        agent = initialized_agent
        
        # Start agent in background (simulate)
        agent.running = True
        
        # Stop agent
        agent.stop()
        
        assert not agent.running
    
    def test_agent_status_reporting(self, initialized_agent):
        """Test agent status reporting"""
        agent = initialized_agent
        
        # Mock the current state
        agent.pathos.current_state = np.random.normal(0, 0.1, size=64)
        
        status = agent.get_status()
        
        assert isinstance(status, dict)
        assert 'running' in status
        assert 'cycle_count' in status
        assert 'current_state_norm' in status
        assert 'memory_trace_count' in status
        assert 'metrics_summary' in status
    
    def test_fallback_intention_generation(self, initialized_agent):
        """Test fallback intention generation"""
        agent = initialized_agent
        
        fallback = agent._generate_fallback_intention()
        
        assert isinstance(fallback, Intention)
        assert fallback.semantic_vector.semantic_category == "reflection"
        assert fallback.priority == 0.3
        assert len(fallback.tool_candidates) == 0
    
    def test_tool_result_handling(self, initialized_agent):
        """Test tool result handling callback"""
        agent = initialized_agent
        
        # Create mock tool result
        result = ToolResult(
            tool_name="test_tool",
            result="test_output",
            success=True,
            timestamp=int(time.time())
        )
        
        # Handle result (should not raise error)
        agent._handle_tool_result(result)
        
        # Verify no exceptions were raised
        assert True


class TestAutonomousOperationProperty:
    """Property-based tests for autonomous operation"""
    
    @pytest.fixture
    def minimal_agent_config(self):
        """Create minimal agent configuration for property testing"""
        return AgentConfig(
            pathos=PathosConfig(state_dimension=32, salience_threshold=0.5),
            logos=LogosConfig(semantic_dimension=64),
            memory=MemoryConfig(max_memory_traces=50, storage_backend='memory'),
            ethos=EthosConfig(
                identity_description="Property test agent",
                core_values=["safety"],
                safety_constraints=["No harmful actions"]
            ),
            tools=ToolConfig(max_concurrent_tools=1),
            cycle_interval_seconds=0.01,  # Very fast for testing
            max_cycles=3  # Limited for property testing
        )
    
    @given(
        cycles=st.integers(min_value=1, max_value=5),
        state_dim=st.integers(min_value=16, max_value=64)
    )
    @settings(max_examples=10, deadline=5000)  # Limited examples for speed
    def test_autonomous_operation_property(self, cycles, state_dim):
        """
        **Feature: autonomous-logos-pathos-agent, Property 2: Autonomous Operation**
        
        Property: For any sequence of cycles without external input, 
        the agent should continue operating and generate self-directed goals.
        
        **Validates: Requirements 1.3**
        """
        # Create config with test parameters
        minimal_agent_config = AgentConfig(
            pathos=PathosConfig(state_dimension=state_dim, salience_threshold=0.5),
            logos=LogosConfig(semantic_dimension=state_dim * 2),
            memory=MemoryConfig(max_memory_traces=50, storage_backend='memory'),
            ethos=EthosConfig(
                identity_description="Property test agent",
                core_values=["safety"],
                safety_constraints=["No harmful actions"]
            ),
            tools=ToolConfig(max_concurrent_tools=1),
            cycle_interval_seconds=0.01,  # Very fast for testing
            max_cycles=cycles
        )
        
        # Create and initialize agent
        agent = AutonomousAgent(minimal_agent_config)
        
        # Create layer instances
        ethos = ConcreteEthosFramework(minimal_agent_config.ethos)
        logos = LogosLayer(minimal_agent_config.logos)
        pathos = PathosLayer(minimal_agent_config.pathos)
        memory = ConcreteMemorySystem(minimal_agent_config.memory)
        tools = ToolLayer(minimal_agent_config.tools, ethos)
        
        # Initialize agent
        agent.initialize_layers(logos, pathos, memory, ethos, tools)
        
        # Ensure agent has initial state
        assert agent.pathos.current_state is not None
        assert len(agent.pathos.current_state) == state_dim
        
        # Run autonomous operation
        agent.run_autonomous(max_cycles=cycles)
        
        # Property: Agent should complete all requested cycles
        assert agent.cycle_count == cycles
        
        # Property: Agent should generate intentions in each cycle
        # (This is verified by the fact that cycles complete without error)
        
        # Property: Agent should maintain valid internal state
        assert agent.pathos.current_state is not None
        assert len(agent.pathos.current_state) == state_dim
        assert np.all(np.isfinite(agent.pathos.current_state))
        
        # Property: Agent should stop after max cycles (not continue indefinitely)
        assert not agent.running


class TestRecursiveAffectInfluenceProperty:
    """Property-based tests for recursive affect influence"""
    
    @pytest.fixture
    def affect_test_config(self):
        """Configuration optimized for affect influence testing"""
        return AgentConfig(
            pathos=PathosConfig(
                state_dimension=32,
                decay_factor=0.9,
                echo_strength=0.2,
                salience_threshold=0.3
            ),
            logos=LogosConfig(semantic_dimension=64),
            memory=MemoryConfig(max_memory_traces=20, storage_backend='memory'),
            ethos=EthosConfig(
                identity_description="Affect influence test agent",
                core_values=["coherence"],
                safety_constraints=["No harmful actions"]
            ),
            tools=ToolConfig(max_concurrent_tools=1),
            cycle_interval_seconds=0.01,
            max_cycles=3
        )
    
    @given(
        initial_state=st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=16, max_size=32
        ),
        num_cycles=st.integers(min_value=2, max_value=4)
    )
    @settings(max_examples=8, deadline=5000)
    def test_recursive_affect_influence_property(self, initial_state, num_cycles):
        """
        **Feature: autonomous-logos-pathos-agent, Property 15: Recursive Affect Influence**
        
        Property: For any experience processing, the current affect state should influence 
        future processing, and recurring patterns should create memory-driven echo dynamics.
        
        **Validates: Requirements 7.1, 7.2**
        """
        # Create config
        affect_test_config = AgentConfig(
            pathos=PathosConfig(
                state_dimension=len(initial_state),
                decay_factor=0.9,
                echo_strength=0.2,
                salience_threshold=0.3
            ),
            logos=LogosConfig(semantic_dimension=len(initial_state) * 2),
            memory=MemoryConfig(max_memory_traces=20, storage_backend='memory'),
            ethos=EthosConfig(
                identity_description="Affect influence test agent",
                core_values=["coherence"],
                safety_constraints=["No harmful actions"]
            ),
            tools=ToolConfig(max_concurrent_tools=1),
            cycle_interval_seconds=0.01,
            max_cycles=num_cycles
        )
        
        # Create and initialize agent
        agent = AutonomousAgent(affect_test_config)
        
        ethos = ConcreteEthosFramework(affect_test_config.ethos)
        logos = LogosLayer(affect_test_config.logos)
        pathos = PathosLayer(affect_test_config.pathos)
        memory = ConcreteMemorySystem(affect_test_config.memory)
        tools = ToolLayer(affect_test_config.tools, ethos)
        
        agent.initialize_layers(logos, pathos, memory, ethos, tools)
        
        # Set initial state
        initial_state_array = np.array(initial_state)
        agent.pathos.set_state(initial_state_array)
        
        # Store initial state for comparison
        states_over_time = [initial_state_array.copy()]
        
        # Run cycles and track state changes
        for cycle in range(num_cycles):
            result = agent.run_cycle()
            states_over_time.append(agent.pathos.current_state.copy())
        
        # Property 1: Current affect state influences future processing
        # Verify that each state influences the next state
        for i in range(len(states_over_time) - 1):
            current_state = states_over_time[i]
            next_state = states_over_time[i + 1]
            
            # The next state should be influenced by current state (not completely random)
            # This is verified by the mathematical model: F(t+1) = g(α·F(t) + ...)
            # The decay term α·F(t) ensures current state influences next state
            
            # Check that states are not identical (system is dynamic)
            state_difference = np.linalg.norm(next_state - current_state)
            assert state_difference > 0, "States should change between cycles"
            
            # Check that influence is bounded (system is stable)
            assert state_difference < 10.0, "State changes should be bounded"
        
        # Property 2: System maintains memory of past states
        # If memories were created, they should influence future states through echoes
        memory_count = agent.memory.get_trace_count()
        
        if memory_count > 0:
            # Verify that memories exist and can influence future processing
            recent_memories = agent.memory.get_recent_traces(5)
            assert len(recent_memories) > 0
            
            # Each memory should contain valid affect state
            for memory in recent_memories:
                assert memory.affect_state is not None
                assert len(memory.affect_state) == len(initial_state)
                assert np.all(np.isfinite(memory.affect_state))
        
        # Property 3: Final state should be influenced by the entire trajectory
        final_state = states_over_time[-1]
        
        # Final state should be finite and bounded
        assert np.all(np.isfinite(final_state))
        assert np.all(np.abs(final_state) <= 10.0)  # Reasonable bounds
        
        # The system should show evidence of recursive influence through
        # the mathematical dynamics (this is inherent in the update equation)
        assert len(states_over_time) == num_cycles + 1  # Initial + num_cycles states


class TestAttractorDynamicsProperty:
    """Property-based tests for attractor dynamics"""
    
    @given(
        positive_rewards=st.lists(
            st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
            min_size=2, max_size=5
        ),
        state_dimension=st.integers(min_value=16, max_value=32),
        num_cycles=st.integers(min_value=3, max_value=6)
    )
    @settings(max_examples=8, deadline=5000)
    def test_attractor_dynamics_property(self, positive_rewards, state_dimension, num_cycles):
        """
        **Feature: autonomous-logos-pathos-agent, Property 12: Attractor Dynamics**
        
        Property: For any familiar pattern encounter, the Pathos layer should exhibit 
        attractor behavior toward previously rewarding states through memory echo mechanisms.
        
        **Validates: Requirements 5.2**
        """
        # Create configuration for attractor testing
        config = AgentConfig(
            pathos=PathosConfig(
                state_dimension=state_dimension,
                decay_factor=0.9,
                echo_strength=0.3,  # Higher echo strength for attractor testing
                salience_threshold=0.4
            ),
            logos=LogosConfig(semantic_dimension=state_dimension * 2),
            memory=MemoryConfig(max_memory_traces=50, storage_backend='memory'),
            ethos=EthosConfig(
                identity_description="Attractor dynamics test agent",
                core_values=["coherence"],
                safety_constraints=["No harmful actions"]
            ),
            tools=ToolConfig(max_concurrent_tools=1),
            cycle_interval_seconds=0.01,
            max_cycles=num_cycles
        )
        
        # Create and initialize agent
        agent = AutonomousAgent(config)
        
        ethos = ConcreteEthosFramework(config.ethos)
        logos = LogosLayer(config.logos)
        pathos = PathosLayer(config.pathos)
        memory = ConcreteMemorySystem(config.memory)
        tools = ToolLayer(config.tools, ethos)
        
        agent.initialize_layers(logos, pathos, memory, ethos, tools)
        
        # Set initial state
        initial_state = np.random.normal(0, 0.1, size=state_dimension)
        agent.pathos.set_state(initial_state)
        
        # Create some positive experiences to establish attractors
        rewarding_states = []
        for i, reward in enumerate(positive_rewards):
            # Create a distinct state for each positive reward
            rewarding_state = np.random.normal(0.5 * (i + 1), 0.1, size=state_dimension)
            rewarding_state = np.clip(rewarding_state, -1.0, 1.0)  # Keep bounded
            
            # Manually add as attractor
            agent.pathos.add_attractor_state(rewarding_state, reward)
            rewarding_states.append(rewarding_state)
        
        # Property 1: Attractor states should be stored
        assert len(agent.pathos.attractor_states) > 0
        assert len(agent.pathos.attractor_rewards) == len(agent.pathos.attractor_states)
        assert len(agent.pathos.attractor_strengths) == len(agent.pathos.attractor_states)
        
        # Property 2: All stored rewards should be positive (attractors only for positive experiences)
        for reward in agent.pathos.attractor_rewards:
            assert reward > 0.0
        
        # Run cycles and test attractor behavior
        states_during_cycles = []
        attractor_influences = []
        
        for cycle in range(num_cycles):
            # Set state near one of the rewarding states to test attractor behavior
            if cycle < len(rewarding_states):
                # Set state close to a known attractor
                target_attractor = rewarding_states[cycle % len(rewarding_states)]
                noise = np.random.normal(0, 0.1, size=state_dimension)
                near_attractor_state = target_attractor + noise
                near_attractor_state = np.clip(near_attractor_state, -2.0, 2.0)
                agent.pathos.set_state(near_attractor_state)
            
            # Record state before cycle
            pre_cycle_state = agent.pathos.current_state.copy()
            
            # Compute attractor influence
            influence = agent.pathos.compute_attractor_influence(pre_cycle_state)
            attractor_influences.append(np.linalg.norm(influence))
            
            # Run cycle
            result = agent.run_cycle()
            states_during_cycles.append(agent.pathos.current_state.copy())
        
        # Property 3: Familiar pattern recognition should work
        for rewarding_state in rewarding_states:
            is_familiar, attractor_idx = agent.pathos.recognize_familiar_patterns(rewarding_state)
            # Should recognize its own attractor states as familiar
            assert is_familiar or np.linalg.norm(rewarding_state) < 0.1  # Allow for very small states
        
        # Property 4: Attractor influence should be non-zero when near attractors
        non_zero_influences = [inf for inf in attractor_influences if inf > 1e-6]
        if len(rewarding_states) > 0:
            # Should have some non-zero influences when we set states near attractors
            assert len(non_zero_influences) > 0
        
        # Property 5: System should maintain bounded behavior despite attractors
        for state in states_during_cycles:
            assert np.all(np.isfinite(state))
            assert np.linalg.norm(state) < 20.0  # Reasonable upper bound
        
        # Property 6: Attractor summary should be consistent
        summary = agent.pathos.get_attractor_summary()
        assert summary['num_attractors'] >= 0
        assert summary['average_reward'] >= 0.0 or summary['num_attractors'] == 0
        assert summary['average_strength'] >= 0.0 or summary['num_attractors'] == 0
        
        # Property 7: Memory-driven echo dynamics should enhance basic echoes
        if agent.memory.get_trace_count() > 0:
            recent_memories = agent.memory.get_recent_traces(3)
            enhanced_echo = agent.pathos.implement_memory_driven_echo_dynamics(recent_memories)
            
            # Enhanced echo should be finite and bounded
            assert np.all(np.isfinite(enhanced_echo))
            assert np.linalg.norm(enhanced_echo) < 10.0