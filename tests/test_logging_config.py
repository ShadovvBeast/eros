"""
Tests for the logging and monitoring system.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
import tempfile
import os
import json
from datetime import datetime

from src.logging_config import InstrumentationCollector, AgentLogger, setup_logging
from src.models import SemanticVector


class TestInstrumentationCollector:
    """Test the InstrumentationCollector class"""
    
    def test_initialization(self):
        """Test collector initialization"""
        collector = InstrumentationCollector(enable_monitoring=True)
        assert collector.enable_monitoring is True
        assert len(collector.metrics) == 9  # All metric categories
        assert collector.pattern_window_size == 10
        
        # Test disabled monitoring
        disabled_collector = InstrumentationCollector(enable_monitoring=False)
        assert disabled_collector.enable_monitoring is False
    
    def test_pathos_state_recording(self):
        """Test Pathos state trajectory recording"""
        collector = InstrumentationCollector(enable_monitoring=True)
        
        state_vector = np.random.random(64)
        homeostatic_balance = {'stability': 0.8, 'complexity': 0.6}
        
        collector.record_pathos_state(1, state_vector, 0.5, homeostatic_balance)
        
        assert len(collector.metrics['pathos_trajectories']) == 1
        record = collector.metrics['pathos_trajectories'][0]
        assert record['cycle'] == 1
        assert record['internal_reward'] == 0.5
        assert record['homeostatic_balance'] == homeostatic_balance
        assert 'timestamp' in record
        assert 'state_norm' in record
    
    def test_memory_event_recording(self):
        """Test memory event recording"""
        collector = InstrumentationCollector(enable_monitoring=True)
        
        collector.record_memory_event('store', 0.8, 15)
        
        assert len(collector.metrics['memory_events']) == 1
        record = collector.metrics['memory_events'][0]
        assert record['event_type'] == 'store'
        assert record['salience'] == 0.8
        assert record['total_traces'] == 15
    
    def test_preference_change_recording(self):
        """Test preference change recording and drift analysis"""
        collector = InstrumentationCollector(enable_monitoring=True)
        
        # Record several preference changes to trigger drift analysis
        collector.record_preference_change('exploration', 0.5, 0.6)
        collector.record_preference_change('exploration', 0.6, 0.7)
        collector.record_preference_change('exploration', 0.7, 0.8)
        collector.record_preference_change('exploration', 0.8, 0.9)
        collector.record_preference_change('exploration', 0.9, 0.95)
        
        assert len(collector.metrics['preference_changes']) == 5
        assert len(collector.metrics['preference_drift_analysis']) >= 1
        
        # Check preference history tracking
        assert 'exploration' in collector.preference_history
        assert len(collector.preference_history['exploration']) == 5
    
    def test_behavior_pattern_recording(self):
        """Test behavior pattern analysis"""
        collector = InstrumentationCollector(enable_monitoring=True)
        
        # Record enough patterns to trigger analysis
        for i in range(12):
            intention = f"explore_{i % 3}"  # Create repetition
            tool = "tool_a" if i % 2 == 0 else "tool_b"
            reward = 0.5 + 0.1 * i
            
            collector.record_behavior_pattern(i, intention, tool, reward, 0.8)
        
        assert len(collector.metrics['behavior_patterns']) >= 1
        
        # Check that pattern analysis is working
        pattern = collector.metrics['behavior_patterns'][-1]
        assert 'intention_repetition_rate' in pattern
        assert 'tool_preference_entropy' in pattern
        assert 'reward_trend' in pattern
    
    def test_tool_usage_recording(self):
        """Test tool usage recording"""
        collector = InstrumentationCollector(enable_monitoring=True)
        
        collector.record_tool_usage('search_tool', True, 1.5)
        collector.record_tool_usage('analyze_tool', False, 0.8)
        
        assert len(collector.metrics['tool_usage']) == 2
        
        success_record = collector.metrics['tool_usage'][0]
        assert success_record['tool_name'] == 'search_tool'
        assert success_record['success'] is True
        assert success_record['execution_time'] == 1.5
    
    def test_attractor_pattern_recording(self):
        """Test attractor pattern recording"""
        collector = InstrumentationCollector(enable_monitoring=True)
        
        collector.record_attractor_pattern(5, 3, 0.85, True)
        
        assert len(collector.metrics['attractor_patterns']) == 1
        record = collector.metrics['attractor_patterns'][0]
        assert record['cycle'] == 5
        assert record['attractor_count'] == 3
        assert record['strongest_attractor_strength'] == 0.85
        assert record['pattern_recognized'] is True
    
    def test_metrics_summary(self):
        """Test metrics summary generation"""
        collector = InstrumentationCollector(enable_monitoring=True)
        
        # Add some data
        collector.record_pathos_state(1, np.random.random(64), 0.5, {})
        collector.record_memory_event('store', 0.8, 10)
        collector.record_tool_usage('test_tool', True, 1.0)
        
        summary = collector.get_metrics_summary()
        
        assert 'runtime_seconds' in summary
        assert summary['total_memory_events'] == 1
        assert summary['total_tool_calls'] == 1
        assert summary['preference_changes'] == 0
    
    def test_preference_drift_summary(self):
        """Test preference drift summary"""
        collector = InstrumentationCollector(enable_monitoring=True)
        
        # Create significant drift
        for i in range(6):
            collector.record_preference_change('test_category', 0.5, 0.5 + i * 0.1)
        
        summary = collector.get_preference_drift_summary()
        
        assert 'total_categories_tracked' in summary
        assert 'categories_with_significant_drift' in summary
        assert 'average_volatility' in summary
    
    def test_behavior_pattern_summary(self):
        """Test behavior pattern summary"""
        collector = InstrumentationCollector(enable_monitoring=True)
        
        # Generate behavior patterns
        for i in range(15):
            collector.record_behavior_pattern(i, f"intention_{i%2}", f"tool_{i%3}", 0.5, 0.8)
        
        summary = collector.get_behavior_pattern_summary()
        
        assert 'average_intention_repetition' in summary
        assert 'average_tool_entropy' in summary
        assert 'behavior_stability' in summary
        assert 'exploration_level' in summary
    
    def test_export_metrics(self):
        """Test metrics export functionality"""
        collector = InstrumentationCollector(enable_monitoring=True)
        
        # Add some test data
        collector.record_pathos_state(1, np.random.random(64), 0.5, {'test': 0.8})
        collector.record_preference_change('test', 0.5, 0.7)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            collector.export_metrics(temp_path)
            
            # Verify file was created and contains expected data
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert 'summary' in data
            assert 'preference_drift_summary' in data
            assert 'behavior_pattern_summary' in data
            assert 'detailed_metrics' in data
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_disabled_monitoring(self):
        """Test that disabled monitoring doesn't record data"""
        collector = InstrumentationCollector(enable_monitoring=False)
        
        collector.record_pathos_state(1, np.random.random(64), 0.5, {})
        collector.record_memory_event('store', 0.8, 10)
        collector.record_preference_change('test', 0.5, 0.7)
        
        # All metrics should be empty
        for metric_list in collector.metrics.values():
            assert len(metric_list) == 0
        
        # Summaries should be empty
        assert collector.get_metrics_summary() == {}
        assert collector.get_preference_drift_summary() == {}
        assert collector.get_behavior_pattern_summary() == {}


class TestAgentLogger:
    """Test the AgentLogger class"""
    
    def test_logger_initialization(self):
        """Test logger initialization"""
        logger = AgentLogger(log_level='DEBUG')
        assert logger.logger.level == 10  # DEBUG level
        
        # Test with file logging
        import tempfile
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, 'test.log')
        
        try:
            file_logger = AgentLogger(log_level='INFO', log_file=temp_path)
            file_logger.info("Test message")
            
            # Verify log file was created
            assert os.path.exists(temp_path)
            
        finally:
            # Clean up handlers to release file
            for handler in file_logger.logger.handlers[:]:
                handler.close()
                file_logger.logger.removeHandler(handler)
            
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            os.rmdir(temp_dir)
    
    def test_structured_logging(self):
        """Test structured logging with kwargs"""
        logger = AgentLogger(log_level='DEBUG')
        
        # These should not raise exceptions
        logger.info("Test message", key1="value1", key2=42)
        logger.debug("Debug message", data={"nested": "value"})
        logger.warning("Warning message", error_code=404)
        logger.error("Error message", exception="TestException")


# Property-based tests
class TestMonitoringInstrumentationProperties:
    """
    **Feature: autonomous-logos-pathos-agent, Property 19: Monitoring Instrumentation**
    
    Property-based tests for monitoring instrumentation requirements.
    """
    
    @given(
        cycles=st.integers(min_value=1, max_value=100),
        state_dim=st.integers(min_value=32, max_value=256),
        num_categories=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100)
    def test_monitoring_instrumentation_property(self, cycles, state_dim, num_categories):
        """
        **Property 19: Monitoring Instrumentation**
        **Validates: Requirements 8.5**
        
        For any autonomous operation period, instrumentation data should be available 
        for monitoring preference drift, behavior patterns, and system health.
        """
        collector = InstrumentationCollector(enable_monitoring=True)
        
        # Simulate autonomous operation cycles
        categories = [f"category_{i}" for i in range(num_categories)]
        
        for cycle in range(cycles):
            # Simulate Pathos state recording
            state_vector = np.random.random(state_dim)
            internal_reward = np.random.uniform(-1.0, 1.0)
            homeostatic_balance = {
                'stability': np.random.uniform(0.0, 1.0),
                'complexity': np.random.uniform(0.0, 1.0)
            }
            collector.record_pathos_state(cycle, state_vector, internal_reward, homeostatic_balance)
            
            # Simulate preference changes
            if cycle > 0:  # Need previous state for preference changes
                category = np.random.choice(categories)
                old_weight = np.random.uniform(0.0, 1.0)
                new_weight = np.random.uniform(0.0, 1.0)
                collector.record_preference_change(category, old_weight, new_weight)
            
            # Simulate behavior patterns
            intention = f"intention_{np.random.randint(0, 3)}"
            tool = f"tool_{np.random.randint(0, 2)}" if np.random.random() > 0.3 else None
            reward = np.random.uniform(-0.5, 1.0)
            state_similarity = np.random.uniform(0.0, 1.0)
            collector.record_behavior_pattern(cycle, intention, tool, reward, state_similarity)
            
            # Simulate memory events
            if np.random.random() > 0.7:  # 30% chance of memory storage
                salience = np.random.uniform(0.0, 1.0)
                trace_count = cycle + 1
                collector.record_memory_event('store', salience, trace_count)
            
            # Simulate tool usage
            if tool and np.random.random() > 0.5:
                success = np.random.random() > 0.3
                execution_time = np.random.uniform(0.1, 5.0)
                collector.record_tool_usage(tool, success, execution_time)
        
        # Verify instrumentation data is available
        
        # 1. Pathos trajectory data should be available
        assert len(collector.metrics['pathos_trajectories']) == cycles
        for trajectory in collector.metrics['pathos_trajectories']:
            assert 'cycle' in trajectory
            assert 'timestamp' in trajectory
            assert 'state_norm' in trajectory
            assert 'internal_reward' in trajectory
            assert 'homeostatic_balance' in trajectory
        
        # 2. Preference drift monitoring should be available
        if collector.metrics['preference_changes']:
            # Should have preference change records
            assert len(collector.metrics['preference_changes']) > 0
            for change in collector.metrics['preference_changes']:
                assert 'semantic_category' in change
                assert 'old_weight' in change
                assert 'new_weight' in change
                assert 'change' in change
                assert 'timestamp' in change
        
        # 3. Behavior pattern analysis should be available
        if len(collector.metrics['behavior_patterns']) > 0:
            for pattern in collector.metrics['behavior_patterns']:
                assert 'cycle' in pattern
                assert 'intention_repetition_rate' in pattern
                assert 'tool_preference_entropy' in pattern
                assert 'reward_trend' in pattern
                assert 'state_similarity' in pattern
        
        # 4. System health metrics should be available
        summary = collector.get_metrics_summary()
        assert 'runtime_seconds' in summary
        assert 'total_cycles' in summary
        assert summary['total_cycles'] == 0  # No cycle timings recorded in this test
        
        # 5. Preference drift analysis should be available if there's enough data
        drift_summary = collector.get_preference_drift_summary()
        if collector.preference_history:
            assert 'total_categories_tracked' in drift_summary
            assert drift_summary['total_categories_tracked'] > 0
        
        # 6. Behavior pattern summary should be available if there's enough data
        behavior_summary = collector.get_behavior_pattern_summary()
        if collector.metrics['behavior_patterns']:
            assert 'behavior_stability' in behavior_summary
            assert 'exploration_level' in behavior_summary
    
    @given(
        preference_changes=st.lists(
            st.tuples(
                st.text(min_size=1, max_size=20),  # category
                st.floats(min_value=0.0, max_value=1.0),  # old_weight
                st.floats(min_value=0.0, max_value=1.0)   # new_weight
            ),
            min_size=1,
            max_size=50
        )
    )
    @settings(max_examples=50)
    def test_preference_drift_monitoring_property(self, preference_changes):
        """
        Property: Preference drift monitoring should detect and analyze changes
        in preference weights over time.
        """
        collector = InstrumentationCollector(enable_monitoring=True)
        
        # Record all preference changes
        for category, old_weight, new_weight in preference_changes:
            collector.record_preference_change(category, old_weight, new_weight)
        
        # Verify preference drift monitoring
        assert len(collector.metrics['preference_changes']) == len(preference_changes)
        
        # Check that preference history is maintained
        categories_seen = set(category for category, _, _ in preference_changes)
        for category in categories_seen:
            assert category in collector.preference_history
            assert len(collector.preference_history[category]) > 0
        
        # If enough data, drift analysis should be available
        if any(len(collector.preference_history[cat]) >= 5 for cat in collector.preference_history):
            assert len(collector.metrics['preference_drift_analysis']) > 0
    
    @given(
        behavior_data=st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=1000),  # cycle
                st.text(min_size=1, max_size=50),          # intention
                st.one_of(st.none(), st.text(min_size=1, max_size=20)),  # tool
                st.floats(min_value=-1.0, max_value=2.0),  # reward
                st.floats(min_value=0.0, max_value=1.0)    # state_similarity
            ),
            min_size=10,
            max_size=30
        )
    )
    @settings(max_examples=50)
    def test_behavior_pattern_analysis_property(self, behavior_data):
        """
        Property: Behavior pattern analysis should detect patterns in agent
        intentions, tool usage, and reward trends.
        """
        collector = InstrumentationCollector(enable_monitoring=True)
        
        # Record behavior patterns
        for cycle, intention, tool, reward, state_similarity in behavior_data:
            collector.record_behavior_pattern(cycle, intention, tool, reward, state_similarity)
        
        # Should have behavior pattern analyses (after window fills)
        if len(behavior_data) >= collector.pattern_window_size:
            assert len(collector.metrics['behavior_patterns']) > 0
            
            # Each pattern analysis should have required metrics
            for pattern in collector.metrics['behavior_patterns']:
                assert 0.0 <= pattern['intention_repetition_rate'] <= 1.0
                assert pattern['tool_preference_entropy'] >= 0.0
                assert isinstance(pattern['reward_trend'], (int, float))
                assert 0.0 <= pattern['state_similarity'] <= 1.0