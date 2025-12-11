"""
Logging and instrumentation configuration for the Autonomous Logos-Pathos-Memory Agent.
"""

import logging
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import json
import os
import numpy as np


class AgentLogger:
    """Centralized logging system for the agent"""
    
    def __init__(self, log_level: str = 'INFO', log_file: Optional[str] = None):
        self.logger = logging.getLogger('autonomous_agent')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data"""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.info(message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data"""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.debug(message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data"""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.warning(message)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional structured data"""
        if kwargs:
            message = f"{message} | {json.dumps(kwargs)}"
        self.logger.error(message)


class InstrumentationCollector:
    """Collects instrumentation data for monitoring and analysis"""
    
    def __init__(self, enable_monitoring: bool = True):
        self.enable_monitoring = enable_monitoring
        self.metrics: Dict[str, Any] = {
            'pathos_trajectories': [],
            'memory_events': [],
            'preference_changes': [],
            'behavior_patterns': [],
            'tool_usage': [],
            'cycle_timings': [],
            'error_events': [],
            'attractor_patterns': [],
            'preference_drift_analysis': []
        }
        self.start_time = datetime.now()
        
        # Behavior pattern analysis state
        self.recent_intentions: List[str] = []
        self.recent_tools: List[str] = []
        self.recent_rewards: List[float] = []
        self.pattern_window_size = 10
        
        # Preference drift tracking
        self.preference_baseline: Dict[str, float] = {}
        self.preference_history: Dict[str, List[Tuple[str, float]]] = {}  # category -> [(timestamp, weight), ...]
    
    def record_pathos_state(self, cycle: int, state_vector: Any, internal_reward: float, 
                           homeostatic_balance: Dict[str, float]):
        """Record Pathos state trajectory"""
        if not self.enable_monitoring:
            return
            
        self.metrics['pathos_trajectories'].append({
            'cycle': cycle,
            'timestamp': datetime.now().isoformat(),
            'state_norm': float(np.linalg.norm(state_vector)) if hasattr(state_vector, 'shape') else 0.0,
            'internal_reward': internal_reward,
            'homeostatic_balance': homeostatic_balance
        })
    
    def record_memory_event(self, event_type: str, salience: float, trace_count: int):
        """Record memory storage/retrieval events"""
        if not self.enable_monitoring:
            return
            
        self.metrics['memory_events'].append({
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'salience': salience,
            'total_traces': trace_count
        })
    
    def record_preference_change(self, semantic_category: str, old_weight: float, new_weight: float):
        """Record preference weight changes"""
        if not self.enable_monitoring:
            return
        
        timestamp = datetime.now().isoformat()
        change = new_weight - old_weight
            
        self.metrics['preference_changes'].append({
            'semantic_category': semantic_category,
            'timestamp': timestamp,
            'old_weight': old_weight,
            'new_weight': new_weight,
            'change': change
        })
        
        # Track preference history for drift analysis
        if semantic_category not in self.preference_history:
            self.preference_history[semantic_category] = []
            self.preference_baseline[semantic_category] = old_weight
        
        self.preference_history[semantic_category].append((timestamp, new_weight))
        
        # Analyze preference drift if we have enough history
        if len(self.preference_history[semantic_category]) >= 5:
            self._analyze_preference_drift(semantic_category)
    
    def record_tool_usage(self, tool_name: str, success: bool, execution_time: float):
        """Record tool usage statistics"""
        if not self.enable_monitoring:
            return
            
        self.metrics['tool_usage'].append({
            'tool_name': tool_name,
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'execution_time': execution_time
        })
    
    def record_cycle_timing(self, cycle: int, duration: float, phase_timings: Dict[str, float]):
        """Record cycle execution timings"""
        if not self.enable_monitoring:
            return
            
        self.metrics['cycle_timings'].append({
            'cycle': cycle,
            'timestamp': datetime.now().isoformat(),
            'total_duration': duration,
            'phase_timings': phase_timings
        })
    
    def record_error(self, error_type: str, error_message: str, component: str):
        """Record error events"""
        if not self.enable_monitoring:
            return
            
        self.metrics['error_events'].append({
            'error_type': error_type,
            'error_message': error_message,
            'component': component,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics"""
        if not self.enable_monitoring:
            return {}
        
        runtime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'runtime_seconds': runtime,
            'total_cycles': len(self.metrics['cycle_timings']),
            'total_memory_events': len(self.metrics['memory_events']),
            'total_tool_calls': len(self.metrics['tool_usage']),
            'total_errors': len(self.metrics['error_events']),
            'preference_changes': len(self.metrics['preference_changes']),
            'behavior_patterns_analyzed': len(self.metrics['behavior_patterns']),
            'attractor_patterns_recorded': len(self.metrics['attractor_patterns']),
            'preference_drift_analyses': len(self.metrics['preference_drift_analysis']),
            'avg_cycle_time': sum(t['total_duration'] for t in self.metrics['cycle_timings']) / max(1, len(self.metrics['cycle_timings']))
        }
    
    def record_behavior_pattern(self, cycle: int, intention: str, tool_used: Optional[str], 
                               reward: float, state_similarity: float):
        """Record behavior patterns for analysis"""
        if not self.enable_monitoring:
            return
        
        # Update recent behavior tracking
        self.recent_intentions.append(intention)
        self.recent_tools.append(tool_used or "none")
        self.recent_rewards.append(reward)
        
        # Keep only recent window
        if len(self.recent_intentions) > self.pattern_window_size:
            self.recent_intentions.pop(0)
            self.recent_tools.pop(0)
            self.recent_rewards.pop(0)
        
        # Analyze patterns if we have enough data
        if len(self.recent_intentions) >= self.pattern_window_size:
            pattern_analysis = self._analyze_behavior_patterns()
            
            self.metrics['behavior_patterns'].append({
                'cycle': cycle,
                'timestamp': datetime.now().isoformat(),
                'intention_repetition_rate': pattern_analysis['intention_repetition'],
                'tool_preference_entropy': pattern_analysis['tool_entropy'],
                'reward_trend': pattern_analysis['reward_trend'],
                'state_similarity': state_similarity,
                'dominant_intention_category': pattern_analysis['dominant_intention'],
                'most_used_tool': pattern_analysis['most_used_tool']
            })
    
    def record_attractor_pattern(self, cycle: int, attractor_count: int, 
                                strongest_attractor_strength: float, pattern_recognized: bool):
        """Record attractor dynamics patterns"""
        if not self.enable_monitoring:
            return
        
        self.metrics['attractor_patterns'].append({
            'cycle': cycle,
            'timestamp': datetime.now().isoformat(),
            'attractor_count': attractor_count,
            'strongest_attractor_strength': strongest_attractor_strength,
            'pattern_recognized': pattern_recognized
        })
    
    def _analyze_behavior_patterns(self) -> Dict[str, Any]:
        """Analyze recent behavior patterns"""
        # Intention repetition analysis
        unique_intentions = len(set(self.recent_intentions))
        intention_repetition = 1.0 - (unique_intentions / len(self.recent_intentions))
        
        # Tool usage entropy
        tool_counts = {}
        for tool in self.recent_tools:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
        
        if len(tool_counts) > 1:
            total_tools = sum(tool_counts.values())
            tool_entropy = -sum((count/total_tools) * np.log2(count/total_tools) 
                               for count in tool_counts.values())
        else:
            tool_entropy = 0.0
        
        # Reward trend analysis
        if len(self.recent_rewards) >= 3:
            recent_avg = np.mean(self.recent_rewards[-3:])
            earlier_avg = np.mean(self.recent_rewards[:-3])
            reward_trend = recent_avg - earlier_avg
        else:
            reward_trend = 0.0
        
        # Dominant patterns
        most_common_intention = max(set(self.recent_intentions), 
                                  key=self.recent_intentions.count)
        most_used_tool = max(set(self.recent_tools), 
                           key=self.recent_tools.count)
        
        return {
            'intention_repetition': intention_repetition,
            'tool_entropy': tool_entropy,
            'reward_trend': reward_trend,
            'dominant_intention': most_common_intention,
            'most_used_tool': most_used_tool
        }
    
    def _analyze_preference_drift(self, semantic_category: str):
        """Analyze preference drift for a specific category"""
        history = self.preference_history[semantic_category]
        baseline = self.preference_baseline[semantic_category]
        
        if len(history) < 5:
            return
        
        # Calculate drift metrics
        recent_weights = [weight for _, weight in history[-5:]]
        current_weight = recent_weights[-1]
        
        # Drift magnitude from baseline
        drift_from_baseline = abs(current_weight - baseline)
        
        # Trend analysis (linear regression slope approximation)
        x_values = list(range(len(recent_weights)))
        if len(recent_weights) > 1:
            slope = (recent_weights[-1] - recent_weights[0]) / (len(recent_weights) - 1)
        else:
            slope = 0.0
        
        # Volatility (standard deviation of recent changes)
        if len(recent_weights) > 1:
            changes = [recent_weights[i] - recent_weights[i-1] 
                      for i in range(1, len(recent_weights))]
            volatility = float(np.std(changes))
        else:
            volatility = 0.0
        
        drift_analysis = {
            'semantic_category': semantic_category,
            'timestamp': datetime.now().isoformat(),
            'drift_from_baseline': drift_from_baseline,
            'trend_slope': slope,
            'volatility': volatility,
            'current_weight': current_weight,
            'baseline_weight': baseline,
            'significant_drift': drift_from_baseline > 0.3  # Configurable threshold
        }
        
        self.metrics['preference_drift_analysis'].append(drift_analysis)
    
    def get_preference_drift_summary(self) -> Dict[str, Any]:
        """Get summary of preference drift patterns"""
        if not self.enable_monitoring:
            return {}
        
        # Always return basic tracking info
        base_summary = {
            'total_categories_tracked': len(self.preference_history),
            'categories_with_significant_drift': 0,
            'average_volatility': 0.0,
            'recent_significant_drifts': 0,
            'most_volatile_category': None
        }
        
        if not self.metrics['preference_drift_analysis']:
            return base_summary
        
        recent_drifts = self.metrics['preference_drift_analysis'][-10:]  # Last 10 analyses
        
        significant_drifts = [d for d in recent_drifts if d['significant_drift']]
        avg_volatility = np.mean([d['volatility'] for d in recent_drifts])
        categories_with_drift = len(set(d['semantic_category'] for d in significant_drifts))
        
        return {
            'total_categories_tracked': len(self.preference_history),
            'categories_with_significant_drift': categories_with_drift,
            'average_volatility': float(avg_volatility),
            'recent_significant_drifts': len(significant_drifts),
            'most_volatile_category': max(recent_drifts, key=lambda x: x['volatility'])['semantic_category'] if recent_drifts else None
        }
    
    def get_behavior_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of behavior patterns"""
        if not self.enable_monitoring:
            return {}
        
        # Always return basic structure
        base_summary = {
            'average_intention_repetition': 0.0,
            'average_tool_entropy': 0.0,
            'average_reward_trend': 0.0,
            'behavior_stability': 1.0,
            'exploration_level': 0.0
        }
        
        if not self.metrics['behavior_patterns']:
            return base_summary
        
        recent_patterns = self.metrics['behavior_patterns'][-5:]  # Last 5 pattern analyses
        
        avg_repetition = np.mean([p['intention_repetition_rate'] for p in recent_patterns])
        avg_entropy = np.mean([p['tool_preference_entropy'] for p in recent_patterns])
        avg_reward_trend = np.mean([p['reward_trend'] for p in recent_patterns])
        
        return {
            'average_intention_repetition': float(avg_repetition),
            'average_tool_entropy': float(avg_entropy),
            'average_reward_trend': float(avg_reward_trend),
            'behavior_stability': float(1.0 - avg_repetition),  # Lower repetition = more diverse behavior
            'exploration_level': float(avg_entropy)  # Higher entropy = more exploration
        }

    def export_metrics(self, filepath: str):
        """Export all metrics to JSON file"""
        if not self.enable_monitoring:
            return
        
        # Create directory if filepath contains a directory
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump({
                'summary': self.get_metrics_summary(),
                'preference_drift_summary': self.get_preference_drift_summary(),
                'behavior_pattern_summary': self.get_behavior_pattern_summary(),
                'detailed_metrics': self.metrics
            }, f, indent=2)


# Global instances
logger = AgentLogger()
instrumentation = InstrumentationCollector()


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None, 
                 enable_monitoring: bool = True):
    """Setup global logging and instrumentation"""
    global logger, instrumentation
    
    logger = AgentLogger(log_level, log_file)
    instrumentation = InstrumentationCollector(enable_monitoring)
    
    logger.info("Logging and instrumentation initialized", 
                log_level=log_level, monitoring_enabled=enable_monitoring)