"""
Dukkha (Dissatisfaction) Tab

Real-time visualization of healthy dissatisfaction levels that drive agent growth.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from collections import deque
from datetime import datetime
import warnings
import logging
warnings.filterwarnings('ignore')

from .base_tab import BaseTab

logger = logging.getLogger(__name__)


class DukkhaTab(BaseTab):
    """Dukkha (dissatisfaction) visualization tab."""
    
    def __init__(self, notebook, collector):
        """Initialize dukkha tab."""
        self.collector = collector
        self.history_length = 100
        
        # Dukkha type histories
        self.dukkha_histories = {
            'stagnation_dissatisfaction': deque(maxlen=self.history_length),
            'curiosity_gap_tension': deque(maxlen=self.history_length),
            'mastery_challenge_pressure': deque(maxlen=self.history_length),
            'existential_questioning': deque(maxlen=self.history_length),
            'goal_frustration': deque(maxlen=self.history_length),
            'novelty_hunger': deque(maxlen=self.history_length)
        }
        
        self.total_dissatisfaction_history = deque(maxlen=self.history_length)
        self.time_history = deque(maxlen=self.history_length)
        self.emotional_state_history = deque(maxlen=self.history_length)
        self.growth_suggestions_history = deque(maxlen=self.history_length)
        
        super().__init__(notebook, "Dukkha State", "🔥")
        self._create_dukkha_display()
    
    def _create_dukkha_display(self):
        """Create the dukkha visualization display."""
        # Create figure with subplots
        fig = Figure(figsize=(16, 12), facecolor='white')
        fig.suptitle('🔥 Dukkha (Dissatisfaction) Real-Time Monitoring', fontsize=16, fontweight='bold')
        
        # Create subplots in a 3x2 grid
        self.ax_total = fig.add_subplot(3, 2, 1)
        self.ax_types = fig.add_subplot(3, 2, 2)
        self.ax_radar = fig.add_subplot(3, 2, 3, projection='polar')
        self.ax_emotional = fig.add_subplot(3, 2, 4)
        self.ax_suggestions = fig.add_subplot(3, 2, 5)
        self.ax_insights = fig.add_subplot(3, 2, 6)
        
        # Setup plots
        self._setup_plots()
        
        # Add matplotlib canvas
        self._add_matplotlib_canvas(fig)
    
    def _setup_plots(self):
        """Setup the visualization plots"""
        # Total dissatisfaction over time
        self.ax_total.set_title('Total Dissatisfaction Over Time')
        self.ax_total.set_xlabel('Time Steps')
        self.ax_total.set_ylabel('Total Dissatisfaction')
        self.ax_total.grid(True, alpha=0.3)
        
        # Individual dukkha types
        self.ax_types.set_title('Dukkha Types Over Time')
        self.ax_types.set_xlabel('Time Steps')
        self.ax_types.set_ylabel('Dissatisfaction Level')
        self.ax_types.grid(True, alpha=0.3)
        
        # Radar chart for current state
        self.ax_radar.set_title('Current Dukkha Profile', pad=20)
        
        # Emotional state evolution
        self.ax_emotional.set_title('Emotional State Evolution')
        
        # Growth suggestions
        self.ax_suggestions.set_title('Recent Growth Suggestions')
        
        # Insights and analysis
        self.ax_insights.set_title('Dukkha Insights')
        
        plt.tight_layout()
    
    def update_display(self):
        """Update dukkha display with current data."""
        try:
            logger.info("=== DUKKHA UPDATE_DISPLAY START ===")
            
            # Check if we have dukkha data in the collector
            if not hasattr(self.collector, 'metrics'):
                logger.info("No metrics in collector, showing no data message")
                self._show_no_data_message()
                return
            
            logger.info(f"Collector metrics keys: {list(self.collector.metrics.keys())}")
            
            # Log detailed info about each metric type
            for key, value in self.collector.metrics.items():
                if isinstance(value, (list, tuple)):
                    logger.info(f"  {key}: {len(value)} items")
                    if value and len(value) > 0:
                        logger.info(f"    Sample item type: {type(value[-1])}")
                        if hasattr(value[-1], '__dict__'):
                            logger.info(f"    Sample item attributes: {list(value[-1].__dict__.keys())}")
                else:
                    logger.info(f"  {key}: {type(value)} = {str(value)[:100]}")
            
            # Look for dukkha data in various possible locations
            dukkha_data = None
            
            # Check for direct dukkha metrics
            if 'dukkha_states' in self.collector.metrics:
                logger.info("Found dukkha_states in metrics")
                dukkha_trajectories = self.collector.metrics['dukkha_states']
                if dukkha_trajectories:
                    dukkha_data = dukkha_trajectories[-1]
                    logger.info(f"Using dukkha_states data: {type(dukkha_data)}")
            
            # Check for pathos trajectories that might contain dukkha
            elif 'pathos_trajectories' in self.collector.metrics:
                logger.info("Found pathos_trajectories in metrics")
                pathos_trajectories = self.collector.metrics['pathos_trajectories']
                if pathos_trajectories:
                    latest_pathos = pathos_trajectories[-1]
                    if 'dukkha_state' in latest_pathos:
                        dukkha_data = latest_pathos['dukkha_state']
                        logger.info(f"Using pathos dukkha_state data: {type(dukkha_data)}")
            
            # Check for memory traces that might contain dukkha info
            elif 'memory_traces' in self.collector.metrics:
                logger.info("Found memory_traces in metrics")
                memory_traces = self.collector.metrics['memory_traces']
                logger.info(f"Memory traces count: {len(memory_traces) if memory_traces else 0}")
                
                if memory_traces:
                    # Look for dukkha information in recent memory traces
                    logger.info("Checking recent memory traces for dukkha data...")
                    for i, trace in enumerate(reversed(memory_traces[-10:])):  # Check last 10 traces
                        logger.info(f"  Trace {i}: type={type(trace)}")
                        logger.info(f"  Trace {i}: has metadata={hasattr(trace, 'metadata')}")
                        
                        if hasattr(trace, 'metadata'):
                            logger.info(f"  Trace {i}: metadata keys={list(trace.metadata.keys()) if trace.metadata else 'None'}")
                            if 'dukkha_state' in trace.metadata:
                                dukkha_data = trace.metadata['dukkha_state']
                                logger.info(f"  FOUND dukkha_state in trace {i}: {dukkha_data}")
                                logger.info(f"Using memory trace dukkha_state data: {type(dukkha_data)}")
                                break
                        else:
                            logger.info(f"  Trace {i}: no metadata attribute")
                    
                    if not dukkha_data:
                        logger.warning("No dukkha_state found in any recent memory traces!")
                        # Let's check what's actually in the metadata
                        if memory_traces:
                            sample_trace = memory_traces[-1]
                            logger.warning(f"Sample trace type: {type(sample_trace)}")
                            logger.warning(f"Sample trace attributes: {dir(sample_trace)}")
                            if hasattr(sample_trace, 'metadata') and sample_trace.metadata:
                                logger.warning(f"Sample trace metadata: {sample_trace.metadata}")
            else:
                logger.warning("No memory_traces found in collector metrics")
            
            if dukkha_data:
                logger.info("Found real dukkha data, updating display")
                self._update_with_dukkha_data(dukkha_data)
            else:
                logger.warning("NO REAL DUKKHA DATA FOUND - This should not happen in production!")
                logger.warning("Available metrics keys: " + str(list(self.collector.metrics.keys()) if hasattr(self.collector, 'metrics') else 'No metrics'))
                # PRODUCTION: Show no data message instead of demo data
                self._show_no_data_message()
                
            logger.info("=== DUKKHA UPDATE_DISPLAY END ===")
                
        except Exception as e:
            logger.error(f"=== DUKKHA UPDATE_DISPLAY ERROR ===")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            self._show_error_message(f"Error updating dukkha display: {str(e)}")
    
    def _update_with_dukkha_data(self, dukkha_data):
        """Update visualization with actual dukkha data."""
        try:
            logger.info("=== _UPDATE_WITH_DUKKHA_DATA START ===")
            logger.info(f"dukkha_data type: {type(dukkha_data)}")
            logger.info(f"dukkha_data content: {dukkha_data}")
            
            # Extract dukkha levels
            dukkha_levels = dukkha_data.get('dukkha_levels', {})
            total_dissatisfaction = dukkha_data.get('total_dissatisfaction', 0.0)
            
            logger.info(f"dukkha_levels: {dukkha_levels}")
            logger.info(f"total_dissatisfaction: {total_dissatisfaction}")
            
            # Store data
            for dukkha_type, level in dukkha_levels.items():
                if dukkha_type in self.dukkha_histories:
                    self.dukkha_histories[dukkha_type].append(level)
                    logger.debug(f"Added {dukkha_type}: {level}")
            
            self.total_dissatisfaction_history.append(total_dissatisfaction)
            self.time_history.append(len(self.time_history))
            
            logger.info(f"History lengths - time: {len(self.time_history)}, total: {len(self.total_dissatisfaction_history)}")
            
            # Determine emotional state based on dukkha
            emotional_state = self._determine_emotional_state(dukkha_levels, total_dissatisfaction)
            self.emotional_state_history.append(emotional_state)
            
            logger.info(f"Emotional state: {emotional_state}")
            
            # Generate growth suggestions
            suggestions = self._generate_growth_suggestions(dukkha_levels)
            self.growth_suggestions_history.append(suggestions)
            
            logger.info(f"Growth suggestions: {suggestions}")
            
            # Update plots
            logger.info("About to call _update_plots")
            self._update_plots()
            logger.info("_update_plots completed")
            
            # Refresh canvas with error handling
            if self.canvas:
                try:
                    logger.info("About to draw canvas")
                    self.canvas.draw()
                    logger.info("Canvas draw completed")
                except Exception as canvas_error:
                    logger.warning(f"Canvas drawing error in dukkha tab: {canvas_error}")
                    # Try to clear and redraw
                    try:
                        self.canvas.flush_events()
                        self.canvas.draw()
                        logger.info("Canvas draw retry succeeded")
                    except Exception as retry_error:
                        logger.error(f"Failed to recover from canvas error: {retry_error}")
                        # Don't re-raise - let the tab continue functioning
            
            logger.info("=== _UPDATE_WITH_DUKKHA_DATA END ===")
            
        except Exception as e:
            logger.error(f"=== _UPDATE_WITH_DUKKHA_DATA ERROR ===")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise  # Re-raise to be caught by update_display
    
    def _generate_demo_dukkha_data(self):
        """Generate demonstration dukkha data when real data isn't available."""
        # Simulate realistic dukkha patterns
        time_factor = len(self.time_history) * 0.1
        
        # Create realistic dukkha levels that evolve over time
        dukkha_levels = {
            'stagnation_dissatisfaction': max(0.0, 0.3 + 0.4 * np.sin(time_factor * 0.5) + np.random.normal(0, 0.1)),
            'curiosity_gap_tension': max(0.0, 0.2 + 0.3 * np.sin(time_factor * 0.7 + 1) + np.random.normal(0, 0.08)),
            'mastery_challenge_pressure': max(0.0, 0.25 + 0.2 * np.sin(time_factor * 0.3 + 2) + np.random.normal(0, 0.06)),
            'existential_questioning': max(0.0, 0.4 + 0.3 * np.sin(time_factor * 0.2 + 3) + np.random.normal(0, 0.05)),
            'goal_frustration': max(0.0, 0.15 + 0.25 * np.sin(time_factor * 0.6 + 4) + np.random.normal(0, 0.07)),
            'novelty_hunger': max(0.0, 0.35 + 0.4 * np.sin(time_factor * 0.8 + 5) + np.random.normal(0, 0.09))
        }
        
        # Clamp values to [0, 1]
        for key in dukkha_levels:
            dukkha_levels[key] = min(1.0, max(0.0, dukkha_levels[key]))
        
        total_dissatisfaction = sum(dukkha_levels.values()) / len(dukkha_levels)
        
        # Create demo data structure
        demo_data = {
            'dukkha_levels': dukkha_levels,
            'total_dissatisfaction': total_dissatisfaction,
            'active_curiosities': int(3 + 2 * np.sin(time_factor * 0.3)),
            'current_goals': int(2 + np.sin(time_factor * 0.4)),
            'mastery_domains': int(4 + np.sin(time_factor * 0.2)),
            'action_diversity': max(0.1, min(1.0, 0.6 + 0.3 * np.sin(time_factor * 0.5))),
            'time_since_significant_change': max(0.1, 5 + 10 * np.sin(time_factor * 0.1))
        }
        
        self._update_with_dukkha_data(demo_data)
    
    def _determine_emotional_state(self, dukkha_levels, total_dissatisfaction):
        """Determine emotional state based on dukkha levels."""
        if total_dissatisfaction > 0.6:
            if dukkha_levels.get('stagnation_dissatisfaction', 0) > 0.5:
                return "Restless & Eager for Change"
            elif dukkha_levels.get('curiosity_gap_tension', 0) > 0.4:
                return "Curious & Driven to Explore"
            elif dukkha_levels.get('existential_questioning', 0) > 0.4:
                return "Contemplative & Searching"
            else:
                return "Challenged & Motivated"
        elif total_dissatisfaction > 0.3:
            return "Mildly Dissatisfied but Purposeful"
        else:
            return "Content but Growth-Aware"
    
    def _generate_growth_suggestions(self, dukkha_levels):
        """Generate growth suggestions based on dukkha levels."""
        suggestions = []
        
        # Get top 2 dissatisfaction sources
        sorted_dukkha = sorted(dukkha_levels.items(), key=lambda x: x[1], reverse=True)[:2]
        
        suggestion_map = {
            'stagnation_dissatisfaction': "Try a completely different approach",
            'curiosity_gap_tension': "Investigate unresolved questions",
            'mastery_challenge_pressure': "Seek more complex challenges",
            'existential_questioning': "Reflect on deeper purpose",
            'goal_frustration': "Break down goals into smaller steps",
            'novelty_hunger': "Explore something entirely new"
        }
        
        for dukkha_type, level in sorted_dukkha:
            if level > 0.2:  # Only suggest for significant dissatisfaction
                suggestions.append(suggestion_map.get(dukkha_type, "Continue growth efforts"))
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def _update_plots(self):
        """Update all visualization plots."""
        try:
            logger.info("=== _UPDATE_PLOTS START ===")
            
            if len(self.total_dissatisfaction_history) < 1:
                logger.info("No data in total_dissatisfaction_history, returning")
                return
            
            logger.info(f"Data lengths - total_dissatisfaction: {len(self.total_dissatisfaction_history)}, time: {len(self.time_history)}")
            
            # Update total dissatisfaction plot
            logger.info("Updating total dissatisfaction plot")
            self.ax_total.clear()
            self.ax_total.set_title('Total Dissatisfaction Over Time')
            self.ax_total.set_xlabel('Time Steps')
            self.ax_total.set_ylabel('Total Dissatisfaction')
            self.ax_total.grid(True, alpha=0.3)
            
            times = list(self.time_history)
            total_values = list(self.total_dissatisfaction_history)
            
            logger.info(f"times: {times}")
            logger.info(f"total_values: {total_values}")
            
            if len(times) == 0 or len(total_values) == 0:
                logger.info("Empty times or total_values, returning")
                return
            
            # Ensure times and values have the same length
            min_length = min(len(times), len(total_values))
            logger.info(f"min_length: {min_length}")
            times = times[:min_length]
            total_values = total_values[:min_length]
            
            logger.info(f"After slicing - times: {times}, total_values: {total_values}")
            
            try:
                self.ax_total.plot(times, total_values, 'r-', linewidth=2, label='Total Dissatisfaction')
                logger.info("Total plot line created successfully")
            except Exception as plot_error:
                logger.error(f"Error creating total plot line: {plot_error}")
                raise
            
            # Add threshold lines
            try:
                self.ax_total.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Mild Threshold')
                self.ax_total.axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='High Threshold')
                logger.info("Threshold lines added successfully")
            except Exception as threshold_error:
                logger.error(f"Error adding threshold lines: {threshold_error}")
                raise
            
            # Color-code the line based on intensity
            try:
                logger.info(f"About to create scatter points for {len(total_values)} values")
                for i in range(len(total_values)):
                    color = 'green' if total_values[i] < 0.3 else 'orange' if total_values[i] < 0.6 else 'red'
                    logger.debug(f"Scatter point {i}: times[{i}]={times[i]}, total_values[{i}]={total_values[i]}, color={color}")
                    self.ax_total.scatter(times[i], total_values[i], c=color, alpha=0.6, s=30)
                logger.info("Scatter points created successfully")
            except Exception as scatter_error:
                logger.error(f"Error creating scatter points: {scatter_error}")
                logger.error(f"times type: {type(times)}, len: {len(times)}")
                logger.error(f"total_values type: {type(total_values)}, len: {len(total_values)}")
                logger.error(f"Error at index: {i if 'i' in locals() else 'unknown'}")
                raise
            
            try:
                self.ax_total.legend()
                self.ax_total.set_ylim(0, 1)
                logger.info("Total plot legend and limits set successfully")
            except Exception as legend_error:
                logger.error(f"Error setting legend/limits: {legend_error}")
                raise
            
            logger.info("Total dissatisfaction plot completed successfully")
            
            # Update individual dukkha types plot
            logger.info("Updating individual dukkha types plot")
            self.ax_types.clear()
            self.ax_types.set_title('Dukkha Types Over Time')
            self.ax_types.set_xlabel('Time Steps')
            self.ax_types.set_ylabel('Dissatisfaction Level')
            self.ax_types.grid(True, alpha=0.3)
            
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
            labels = [
                'Stagnation', 'Curiosity Gap', 'Mastery Challenge',
                'Existential', 'Goal Frustration', 'Novelty Hunger'
            ]
            
            logger.info(f"Processing {len(self.dukkha_histories)} dukkha types")
            
            for i, (dukkha_type, history) in enumerate(self.dukkha_histories.items()):
                logger.debug(f"Processing dukkha type {i}: {dukkha_type}, history length: {len(history)}")
                if len(history) > 0 and len(times) > 0:
                    values = list(history)
                    logger.debug(f"  values: {values}")
                    # Ensure we have matching time indices for the values
                    try:
                        if len(values) <= len(times):
                            history_times = times[-len(values):]
                            logger.debug(f"  Case 1: history_times = {history_times}")
                        else:
                            history_times = times
                            values = values[-len(times):]
                            logger.debug(f"  Case 2: history_times = {history_times}, values = {values}")
                        
                        # Only plot if we have data and matching lengths
                        if len(history_times) > 0 and len(values) > 0 and len(history_times) == len(values):
                            logger.debug(f"  Plotting {dukkha_type} with {len(history_times)} points")
                            self.ax_types.plot(history_times, values, 
                                             color=colors[i % len(colors)], 
                                             linewidth=1.5, 
                                             label=labels[i] if i < len(labels) else dukkha_type,
                                             alpha=0.8)
                            logger.debug(f"  Successfully plotted {dukkha_type}")
                        else:
                            logger.debug(f"  Skipping {dukkha_type} - length mismatch or empty data")
                    except (IndexError, ValueError) as e:
                        # Skip this dukkha type if there's a slicing error
                        logger.debug(f"Skipping dukkha type {dukkha_type} due to slicing error: {e}")
                        continue
                    except Exception as plot_error:
                        logger.error(f"Unexpected error plotting {dukkha_type}: {plot_error}")
                        raise
            
            try:
                self.ax_types.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                self.ax_types.set_ylim(0, 1)
                logger.info("Dukkha types legend set successfully")
            except Exception as legend_error:
                logger.warning(f"Legend error in dukkha types plot: {legend_error}")
                # Try simpler legend
                try:
                    self.ax_types.legend()
                    self.ax_types.set_ylim(0, 1)
                    logger.info("Simple legend fallback successful")
                except Exception:
                    logger.warning("Simple legend fallback also failed")
                    pass  # Skip legend if it continues to fail
            
            logger.info("Individual dukkha types plot completed successfully")
            
            # Update radar chart for current state
            if len(self.total_dissatisfaction_history) > 0:
                logger.info("Updating radar chart")
                self._update_radar_chart()
                logger.info("Radar chart completed")
            
            # Update emotional state plot
            logger.info("Updating emotional state plot")
            self._update_emotional_state_plot()
            logger.info("Emotional state plot completed")
            
            # Update growth suggestions
            logger.info("Updating growth suggestions plot")
            self._update_growth_suggestions_plot()
            logger.info("Growth suggestions plot completed")
            
            # Update insights
            logger.info("Updating insights plot")
            self._update_insights_plot()
            logger.info("Insights plot completed")
            
            logger.info("=== _UPDATE_PLOTS END ===")
            
        except Exception as e:
            logger.error(f"=== _UPDATE_PLOTS ERROR ===")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise  # Re-raise to be caught by calling method
    
    def _update_radar_chart(self):
        """Update the radar chart showing current dukkha profile."""
        self.ax_radar.clear()
        self.ax_radar.set_title('Current Dukkha Profile', pad=20)
        
        # Get current values
        current_values = []
        labels = []
        
        label_map = {
            'stagnation_dissatisfaction': 'Stagnation',
            'curiosity_gap_tension': 'Curiosity',
            'mastery_challenge_pressure': 'Mastery',
            'existential_questioning': 'Existential',
            'goal_frustration': 'Goals',
            'novelty_hunger': 'Novelty'
        }
        
        for dukkha_type, history in self.dukkha_histories.items():
            if len(history) > 0:
                current_values.append(history[-1])
                labels.append(label_map.get(dukkha_type, dukkha_type))
        
        if current_values:
            try:
                # Create radar chart
                angles = np.linspace(0, 2 * np.pi, len(current_values), endpoint=False)
                # Complete the circle by adding the first value at the end
                current_values_circle = current_values + [current_values[0]]
                angles_circle = np.concatenate((angles, [angles[0]]))
                
                self.ax_radar.plot(angles_circle, current_values_circle, 'r-', linewidth=2, alpha=0.8)
                self.ax_radar.fill(angles_circle, current_values_circle, 'red', alpha=0.25)
                
                # Add labels
                self.ax_radar.set_xticks(angles)
                self.ax_radar.set_xticklabels(labels)
                self.ax_radar.set_ylim(0, 1)
                
                # Add grid circles
                self.ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
                self.ax_radar.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
                self.ax_radar.grid(True)
            except Exception as radar_error:
                logger.warning(f"Radar chart error: {radar_error}")
                # Clear the radar chart and show error message
                self.ax_radar.clear()
                self.ax_radar.text(0.5, 0.5, 'Radar chart\nunavailable', 
                                 ha='center', va='center', transform=self.ax_radar.transAxes)
    
    def _update_emotional_state_plot(self):
        """Update emotional state evolution plot."""
        self.ax_emotional.clear()
        self.ax_emotional.set_title('Emotional State Evolution')
        
        if not self.emotional_state_history:
            self.ax_emotional.text(0.5, 0.5, 'No emotional state data', 
                                 ha='center', va='center', transform=self.ax_emotional.transAxes)
            return
        
        # Count emotional states
        state_counts = {}
        for state in self.emotional_state_history:
            state_counts[state] = state_counts.get(state, 0) + 1
        
        # Create pie chart of emotional states
        if state_counts:
            states = list(state_counts.keys())
            counts = list(state_counts.values())
            
            # Fix: Handle single state case to avoid slice error
            if len(states) == 1:
                colors = ['lightblue']
            else:
                colors = plt.cm.Set3(np.linspace(0, 1, len(states)))
            
            wedges, texts, autotexts = self.ax_emotional.pie(counts, labels=states, autopct='%1.1f%%',
                                                           colors=colors, startangle=90)
            
            # Improve text readability
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        # Show current emotional state
        if self.emotional_state_history:
            current_state = self.emotional_state_history[-1]
            try:
                self.ax_emotional.text(0.5, -1.3, f'Current: {current_state}', 
                                     ha='center', va='center', transform=self.ax_emotional.transAxes,
                                     fontsize=12, fontweight='bold',
                                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            except Exception as text_error:
                logger.warning(f"Text display error in emotional state plot: {text_error}")
                # Try simpler text without bbox
                try:
                    self.ax_emotional.text(0.5, -1.3, f'Current: {current_state}', 
                                         ha='center', va='center', transform=self.ax_emotional.transAxes,
                                         fontsize=12, fontweight='bold')
                except Exception:
                    pass  # Skip text if it continues to fail
    
    def _update_growth_suggestions_plot(self):
        """Update growth suggestions display."""
        self.ax_suggestions.clear()
        self.ax_suggestions.set_title('Recent Growth Suggestions')
        
        if not self.growth_suggestions_history:
            self.ax_suggestions.text(0.5, 0.5, 'No growth suggestions available', 
                                   ha='center', va='center', transform=self.ax_suggestions.transAxes)
            return
        
        # Get recent unique suggestions
        recent_suggestions = []
        # Convert deque to list for slicing, then get last 5 items
        suggestions_list = list(self.growth_suggestions_history)
        last_5_suggestions = suggestions_list[-5:] if len(suggestions_list) >= 5 else suggestions_list
        
        for suggestions in reversed(last_5_suggestions):  # Last 5 cycles
            for suggestion in suggestions:
                if suggestion not in recent_suggestions:
                    recent_suggestions.append(suggestion)
        
        # Display suggestions as text
        y_pos = 0.9
        for i, suggestion in enumerate(recent_suggestions[:6]):  # Show top 6
            self.ax_suggestions.text(0.05, y_pos, f"• {suggestion}", 
                                   transform=self.ax_suggestions.transAxes, fontsize=10,
                                   verticalalignment='top', wrap=True)
            y_pos -= 0.15
        
        self.ax_suggestions.set_xlim(0, 1)
        self.ax_suggestions.set_ylim(0, 1)
        self.ax_suggestions.axis('off')
    
    def _update_insights_plot(self):
        """Update dukkha insights and analysis."""
        self.ax_insights.clear()
        self.ax_insights.set_title('Dukkha Insights')
        
        if not self.total_dissatisfaction_history:
            self.ax_insights.text(0.5, 0.5, 'No insights available', 
                                ha='center', va='center', transform=self.ax_insights.transAxes)
            return
        
        # Calculate insights
        current_total = self.total_dissatisfaction_history[-1] if self.total_dissatisfaction_history else 0.0
        
        # Trend analysis
        trend = "Stable"
        if len(self.total_dissatisfaction_history) >= 5:
            try:
                recent_values = list(self.total_dissatisfaction_history)[-5:]
                if len(recent_values) >= 2:
                    trend = "Increasing" if recent_values[-1] > recent_values[0] else "Decreasing"
            except (IndexError, TypeError):
                trend = "Stable"
        
        # Dominant dukkha type
        dominant_name = "Unknown"
        dominant_value = 0.0
        
        try:
            current_dukkha = {}
            for dukkha_type, history in self.dukkha_histories.items():
                if len(history) > 0:
                    current_dukkha[dukkha_type] = history[-1]
            
            if current_dukkha:
                dominant_type = max(current_dukkha.items(), key=lambda x: x[1])
                dominant_name = dominant_type[0].replace('_', ' ').title()
                dominant_value = dominant_type[1]
        except (ValueError, IndexError, TypeError):
            pass
        
        # Growth assessment
        if current_total > 0.6:
            growth_status = "🔥 High Growth Potential"
            growth_color = "red"
        elif current_total > 0.3:
            growth_status = "⚡ Moderate Growth Drive"
            growth_color = "orange"
        else:
            growth_status = "😌 Low Growth Pressure"
            growth_color = "green"
        
        # Display insights
        insights = [
            f"Total Dissatisfaction: {current_total:.3f}",
            f"Trend: {trend}",
            f"Dominant Type: {dominant_name} ({dominant_value:.3f})",
            f"Growth Status: {growth_status}",
            "",
            "💡 Dukkha Philosophy:",
            "Dissatisfaction is not suffering—",
            "it's the healthy tension that",
            "drives all genuine growth."
        ]
        
        y_pos = 0.95
        for insight in insights:
            try:
                color = growth_color if "Growth Status" in insight else "black"
                fontweight = "bold" if any(x in insight for x in ["Total", "Trend", "Dominant", "Growth Status"]) else "normal"
                
                self.ax_insights.text(0.05, y_pos, insight, 
                                    transform=self.ax_insights.transAxes, fontsize=10,
                                    verticalalignment='top', color=color, fontweight=fontweight)
                y_pos -= 0.1
            except Exception:
                continue  # Skip problematic insights
        
        self.ax_insights.set_xlim(0, 1)
        self.ax_insights.set_ylim(0, 1)
        self.ax_insights.axis('off')
    
    def _show_no_data_message(self):
        """Show message when no data is available."""
        for ax in [self.ax_total, self.ax_types, self.ax_radar, self.ax_emotional, self.ax_suggestions, self.ax_insights]:
            ax.clear()
            ax.text(0.5, 0.5, '🔥 No Dukkha Data Available\n\nWaiting for agent session to start...\n\nDukkha measurements will appear here\nwhen the agent begins running with\nthe dukkha engine enabled.\n\nStart an agent session to see\nreal-time dissatisfaction monitoring.', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7))
        
        if self.canvas:
            self.canvas.draw()
    
    def _show_error_message(self, error_msg: str):
        """Show error message."""
        for ax in [self.ax_total, self.ax_types, self.ax_radar, self.ax_emotional, self.ax_suggestions, self.ax_insights]:
            ax.clear()
            ax.text(0.5, 0.5, f'Error: {error_msg}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10, color='red',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))
        
        if self.canvas:
            self.canvas.draw()
    
    def export_data(self, export_dir: str):
        """Export dukkha data to files."""
        import os
        import json
        
        # Export dukkha history data
        dukkha_data = {
            'total_dissatisfaction_history': list(self.total_dissatisfaction_history),
            'dukkha_histories': {k: list(v) for k, v in self.dukkha_histories.items()},
            'emotional_state_history': list(self.emotional_state_history),
            'growth_suggestions_history': list(self.growth_suggestions_history),
            'time_history': list(self.time_history)
        }
        
        with open(os.path.join(export_dir, 'dukkha_data.json'), 'w') as f:
            json.dump(dukkha_data, f, indent=2)
        
        # Export the visualization
        super().export_data(export_dir)