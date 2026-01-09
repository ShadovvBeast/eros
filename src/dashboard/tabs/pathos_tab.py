"""
Pathos State Tab

Real-time Pathos affective state visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from collections import deque
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optional sklearn import
try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    PCA = None

from .base_tab import BaseTab


class PathosTab(BaseTab):
    """Pathos state visualization tab."""
    
    def __init__(self, notebook, collector, pathos_states):
        """Initialize pathos tab."""
        self.collector = collector
        self.pathos_states = pathos_states
        self.state_dimension = 128
        self.history_length = 100
        self.state_history = deque(maxlen=self.history_length)
        self.reward_history = deque(maxlen=self.history_length)
        self.time_history = deque(maxlen=self.history_length)
        self.homeostatic_history = deque(maxlen=self.history_length)
        
        super().__init__(notebook, "Pathos State", "💝")
        self._create_pathos_display()
    
    def _create_pathos_display(self):
        """Create the pathos visualization display."""
        # Create figure with subplots
        fig = Figure(figsize=(16, 12), facecolor='white')
        fig.suptitle('Pathos State Real-Time Visualization', fontsize=16, fontweight='bold')
        
        # Create subplots in a 2x3 grid for more information
        self.ax_state_norm = fig.add_subplot(2, 3, 1)
        self.ax_reward = fig.add_subplot(2, 3, 2)
        self.ax_heatmap = fig.add_subplot(2, 3, 3)
        self.ax_phase = fig.add_subplot(2, 3, 4)
        self.ax_homeostatic = fig.add_subplot(2, 3, 5)
        self.ax_statistics = fig.add_subplot(2, 3, 6)
        
        # Setup plots
        self._setup_plots()
        
        # Add matplotlib canvas
        self._add_matplotlib_canvas(fig)
    
    def _setup_plots(self):
        """Setup the visualization plots"""
        # Initial empty plots - will be populated when data arrives
        self.ax_state_norm.set_title('State Vector Norm Over Time')
        self.ax_state_norm.set_xlabel('Time Steps')
        self.ax_state_norm.set_ylabel('||F(t)||')
        
        self.ax_reward.set_title('Internal Reward Over Time')
        self.ax_reward.set_xlabel('Time Steps')
        self.ax_reward.set_ylabel('Internal Reward')
        
        self.ax_heatmap.set_title('State Vector Components')
        
        self.ax_phase.set_title('Phase Space (State Norm vs Reward)')
        self.ax_phase.set_xlabel('State Norm')
        self.ax_phase.set_ylabel('Internal Reward')
        
        self.ax_homeostatic.set_title('Homeostatic Balance')
        
        self.ax_statistics.set_title('Current Statistics')
        
        plt.tight_layout()
    
    def update_display(self):
        """Update pathos display with current data."""
        try:
            # Debug: Check collector status
            if not hasattr(self.collector, 'metrics'):
                self._show_no_data_message()
                return
            
            # Debug: Check for pathos trajectories
            if 'pathos_trajectories' not in self.collector.metrics:
                self._show_no_data_message()
                return
            
            trajectories = self.collector.metrics['pathos_trajectories']
            
            # Debug: Check if we have trajectory data
            if not trajectories:
                self._show_no_data_message()
                return
            
            # We have data! Process it
            latest = trajectories[-1]
            
            # Extract state information
            state_norm = latest.get('state_norm', 0.0)
            internal_reward = latest.get('internal_reward', 0.0)
            state_components = latest.get('state_components', [])
            homeostatic_balance = latest.get('homeostatic_balance', {})
            
            # Create state vector from available components
            if state_components:
                # Use actual state components and pad/truncate to match dimension
                state_vector = np.array(state_components)
                if len(state_vector) < self.state_dimension:
                    # Pad with zeros
                    padding = np.zeros(self.state_dimension - len(state_vector))
                    state_vector = np.concatenate([state_vector, padding])
                elif len(state_vector) > self.state_dimension:
                    # Truncate
                    state_vector = state_vector[:self.state_dimension]
            else:
                # Fallback: generate representative vector based on norm
                if state_norm > 0:
                    state_vector = np.random.normal(0, state_norm/10, self.state_dimension)
                    state_vector = state_vector * (state_norm / np.linalg.norm(state_vector))
                else:
                    state_vector = np.zeros(self.state_dimension)
            
            # Update with new data
            self._update_state(state_vector, internal_reward, homeostatic_balance)
                
        except Exception as e:
            self._show_error_message(f"Error updating pathos display: {str(e)}")
    
    def _update_state(self, state_vector: np.ndarray, internal_reward: float, homeostatic_balance: dict = None):
        """Update visualization with new state data"""
        # Store data
        self.state_history.append(state_vector.copy())
        self.reward_history.append(internal_reward)
        self.time_history.append(len(self.time_history))
        self.homeostatic_history.append(homeostatic_balance or {})
        
        # Update plots
        self._update_plots()
        
        # Refresh canvas
        if self.canvas:
            self.canvas.draw()
    
    def _update_homeostatic_plot(self):
        """Update homeostatic balance visualization."""
        self.ax_homeostatic.clear()
        self.ax_homeostatic.set_title('Homeostatic Balance')
        
        if not self.homeostatic_history:
            self.ax_homeostatic.text(0.5, 0.5, 'No homeostatic data', 
                                   ha='center', va='center', transform=self.ax_homeostatic.transAxes)
            return
        
        # Get the latest homeostatic balance
        latest_balance = self.homeostatic_history[-1]
        if not latest_balance:
            self.ax_homeostatic.text(0.5, 0.5, 'No homeostatic data', 
                                   ha='center', va='center', transform=self.ax_homeostatic.transAxes)
            return
        
        # Create bar chart of balance metrics
        metrics = []
        values = []
        colors = []
        
        for metric, value in latest_balance.items():
            if metric != 'total_discomfort':
                metrics.append(metric.capitalize())
                values.append(value)
                # Color code: green for low discomfort, red for high
                if value < 0.1:
                    colors.append('green')
                elif value < 0.5:
                    colors.append('orange')
                else:
                    colors.append('red')
        
        if metrics:
            bars = self.ax_homeostatic.bar(metrics, values, color=colors, alpha=0.7)
            self.ax_homeostatic.set_ylabel('Discomfort Level')
            self.ax_homeostatic.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                self.ax_homeostatic.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    def _update_statistics_plot(self):
        """Update current statistics display."""
        self.ax_statistics.clear()
        self.ax_statistics.set_title('Current Statistics')
        
        if not self.state_history or not self.reward_history:
            self.ax_statistics.text(0.5, 0.5, 'No data available', 
                                  ha='center', va='center', transform=self.ax_statistics.transAxes)
            return
        
        # Calculate statistics
        current_state = self.state_history[-1]
        recent_rewards = list(self.reward_history)[-10:]  # Last 10 rewards
        
        stats = {
            'Current State Norm': f'{np.linalg.norm(current_state):.3f}',
            'State Mean': f'{np.mean(current_state):.3f}',
            'State Std': f'{np.std(current_state):.3f}',
            'Recent Avg Reward': f'{np.mean(recent_rewards):.3f}',
            'Reward Trend': 'Positive' if len(recent_rewards) >= 2 and recent_rewards[-1] > recent_rewards[-2] else 'Negative',
            'Data Points': f'{len(self.state_history)}'
        }
        
        # Display as text
        y_pos = 0.9
        for key, value in stats.items():
            self.ax_statistics.text(0.05, y_pos, f'{key}: {value}', 
                                  transform=self.ax_statistics.transAxes, fontsize=10,
                                  verticalalignment='top')
            y_pos -= 0.15
        
        self.ax_statistics.set_xlim(0, 1)
        self.ax_statistics.set_ylim(0, 1)
        self.ax_statistics.axis('off')
    
    def _update_plots(self):
        """Update all visualization plots"""
        if len(self.state_history) < 2:
            return
        
        # Calculate state norms
        state_norms = [np.linalg.norm(state) for state in self.state_history]
        
        # Clear and update state norm plot
        self.ax_state_norm.clear()
        self.ax_state_norm.set_title('State Vector Norm Over Time')
        self.ax_state_norm.set_xlabel('Time Steps')
        self.ax_state_norm.set_ylabel('||F(t)||')
        self.ax_state_norm.plot(list(self.time_history), state_norms, 'b-', linewidth=2, label='State Norm')
        
        # Add trend line if we have enough data
        if len(state_norms) >= 5:
            # Simple moving average
            window = min(5, len(state_norms))
            moving_avg = []
            for i in range(len(state_norms)):
                start_idx = max(0, i - window + 1)
                avg = np.mean(state_norms[start_idx:i+1])
                moving_avg.append(avg)
            self.ax_state_norm.plot(list(self.time_history), moving_avg, 'g--', alpha=0.7, linewidth=1, label='Trend')
        
        self.ax_state_norm.grid(True, alpha=0.3)
        self.ax_state_norm.legend()
        
        # Clear and update reward plot
        self.ax_reward.clear()
        self.ax_reward.set_title('Internal Reward Over Time')
        self.ax_reward.set_xlabel('Time Steps')
        self.ax_reward.set_ylabel('Internal Reward')
        self.ax_reward.plot(list(self.time_history), list(self.reward_history), 'r-', linewidth=2, label='Internal Reward')
        
        # Add zero line for reference
        self.ax_reward.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        
        # Color positive and negative rewards differently
        rewards = list(self.reward_history)
        times = list(self.time_history)
        for i in range(len(rewards)):
            color = 'green' if rewards[i] > 0 else 'red'
            self.ax_reward.scatter(times[i], rewards[i], c=color, alpha=0.6, s=20)
        
        self.ax_reward.grid(True, alpha=0.3)
        self.ax_reward.legend()
        
        # Update state heatmap (PCA projection or raw components)
        if len(self.state_history) >= 2:
            states_matrix = np.array(list(self.state_history))
            
            self.ax_heatmap.clear()
            
            if SKLEARN_AVAILABLE and states_matrix.shape[1] > 2:
                # Use PCA if sklearn is available
                max_components = min(states_matrix.shape[0], states_matrix.shape[1])
                n_components = min(10, max_components)
                if n_components > 0:
                    pca = PCA(n_components=n_components)
                    states_pca = pca.fit_transform(states_matrix)
                    
                    self.ax_heatmap.set_title('State Vector Components (PCA Projection)')
                    im = self.ax_heatmap.imshow(states_pca[-20:].T, aspect='auto', cmap='coolwarm', interpolation='nearest')
                    self.ax_heatmap.set_xlabel('Time Steps (Recent 20)')
                    self.ax_heatmap.set_ylabel('PCA Components')
                else:
                    # Fallback to raw components
                    self.ax_heatmap.set_title('State Vector Components (Raw - First 10)')
                    im = self.ax_heatmap.imshow(states_matrix[-20:, :10].T, aspect='auto', cmap='coolwarm', interpolation='nearest')
                    self.ax_heatmap.set_xlabel('Time Steps (Recent 20)')
                    self.ax_heatmap.set_ylabel('State Dimensions (First 10)')
            else:
                # Use raw components when sklearn not available or low-dimensional data
                components_to_show = min(10, states_matrix.shape[1])
                self.ax_heatmap.set_title('State Vector Components (Raw)')
                im = self.ax_heatmap.imshow(states_matrix[-20:, :components_to_show].T, aspect='auto', cmap='coolwarm', interpolation='nearest')
                self.ax_heatmap.set_xlabel('Time Steps (Recent 20)')
                self.ax_heatmap.set_ylabel(f'State Dimensions (First {components_to_show})')
        
        # Update phase space plot
        if len(state_norms) >= 2:
            colors = np.arange(len(state_norms))
            self.ax_phase.clear()
            self.ax_phase.set_title('Phase Space (State Norm vs Reward)')
            self.ax_phase.set_xlabel('State Norm')
            self.ax_phase.set_ylabel('Internal Reward')
            scatter = self.ax_phase.scatter(state_norms, list(self.reward_history), 
                                          c=colors, cmap='viridis', alpha=0.6)
            
            # Add trajectory line
            self.ax_phase.plot(state_norms, list(self.reward_history), 'k-', alpha=0.3, linewidth=1)
        
        # Update homeostatic balance plot
        self._update_homeostatic_plot()
        
        # Update statistics plot
        self._update_statistics_plot()
    
    def _show_no_data_message(self):
        """Show message when no data is available."""
        for ax in [self.ax_state_norm, self.ax_reward, self.ax_heatmap, self.ax_phase, self.ax_homeostatic, self.ax_statistics]:
            ax.clear()
            ax.text(0.5, 0.5, 'No Pathos data available\n\nStart an agent session to see\nreal-time affective state visualization', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        
        if self.canvas:
            self.canvas.draw()
    
    def _show_error_message(self, error_msg: str):
        """Show error message."""
        for ax in [self.ax_state_norm, self.ax_reward, self.ax_heatmap, self.ax_phase, self.ax_homeostatic, self.ax_statistics]:
            ax.clear()
            ax.text(0.5, 0.5, f'Error: {error_msg}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10, color='red',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))
        
        if self.canvas:
            self.canvas.draw()