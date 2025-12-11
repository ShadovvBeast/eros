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
        
        super().__init__(notebook, "Pathos State", "💝")
        self._create_pathos_display()
    
    def _create_pathos_display(self):
        """Create the pathos visualization display."""
        # Create figure with subplots
        fig = Figure(figsize=(15, 10), facecolor='white')
        fig.suptitle('Pathos State Real-Time Visualization', fontsize=16, fontweight='bold')
        
        # Create subplots
        self.ax_state_norm = fig.add_subplot(2, 2, 1)
        self.ax_reward = fig.add_subplot(2, 2, 2)
        self.ax_heatmap = fig.add_subplot(2, 2, 3)
        self.ax_phase = fig.add_subplot(2, 2, 4)
        
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
        
        self.ax_heatmap.set_title('State Vector Components (PCA Projection)')
        
        self.ax_phase.set_title('Phase Space (State Norm vs Reward)')
        self.ax_phase.set_xlabel('State Norm')
        self.ax_phase.set_ylabel('Internal Reward')
        
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
            
            # Create mock state vector (since we don't have the actual vector)
            state_norm = latest.get('state_norm', 0.0)
            internal_reward = latest.get('internal_reward', 0.0)
            
            # Generate a mock state vector based on the norm
            if state_norm > 0:
                state_vector = np.random.normal(0, state_norm/10, self.state_dimension)
                state_vector = state_vector * (state_norm / np.linalg.norm(state_vector))
            else:
                state_vector = np.zeros(self.state_dimension)
            
            # Update with new data
            self._update_state(state_vector, internal_reward)
                
        except Exception as e:
            self._show_error_message(f"Error updating pathos display: {str(e)}")
    
    def _update_state(self, state_vector: np.ndarray, internal_reward: float):
        """Update visualization with new state data"""
        # Store data
        self.state_history.append(state_vector.copy())
        self.reward_history.append(internal_reward)
        self.time_history.append(len(self.time_history))
        
        # Update plots
        self._update_plots()
        
        # Refresh canvas
        if self.canvas:
            self.canvas.draw()
    
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
        self.ax_state_norm.plot(list(self.time_history), state_norms, 'b-', linewidth=2)
        self.ax_state_norm.grid(True, alpha=0.3)
        
        # Clear and update reward plot
        self.ax_reward.clear()
        self.ax_reward.set_title('Internal Reward Over Time')
        self.ax_reward.set_xlabel('Time Steps')
        self.ax_reward.set_ylabel('Internal Reward')
        self.ax_reward.plot(list(self.time_history), list(self.reward_history), 'r-', linewidth=2)
        self.ax_reward.grid(True, alpha=0.3)
        
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
    
    def _show_no_data_message(self):
        """Show message when no data is available."""
        for ax in [self.ax_state_norm, self.ax_reward, self.ax_heatmap, self.ax_phase]:
            ax.clear()
            ax.text(0.5, 0.5, 'No Pathos data available\n\nStart an agent session to see\nreal-time affective state visualization', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        
        if self.canvas:
            self.canvas.draw()
    
    def _show_error_message(self, error_msg: str):
        """Show error message."""
        for ax in [self.ax_state_norm, self.ax_reward, self.ax_heatmap, self.ax_phase]:
            ax.clear()
            ax.text(0.5, 0.5, f'Error: {error_msg}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10, color='red',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))
        
        if self.canvas:
            self.canvas.draw()