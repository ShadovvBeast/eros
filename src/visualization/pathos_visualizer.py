"""
Pathos State Visualizer

Real-time visualization of Pathos affective state dynamics.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from collections import deque
from typing import Optional
from datetime import datetime
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class PathosStateVisualizer:
    """Real-time visualization of Pathos affective state dynamics"""
    
    def __init__(self, state_dimension: int = 128, history_length: int = 100):
        self.state_dimension = state_dimension
        self.history_length = history_length
        self.state_history = deque(maxlen=history_length)
        self.reward_history = deque(maxlen=history_length)
        self.time_history = deque(maxlen=history_length)
        
        # Setup matplotlib for non-interactive plotting
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Pathos State Real-Time Visualization')
        
        # Initialize plots
        self._setup_plots()
    
    def _setup_plots(self):
        """Setup the visualization plots"""
        # State norm over time
        self.axes[0, 0].set_title('State Vector Norm Over Time')
        self.axes[0, 0].set_xlabel('Time Steps')
        self.axes[0, 0].set_ylabel('||F(t)||')
        self.state_norm_line, = self.axes[0, 0].plot([], [], 'b-', linewidth=2)
        
        # Internal reward over time
        self.axes[0, 1].set_title('Internal Reward Over Time')
        self.axes[0, 1].set_xlabel('Time Steps')
        self.axes[0, 1].set_ylabel('Internal Reward')
        self.reward_line, = self.axes[0, 1].plot([], [], 'r-', linewidth=2)
        
        # State vector heatmap (2D projection)
        self.axes[1, 0].set_title('State Vector Components (PCA Projection)')
        self.state_heatmap = None
        
        # Phase space plot (state norm vs reward)
        self.axes[1, 1].set_title('Phase Space (State Norm vs Reward)')
        self.axes[1, 1].set_xlabel('State Norm')
        self.axes[1, 1].set_ylabel('Internal Reward')
        self.phase_scatter = self.axes[1, 1].scatter([], [], c=[], cmap='viridis', alpha=0.6)
        
        plt.tight_layout()
    
    def update_state(self, state_vector: np.ndarray, internal_reward: float, timestamp: Optional[str] = None):
        """Update visualization with new state data"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Store data
        self.state_history.append(state_vector.copy())
        self.reward_history.append(internal_reward)
        self.time_history.append(len(self.time_history))
        
        # Update plots
        self._update_plots()
    
    def _update_plots(self):
        """Update all visualization plots"""
        if len(self.state_history) < 2:
            return
        
        # Calculate state norms
        state_norms = [np.linalg.norm(state) for state in self.state_history]
        
        # Update state norm plot
        self.state_norm_line.set_data(list(self.time_history), state_norms)
        self.axes[0, 0].relim()
        self.axes[0, 0].autoscale_view()
        
        # Update reward plot
        self.reward_line.set_data(list(self.time_history), list(self.reward_history))
        self.axes[0, 1].relim()
        self.axes[0, 1].autoscale_view()
        
        # Update state heatmap (PCA projection)
        if len(self.state_history) >= 2:
            states_matrix = np.array(list(self.state_history))
            if states_matrix.shape[1] > 2:
                # Ensure n_components doesn't exceed min(n_samples, n_features)
                max_components = min(states_matrix.shape[0], states_matrix.shape[1])
                n_components = min(10, max_components)
                if n_components > 0:
                    pca = PCA(n_components=n_components)
                    states_pca = pca.fit_transform(states_matrix)
                    
                    self.axes[1, 0].clear()
                    self.axes[1, 0].set_title('State Vector Components (PCA Projection)')
                    im = self.axes[1, 0].imshow(states_pca[-20:].T, aspect='auto', cmap='coolwarm', interpolation='nearest')
                    self.axes[1, 0].set_xlabel('Time Steps (Recent 20)')
                    self.axes[1, 0].set_ylabel('PCA Components')
                else:
                    # Fallback: show raw state components if PCA not possible
                    self.axes[1, 0].clear()
                    self.axes[1, 0].set_title('State Vector Components (Raw)')
                    im = self.axes[1, 0].imshow(states_matrix[-20:, :10].T, aspect='auto', cmap='coolwarm', interpolation='nearest')
                    self.axes[1, 0].set_xlabel('Time Steps (Recent 20)')
                    self.axes[1, 0].set_ylabel('State Dimensions (First 10)')
            else:
                # For low-dimensional states, show directly
                self.axes[1, 0].clear()
                self.axes[1, 0].set_title('State Vector Components')
                im = self.axes[1, 0].imshow(states_matrix[-20:].T, aspect='auto', cmap='coolwarm', interpolation='nearest')
                self.axes[1, 0].set_xlabel('Time Steps (Recent 20)')
                self.axes[1, 0].set_ylabel('State Dimensions')
        
        # Update phase space plot
        if len(state_norms) >= 2:
            colors = np.arange(len(state_norms))
            self.axes[1, 1].clear()
            self.axes[1, 1].set_title('Phase Space (State Norm vs Reward)')
            self.axes[1, 1].set_xlabel('State Norm')
            self.axes[1, 1].set_ylabel('Internal Reward')
            scatter = self.axes[1, 1].scatter(state_norms, list(self.reward_history), 
                                            c=colors, cmap='viridis', alpha=0.6)
            
            # Add trajectory line
            self.axes[1, 1].plot(state_norms, list(self.reward_history), 'k-', alpha=0.3, linewidth=1)
    
    def save_visualization(self, filepath: str):
        """Save current visualization to file"""
        self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
    
    def close(self):
        """Close the visualization"""
        plt.close(self.fig)