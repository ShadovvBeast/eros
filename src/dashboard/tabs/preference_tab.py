"""
Preference Drift Tab

Preference evolution and drift analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from collections import defaultdict

# Optional pandas import
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from .base_tab import BaseTab


class PreferenceTab(BaseTab):
    """Preference drift analysis tab."""
    
    def __init__(self, notebook, collector):
        """Initialize preference tab."""
        self.collector = collector
        self.drift_threshold = 0.3
        super().__init__(notebook, "Preference Drift", "📊")
        self._create_preference_display()
    
    def _create_preference_display(self):
        """Create the preference analysis display."""
        # Create figure with subplots
        fig = Figure(figsize=(15, 10), facecolor='white')
        fig.suptitle('Preference Drift Analysis', fontsize=16, fontweight='bold')
        
        # Create subplots
        self.ax_weights = fig.add_subplot(2, 2, 1)
        self.ax_drift = fig.add_subplot(2, 2, 2)
        self.ax_volatility = fig.add_subplot(2, 2, 3)
        self.ax_stability = fig.add_subplot(2, 2, 4)
        
        # Add matplotlib canvas
        self._add_matplotlib_canvas(fig)
    
    def update_display(self):
        """Update preference display with current data."""
        try:
            # Check if we have preference data
            if not hasattr(self.collector, 'preference_history') or not self.collector.preference_history:
                self._show_no_data_message()
                return
            
            # Clear all axes
            for ax in [self.ax_weights, self.ax_drift, self.ax_volatility, self.ax_stability]:
                ax.clear()
            
            # Plot 1: Preference weights over time
            self.ax_weights.set_title('Preference Weights Over Time')
            self.ax_weights.set_xlabel('Update Number')
            self.ax_weights.set_ylabel('Weight Value')
            
            for category, history in self.collector.preference_history.items():
                if len(history) > 1:
                    _, weights = zip(*history)
                    self.ax_weights.plot(range(len(weights)), weights, marker='o', label=category, linewidth=2)
            
            self.ax_weights.legend()
            self.ax_weights.grid(True, alpha=0.3)
            
            # Plot 2: Drift magnitude distribution
            self.ax_drift.set_title('Drift Magnitude Distribution')
            
            drift_magnitudes = []
            for category, history in self.collector.preference_history.items():
                if len(history) > 1:
                    _, weights = zip(*history)
                    weights = np.array(weights)
                    drift_magnitudes.append(np.max(weights) - np.min(weights))
            
            if drift_magnitudes:
                self.ax_drift.hist(drift_magnitudes, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
                self.ax_drift.axvline(self.drift_threshold, color='red', linestyle='--', 
                                     label=f'Threshold ({self.drift_threshold})')
                self.ax_drift.legend()
            
            self.ax_drift.set_xlabel('Total Drift Magnitude')
            self.ax_drift.set_ylabel('Frequency')
            
            # Plot 3: Volatility analysis
            self.ax_volatility.set_title('Preference Volatility by Category')
            
            categories = []
            volatilities = []
            
            for category, history in self.collector.preference_history.items():
                if len(history) > 2:
                    _, weights = zip(*history)
                    weights = np.array(weights)
                    changes = np.abs(np.diff(weights))
                    volatility = np.std(changes)
                    categories.append(category)
                    volatilities.append(volatility)
            
            if categories:
                bars = self.ax_volatility.bar(categories, volatilities, color='lightcoral', alpha=0.7)
                self.ax_volatility.set_ylabel('Volatility (Std of Changes)')
                plt.setp(self.ax_volatility.get_xticklabels(), rotation=45, ha='right')
            
            # Plot 4: Stability scores
            self.ax_stability.set_title('Category Stability Scores')
            
            analysis = self._analyze_drift_from_collector()
            if analysis and 'summary' not in analysis:
                stability_categories = []
                stability_scores = []
                
                for category, cat_analysis in analysis.items():
                    if isinstance(cat_analysis, dict) and 'stability_score' in cat_analysis:
                        stability_categories.append(category)
                        stability_scores.append(cat_analysis['stability_score'])
                
                if stability_categories:
                    colors = ['green' if score > 0.7 else 'orange' if score > 0.4 else 'red' 
                             for score in stability_scores]
                    bars = self.ax_stability.bar(stability_categories, stability_scores, color=colors, alpha=0.7)
                    self.ax_stability.set_ylabel('Stability Score')
                    self.ax_stability.set_ylim(0, 1)
                    plt.setp(self.ax_stability.get_xticklabels(), rotation=45, ha='right')
            
            # Refresh canvas
            if self.canvas:
                self.canvas.draw()
                
        except Exception as e:
            self._show_error_message(f"Error updating preference display: {str(e)}")
    
    def _analyze_drift_from_collector(self):
        """Analyze preference drift from instrumentation collector"""
        if not self.collector.preference_history:
            return None
        
        analysis = {}
        
        for category, history in self.collector.preference_history.items():
            if len(history) < 3:
                continue
            
            timestamps, weights = zip(*history)
            weights = np.array(weights)
            
            # Calculate drift metrics
            drift_analysis = self._analyze_category_drift(category, weights, timestamps)
            analysis[category] = drift_analysis
        
        return analysis
    
    def _analyze_category_drift(self, category: str, weights: np.ndarray, timestamps):
        """Analyze drift for a specific category"""
        # Basic statistics
        mean_weight = float(np.mean(weights))
        std_weight = float(np.std(weights))
        min_weight = float(np.min(weights))
        max_weight = float(np.max(weights))
        
        # Trend analysis (linear regression slope)
        x = np.arange(len(weights))
        slope = np.polyfit(x, weights, 1)[0] if len(weights) > 1 else 0.0
        
        # Volatility (rolling standard deviation)
        window_size = min(5, len(weights))
        if window_size > 1:
            if PANDAS_AVAILABLE:
                rolling_std = pd.Series(weights).rolling(window=window_size).std().fillna(0)
                avg_volatility = float(np.mean(rolling_std))
            else:
                # Simple volatility calculation without pandas
                changes = np.abs(np.diff(weights))
                avg_volatility = float(np.std(changes)) if len(changes) > 0 else 0.0
        else:
            avg_volatility = 0.0
        
        # Drift detection
        total_drift = max_weight - min_weight
        significant_drift = total_drift > self.drift_threshold
        
        # Change points detection (simple threshold-based)
        changes = np.abs(np.diff(weights))
        major_changes = np.where(changes > 0.2)[0]
        
        return {
            'mean_weight': mean_weight,
            'std_weight': std_weight,
            'min_weight': min_weight,
            'max_weight': max_weight,
            'total_drift': total_drift,
            'trend_slope': float(slope),
            'avg_volatility': avg_volatility,
            'significant_drift': significant_drift,
            'major_change_points': len(major_changes),
            'stability_score': 1.0 - (avg_volatility + abs(slope)) / 2.0
        }
    
    def _show_no_data_message(self):
        """Show message when no data is available."""
        for ax in [self.ax_weights, self.ax_drift, self.ax_volatility, self.ax_stability]:
            ax.clear()
            ax.text(0.5, 0.5, 'No preference data available\n\nStart an agent session to see\npreference evolution and drift analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
        
        if self.canvas:
            self.canvas.draw()
    
    def _show_error_message(self, error_msg: str):
        """Show error message."""
        for ax in [self.ax_weights, self.ax_drift, self.ax_volatility, self.ax_stability]:
            ax.clear()
            ax.text(0.5, 0.5, f'Error: {error_msg}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10, color='red',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))
        
        if self.canvas:
            self.canvas.draw()