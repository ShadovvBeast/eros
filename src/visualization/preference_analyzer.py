"""
Preference Drift Analyzer

Analyzer for preference drift patterns and trends.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any, Optional

try:
    from ..core.logging_config import InstrumentationCollector
except ImportError:
    from core.logging_config import InstrumentationCollector


class PreferenceDriftAnalyzer:
    """Analyzer for preference drift patterns and trends"""
    
    def __init__(self):
        self.preference_data = defaultdict(list)
        self.drift_threshold = 0.3
    
    def analyze_drift_from_collector(self, collector: InstrumentationCollector) -> Dict[str, Any]:
        """Analyze preference drift from instrumentation collector"""
        if not collector.preference_history:
            return {'error': 'No preference data available'}
        
        analysis = {}
        
        for category, history in collector.preference_history.items():
            if len(history) < 3:
                continue
            
            timestamps, weights = zip(*history)
            weights = np.array(weights)
            
            # Calculate drift metrics
            drift_analysis = self._analyze_category_drift(category, weights, timestamps)
            analysis[category] = drift_analysis
        
        # Overall drift summary
        analysis['summary'] = self._calculate_overall_drift_summary(analysis)
        
        return analysis
    
    def _analyze_category_drift(self, category: str, weights: np.ndarray, timestamps: List[str]) -> Dict[str, Any]:
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
            rolling_std = pd.Series(weights).rolling(window=window_size).std().fillna(0)
            avg_volatility = float(np.mean(rolling_std))
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
    
    def _calculate_overall_drift_summary(self, category_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall drift summary across all categories"""
        if not category_analyses:
            return {}
        
        # Filter out summary key if it exists
        analyses = {k: v for k, v in category_analyses.items() if k != 'summary'}
        
        if not analyses:
            return {}
        
        total_categories = len(analyses)
        significant_drifts = sum(1 for analysis in analyses.values() if analysis.get('significant_drift', False))
        avg_stability = np.mean([analysis.get('stability_score', 0) for analysis in analyses.values()])
        most_volatile = max(analyses.items(), key=lambda x: x[1].get('avg_volatility', 0))
        
        return {
            'total_categories': total_categories,
            'categories_with_significant_drift': significant_drifts,
            'drift_percentage': (significant_drifts / total_categories) * 100,
            'average_stability_score': float(avg_stability),
            'most_volatile_category': most_volatile[0],
            'most_volatile_score': most_volatile[1].get('avg_volatility', 0)
        }
    
    def visualize_preference_drift(self, collector: InstrumentationCollector, save_path: Optional[str] = None):
        """Create visualization of preference drift patterns"""
        if not collector.preference_history:
            print("No preference data available for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Preference Drift Analysis')
        
        # Plot 1: Preference weights over time
        ax1 = axes[0, 0]
        ax1.set_title('Preference Weights Over Time')
        ax1.set_xlabel('Update Number')
        ax1.set_ylabel('Weight Value')
        
        for category, history in collector.preference_history.items():
            if len(history) > 1:
                _, weights = zip(*history)
                ax1.plot(range(len(weights)), weights, marker='o', label=category, linewidth=2)
        
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Drift magnitude distribution
        ax2 = axes[0, 1]
        ax2.set_title('Drift Magnitude Distribution')
        
        drift_magnitudes = []
        for category, history in collector.preference_history.items():
            if len(history) > 1:
                _, weights = zip(*history)
                weights = np.array(weights)
                drift_magnitudes.append(np.max(weights) - np.min(weights))
        
        if drift_magnitudes:
            ax2.hist(drift_magnitudes, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(self.drift_threshold, color='red', linestyle='--', label=f'Threshold ({self.drift_threshold})')
            ax2.legend()
        
        ax2.set_xlabel('Total Drift Magnitude')
        ax2.set_ylabel('Frequency')
        
        # Plot 3: Volatility analysis
        ax3 = axes[1, 0]
        ax3.set_title('Preference Volatility by Category')
        
        categories = []
        volatilities = []
        
        for category, history in collector.preference_history.items():
            if len(history) > 2:
                _, weights = zip(*history)
                weights = np.array(weights)
                changes = np.abs(np.diff(weights))
                volatility = np.std(changes)
                categories.append(category)
                volatilities.append(volatility)
        
        if categories:
            bars = ax3.bar(categories, volatilities, color='lightcoral', alpha=0.7)
            ax3.set_ylabel('Volatility (Std of Changes)')
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # Plot 4: Stability scores
        ax4 = axes[1, 1]
        ax4.set_title('Category Stability Scores')
        
        analysis = self.analyze_drift_from_collector(collector)
        if 'summary' not in analysis:
            stability_categories = []
            stability_scores = []
            
            for category, cat_analysis in analysis.items():
                if isinstance(cat_analysis, dict) and 'stability_score' in cat_analysis:
                    stability_categories.append(category)
                    stability_scores.append(cat_analysis['stability_score'])
            
            if stability_categories:
                colors = ['green' if score > 0.7 else 'orange' if score > 0.4 else 'red' for score in stability_scores]
                bars = ax4.bar(stability_categories, stability_scores, color=colors, alpha=0.7)
                ax4.set_ylabel('Stability Score')
                ax4.set_ylim(0, 1)
                plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Preference drift analysis saved to: {save_path}")
        
        # Close the figure to free memory instead of showing
        plt.close(fig)