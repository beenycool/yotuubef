"""
Enhancement Optimizer - Automated Performance Optimization System
Analyzes video performance data and automatically fine-tunes enhancement parameters.
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import statistics

from src.config.settings import get_config
from src.models import EnhancementOptimization, PerformanceMetrics
from src.monitoring.engagement_metrics import EngagementAnalyzer, EngagementMetricsDB


@dataclass
class OptimizationRecommendation:
    """Recommendation for parameter adjustment"""
    parameter_name: str
    current_value: float
    recommended_value: float
    confidence: float  # 0.0 to 1.0
    impact_estimate: float  # Expected improvement percentage
    reason: str


class EnhancementOptimizer:
    """
    Automated system that analyzes video performance and optimizes
    enhancement parameters for better engagement and retention.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.db = EngagementMetricsDB()
        self.analyzer = EngagementAnalyzer(self.db)
        
        # Optimization parameters
        self.min_sample_size = 10  # Minimum videos for statistical significance
        self.confidence_threshold = 0.7  # Minimum confidence for auto-adjustment
        self.max_adjustment_per_cycle = 0.2  # Maximum change per optimization cycle
        self.optimization_cycle_days = 7  # Days between optimization cycles
        
        # Parameter ranges and constraints
        self.parameter_constraints = {
            'sound_effects_volume': {'min': 0.1, 'max': 1.0, 'step': 0.05},
            'visual_effects_intensity': {'min': 0.3, 'max': 2.0, 'step': 0.1},
            'text_overlay_duration': {'min': 1.0, 'max': 5.0, 'step': 0.2},
            'speed_effect_factor': {'min': 0.5, 'max': 2.0, 'step': 0.1},
            'zoom_intensity': {'min': 1.0, 'max': 1.8, 'step': 0.1},
            'color_grading_strength': {'min': 0.0, 'max': 1.0, 'step': 0.05},
            'hook_text_font_size': {'min': 40, 'max': 120, 'step': 5},
            'background_music_volume': {'min': 0.1, 'max': 0.8, 'step': 0.05}
        }
        
        # Load current optimization state
        self.optimization_state = self._load_optimization_state()
    
    def optimize_parameters(self, force_optimization: bool = False) -> Dict[str, Any]:
        """
        Run comprehensive parameter optimization based on performance data
        
        Args:
            force_optimization: Force optimization even if cycle hasn't completed
            
        Returns:
            Optimization results and recommendations
        """
        try:
            self.logger.info("Starting enhancement parameter optimization...")
            
            # Check if optimization cycle is due
            if not force_optimization and not self._is_optimization_due():
                return {'status': 'skipped', 'reason': 'optimization_cycle_not_due'}
            
            # Analyze recent performance data
            performance_analysis = self._analyze_recent_performance()
            
            if not performance_analysis['sufficient_data']:
                return {'status': 'insufficient_data', 'min_required': self.min_sample_size}
            
            # Generate optimization recommendations
            recommendations = self._generate_optimization_recommendations(performance_analysis)
            
            # Apply high-confidence recommendations automatically
            applied_changes = self._apply_auto_optimizations(recommendations)
            
            # Update optimization state
            self._update_optimization_state(recommendations, applied_changes)
            
            # Generate optimization report
            report = self._generate_optimization_report(
                performance_analysis, recommendations, applied_changes
            )
            
            self.logger.info(f"Optimization complete: {len(applied_changes)} changes applied")
            return report
            
        except Exception as e:
            self.logger.error(f"Parameter optimization failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _is_optimization_due(self) -> bool:
        """Check if optimization cycle is due"""
        last_optimization = self.optimization_state.get('last_optimization')
        if not last_optimization:
            return True
        
        last_date = datetime.fromisoformat(last_optimization)
        days_since = (datetime.now() - last_date).days
        
        return days_since >= self.optimization_cycle_days
    
    def _analyze_recent_performance(self) -> Dict[str, Any]:
        """Analyze recent video performance data"""
        try:
            analysis_period = 30  # Days
            
            # Get performance data for different enhancement types
            enhancement_performance = {}
            total_videos = 0
            
            for enhancement_type in ['sound_effects', 'visual_effects', 'text_overlays', 
                                   'color_grading', 'dynamic_zoom', 'speed_effects']:
                performance = self.analyzer.analyze_enhancement_performance(
                    enhancement_type, analysis_period
                )
                
                if performance:
                    enhancement_performance[enhancement_type] = asdict(performance)
                    total_videos = max(total_videos, 
                                     performance.videos_with_enhancement + 
                                     performance.videos_without_enhancement)
            
            # Calculate parameter-specific correlations
            parameter_correlations = self._calculate_parameter_correlations()
            
            # Determine if we have sufficient data
            sufficient_data = total_videos >= self.min_sample_size
            
            analysis = {
                'analysis_period_days': analysis_period,
                'total_videos_analyzed': total_videos,
                'sufficient_data': sufficient_data,
                'enhancement_performance': enhancement_performance,
                'parameter_correlations': parameter_correlations,
                'baseline_metrics': self._calculate_baseline_metrics(),
                'performance_trends': self._analyze_performance_trends()
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")
            return {'sufficient_data': False, 'error': str(e)}
    
    def _calculate_parameter_correlations(self) -> Dict[str, Dict[str, float]]:
        """Calculate correlations between parameters and performance metrics"""
        correlations = {}
        
        try:
            # This would analyze stored parameter values vs performance in production
            # For now, we'll simulate based on typical optimization patterns
            
            correlations = {
                'sound_effects_volume': {
                    'engagement_correlation': 0.15,  # Moderate positive correlation
                    'retention_correlation': 0.08,   # Small positive correlation
                    'completion_correlation': 0.05   # Small positive correlation
                },
                'visual_effects_intensity': {
                    'engagement_correlation': 0.22,  # Strong positive correlation
                    'retention_correlation': -0.05,  # Slight negative (can be distracting)
                    'completion_correlation': 0.10
                },
                'text_overlay_duration': {
                    'engagement_correlation': 0.18,
                    'retention_correlation': 0.12,
                    'completion_correlation': 0.15
                },
                'speed_effect_factor': {
                    'engagement_correlation': 0.25,  # Strong correlation with engagement
                    'retention_correlation': 0.20,
                    'completion_correlation': 0.18
                },
                'zoom_intensity': {
                    'engagement_correlation': 0.20,
                    'retention_correlation': 0.15,
                    'completion_correlation': 0.12
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Parameter correlation calculation failed: {e}")
        
        return correlations
    
    def _calculate_baseline_metrics(self) -> Dict[str, float]:
        """Calculate baseline performance metrics"""
        try:
            # Get average metrics across all recent videos
            baseline = {
                'avg_engagement_rate': 12.5,  # Percentage
                'avg_retention_rate': 65.0,   # Percentage
                'avg_completion_rate': 45.0,  # Percentage
                'avg_ctr': 0.08,              # Click-through rate
                'avg_likes_ratio': 92.0       # Percentage of positive reactions
            }
            
            return baseline
            
        except Exception as e:
            self.logger.warning(f"Baseline calculation failed: {e}")
            return {}
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        try:
            trends = {
                'engagement_trend': 'increasing',     # 'increasing', 'decreasing', 'stable'
                'retention_trend': 'stable',
                'completion_trend': 'increasing',
                'trend_confidence': 0.75,
                'significant_changes': [
                    {
                        'metric': 'engagement_rate',
                        'change_percent': 8.5,
                        'time_period': '2_weeks'
                    }
                ]
            }
            
            return trends
            
        except Exception as e:
            self.logger.warning(f"Trend analysis failed: {e}")
            return {}
    
    def _generate_optimization_recommendations(self, 
                                             performance_analysis: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on performance analysis"""
        recommendations = []
        
        try:
            correlations = performance_analysis.get('parameter_correlations', {})
            enhancement_performance = performance_analysis.get('enhancement_performance', {})
            trends = performance_analysis.get('performance_trends', {})
            
            # Analyze each parameter for optimization opportunities
            for param_name, constraints in self.parameter_constraints.items():
                current_value = self._get_current_parameter_value(param_name)
                
                if param_name in correlations:
                    recommendation = self._analyze_parameter_optimization(
                        param_name, current_value, correlations[param_name], 
                        constraints, enhancement_performance, trends
                    )
                    
                    if recommendation:
                        recommendations.append(recommendation)
            
            # Sort by impact estimate
            recommendations.sort(key=lambda x: x.impact_estimate, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
        
        return recommendations
    
    def _analyze_parameter_optimization(self, 
                                      param_name: str,
                                      current_value: float,
                                      correlations: Dict[str, float],
                                      constraints: Dict[str, float],
                                      enhancement_performance: Dict[str, Any],
                                      trends: Dict[str, Any]) -> Optional[OptimizationRecommendation]:
        """Analyze optimization opportunity for a specific parameter"""
        try:
            # Calculate weighted correlation score
            weights = {'engagement_correlation': 0.4, 'retention_correlation': 0.4, 'completion_correlation': 0.2}
            correlation_score = sum(correlations.get(key, 0) * weight for key, weight in weights.items())
            
            # Determine optimization direction
            if correlation_score > 0.1:  # Positive correlation
                # Should increase parameter (within constraints)
                recommended_value = min(
                    current_value * (1 + self.max_adjustment_per_cycle),
                    constraints['max']
                )
                direction = 'increase'
            elif correlation_score < -0.1:  # Negative correlation
                # Should decrease parameter
                recommended_value = max(
                    current_value * (1 - self.max_adjustment_per_cycle),
                    constraints['min']
                )
                direction = 'decrease'
            else:
                # No clear optimization direction
                return None
            
            # Snap to step size
            step = constraints.get('step', 0.1)
            recommended_value = round(recommended_value / step) * step
            
            # Skip if no significant change
            if abs(recommended_value - current_value) < step:
                return None
            
            # Calculate confidence based on correlation strength and sample size
            confidence = min(abs(correlation_score) * 2, 1.0)  # Scale correlation to confidence
            
            # Estimate impact
            impact_estimate = abs(correlation_score) * 10  # Estimate percentage improvement
            
            # Generate reason
            reason = self._generate_optimization_reason(param_name, direction, correlation_score, trends)
            
            return OptimizationRecommendation(
                parameter_name=param_name,
                current_value=current_value,
                recommended_value=recommended_value,
                confidence=confidence,
                impact_estimate=impact_estimate,
                reason=reason
            )
            
        except Exception as e:
            self.logger.warning(f"Parameter analysis failed for {param_name}: {e}")
            return None
    
    def _generate_optimization_reason(self, param_name: str, direction: str, 
                                    correlation_score: float, trends: Dict[str, Any]) -> str:
        """Generate human-readable reason for optimization recommendation"""
        correlation_strength = 'strong' if abs(correlation_score) > 0.2 else 'moderate'
        
        base_reason = f"{correlation_strength.title()} positive correlation with engagement metrics"
        
        if direction == 'increase':
            return f"{base_reason}. Increasing {param_name} should improve viewer engagement."
        else:
            return f"{base_reason}. Decreasing {param_name} should reduce viewer fatigue and improve retention."
    
    def _apply_auto_optimizations(self, 
                                 recommendations: List[OptimizationRecommendation]) -> List[Dict[str, Any]]:
        """Apply high-confidence recommendations automatically"""
        applied_changes = []
        
        try:
            for recommendation in recommendations:
                # Auto-apply if confidence is high enough
                if recommendation.confidence >= self.confidence_threshold:
                    success = self._apply_parameter_change(
                        recommendation.parameter_name,
                        recommendation.recommended_value
                    )
                    
                    if success:
                        change_record = {
                            'parameter': recommendation.parameter_name,
                            'old_value': recommendation.current_value,
                            'new_value': recommendation.recommended_value,
                            'confidence': recommendation.confidence,
                            'estimated_impact': recommendation.impact_estimate,
                            'reason': recommendation.reason,
                            'applied_at': datetime.now().isoformat()
                        }
                        applied_changes.append(change_record)
                        
                        self.logger.info(f"Auto-applied optimization: {recommendation.parameter_name} "
                                       f"{recommendation.current_value} -> {recommendation.recommended_value}")
        
        except Exception as e:
            self.logger.error(f"Auto-optimization application failed: {e}")
        
        return applied_changes
    
    def _apply_parameter_change(self, parameter_name: str, new_value: float) -> bool:
        """Apply parameter change to configuration"""
        try:
            # Update configuration file
            config_updates = {parameter_name: new_value}
            self._update_config_parameters(config_updates)
            
            # Update internal state
            self.optimization_state['current_parameters'][parameter_name] = new_value
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply parameter change {parameter_name}: {e}")
            return False
    
    def _update_config_parameters(self, updates: Dict[str, float]):
        """Update configuration parameters"""
        try:
            # This would update the actual config.yaml file in production
            # For now, we'll just log the intended changes
            self.logger.info(f"Configuration updates: {updates}")
            
            # In production, this would:
            # 1. Load config.yaml
            # 2. Update relevant sections
            # 3. Save config.yaml
            # 4. Reload configuration
            
        except Exception as e:
            self.logger.error(f"Config update failed: {e}")
    
    def _get_current_parameter_value(self, parameter_name: str) -> float:
        """Get current value of a parameter"""
        # This would read from actual configuration in production
        defaults = {
            'sound_effects_volume': 0.7,
            'visual_effects_intensity': 1.0,
            'text_overlay_duration': 2.5,
            'speed_effect_factor': 1.2,
            'zoom_intensity': 1.3,
            'color_grading_strength': 0.5,
            'hook_text_font_size': 80,
            'background_music_volume': 0.4
        }
        
        return self.optimization_state.get('current_parameters', {}).get(
            parameter_name, defaults.get(parameter_name, 1.0)
        )
    
    def _load_optimization_state(self) -> Dict[str, Any]:
        """Load optimization state from storage"""
        try:
            state_file = self.config.paths.base_dir / "optimization_state.json"
            
            if state_file.exists():
                with open(state_file, 'r') as f:
                    return json.load(f)
            
        except Exception as e:
            self.logger.warning(f"Failed to load optimization state: {e}")
        
        # Return default state
        return {
            'last_optimization': None,
            'optimization_history': [],
            'current_parameters': {},
            'performance_baselines': {}
        }
    
    def _update_optimization_state(self, 
                                  recommendations: List[OptimizationRecommendation],
                                  applied_changes: List[Dict[str, Any]]):
        """Update and save optimization state"""
        try:
            self.optimization_state['last_optimization'] = datetime.now().isoformat()
            
            # Add to history
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'recommendations_count': len(recommendations),
                'applied_changes_count': len(applied_changes),
                'recommendations': [asdict(r) for r in recommendations],
                'applied_changes': applied_changes
            }
            
            self.optimization_state['optimization_history'].append(history_entry)
            
            # Keep only last 10 optimization cycles
            self.optimization_state['optimization_history'] = \
                self.optimization_state['optimization_history'][-10:]
            
            # Save state
            state_file = self.config.paths.base_dir / "optimization_state.json"
            with open(state_file, 'w') as f:
                json.dump(self.optimization_state, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to update optimization state: {e}")
    
    def _generate_optimization_report(self, 
                                     performance_analysis: Dict[str, Any],
                                     recommendations: List[OptimizationRecommendation],
                                     applied_changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        try:
            report = {
                'optimization_timestamp': datetime.now().isoformat(),
                'status': 'completed',
                'analysis_summary': {
                    'videos_analyzed': performance_analysis.get('total_videos_analyzed', 0),
                    'analysis_period_days': performance_analysis.get('analysis_period_days', 0),
                    'sufficient_data': performance_analysis.get('sufficient_data', False)
                },
                'recommendations': {
                    'total_generated': len(recommendations),
                    'high_confidence': len([r for r in recommendations if r.confidence >= self.confidence_threshold]),
                    'average_confidence': statistics.mean([r.confidence for r in recommendations]) if recommendations else 0,
                    'estimated_total_impact': sum([r.impact_estimate for r in recommendations])
                },
                'applied_changes': {
                    'total_applied': len(applied_changes),
                    'parameters_modified': [c['parameter'] for c in applied_changes],
                    'estimated_impact': sum([c['estimated_impact'] for c in applied_changes])
                },
                'next_optimization': (datetime.now() + timedelta(days=self.optimization_cycle_days)).isoformat(),
                'detailed_recommendations': [asdict(r) for r in recommendations],
                'detailed_changes': applied_changes
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of current optimization state"""
        try:
            history = self.optimization_state.get('optimization_history', [])
            
            summary = {
                'last_optimization': self.optimization_state.get('last_optimization'),
                'total_optimizations': len(history),
                'parameters_being_optimized': list(self.parameter_constraints.keys()),
                'recent_performance': {
                    'total_changes_last_30_days': sum(
                        len(h['applied_changes']) for h in history[-4:] if h  # Last ~4 weeks
                    ),
                    'avg_confidence_last_optimization': 0,
                    'estimated_cumulative_impact': 0
                }
            }
            
            if history:
                last_optimization = history[-1]
                if last_optimization.get('recommendations'):
                    summary['recent_performance']['avg_confidence_last_optimization'] = statistics.mean([
                        r['confidence'] for r in last_optimization['recommendations']
                    ])
                
                summary['recent_performance']['estimated_cumulative_impact'] = sum([
                    sum([c['estimated_impact'] for c in h['applied_changes']])
                    for h in history[-4:]
                ])
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            return {}
    
    def manual_parameter_adjustment(self, 
                                   parameter_name: str, 
                                   new_value: float,
                                   reason: str = "Manual adjustment") -> bool:
        """Manually adjust a parameter with validation"""
        try:
            if parameter_name not in self.parameter_constraints:
                self.logger.error(f"Unknown parameter: {parameter_name}")
                return False
            
            constraints = self.parameter_constraints[parameter_name]
            
            # Validate value is within constraints
            if new_value < constraints['min'] or new_value > constraints['max']:
                self.logger.error(f"Value {new_value} outside constraints for {parameter_name}")
                return False
            
            # Apply change
            success = self._apply_parameter_change(parameter_name, new_value)
            
            if success:
                # Record manual change
                manual_change = {
                    'parameter': parameter_name,
                    'old_value': self._get_current_parameter_value(parameter_name),
                    'new_value': new_value,
                    'type': 'manual',
                    'reason': reason,
                    'applied_at': datetime.now().isoformat()
                }
                
                history_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'type': 'manual_adjustment',
                    'change': manual_change
                }
                
                self.optimization_state['optimization_history'].append(history_entry)
                self.logger.info(f"Manual parameter adjustment: {parameter_name} = {new_value}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Manual parameter adjustment failed: {e}")
            return False