"""
Automated Enhancement Optimizer for YouTube video processing system.
Monitors performance metrics and automatically fine-tunes enhancement parameters.
"""

import logging
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.engagement_metrics import EngagementMonitor, EnhancementType
from src.processing.sound_effects_manager import SoundEffectsManager
from src.config.settings import get_config, ConfigManager
import yaml


class ParameterOptimizer:
    """Optimizes enhancement parameters based on performance data"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.monitor = EngagementMonitor()
        self.sound_manager = SoundEffectsManager()
        
        # Parameter ranges for optimization
        self.parameter_ranges = {
            'sound_effects': {
                'volume_multiplier': (0.5, 1.5),  # Multiplier for base volume
                'fade_duration': (0.02, 0.1),     # Fade in/out duration
                'density_threshold': (0.1, 0.5),   # Max effects per second
            },
            'visual_effects': {
                'zoom_intensity': (1.02, 1.15),    # Zoom factor range
                'color_grade_intensity': (0.3, 1.0), # Color grading strength
                'transition_duration': (0.1, 0.5),  # Transition effect duration
            },
            'text_overlays': {
                'font_size_multiplier': (0.8, 1.2), # Font size adjustment
                'display_duration': (1.0, 4.0),     # Text display time
                'animation_speed': (0.5, 2.0),      # Animation speed
            }
        }
        
        # Current optimal parameters
        self.optimal_parameters = self._load_optimal_parameters()
    
    def _load_optimal_parameters(self) -> Dict[str, Dict[str, float]]:
        """Load current optimal parameters from file"""
        params_file = self.config.paths.base_dir / "optimal_parameters.json"
        
        if params_file.exists():
            try:
                with open(params_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading optimal parameters: {e}")
        
        # Return default parameters
        return {
            'sound_effects': {
                'volume_multiplier': 1.0,
                'fade_duration': 0.05,
                'density_threshold': 0.25,
            },
            'visual_effects': {
                'zoom_intensity': 1.05,
                'color_grade_intensity': 0.7,
                'transition_duration': 0.3,
            },
            'text_overlays': {
                'font_size_multiplier': 1.0,
                'display_duration': 2.5,
                'animation_speed': 1.0,
            }
        }
    
    def _save_optimal_parameters(self) -> None:
        """Save optimal parameters to file"""
        params_file = self.config.paths.base_dir / "optimal_parameters.json"
        
        try:
            with open(params_file, 'w') as f:
                json.dump(self.optimal_parameters, f, indent=2)
            
            self.logger.info(f"Optimal parameters saved to {params_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving optimal parameters: {e}")
    
    def analyze_parameter_performance(self, enhancement_type: str, 
                                    parameter_name: str, 
                                    days_back: int = 14) -> Dict[str, Any]:
        """Analyze performance of specific parameter values"""
        try:
            # Get videos with the enhancement
            cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            
            # This is a simplified analysis - in a real implementation,
            # you would need to track parameter values for each video
            performance_data = self.monitor.analyzer.analyze_enhancement_performance(
                enhancement_type, days_back
            )
            
            if not performance_data:
                return {}
            
            return {
                'parameter': parameter_name,
                'enhancement_type': enhancement_type,
                'sample_size': performance_data.videos_with_enhancement,
                'avg_engagement': performance_data.avg_engagement_with,
                'avg_retention': performance_data.avg_retention_with,
                'improvement': performance_data.engagement_improvement,
                'confidence': performance_data.sample_size_adequate
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing parameter performance: {e}")
            return {}
    
    def suggest_parameter_adjustments(self) -> Dict[str, Dict[str, float]]:
        """Suggest parameter adjustments based on performance analysis"""
        suggestions = {}
        
        for enhancement_type in ['sound_effects', 'visual_effects', 'text_overlays']:
            enhancement_suggestions = {}
            
            # Get current performance
            performance = self.monitor.analyzer.analyze_enhancement_performance(enhancement_type)
            
            if performance and performance.sample_size_adequate:
                # Analyze current effectiveness
                if performance.engagement_improvement > 10:
                    # High performance - small adjustments to optimize further
                    adjustment_factor = 1.05
                    self.logger.info(f"{enhancement_type} showing high performance (+{performance.engagement_improvement:.1f}%)")
                elif performance.engagement_improvement > 0:
                    # Moderate performance - try to improve
                    adjustment_factor = 1.1
                    self.logger.info(f"{enhancement_type} showing moderate performance (+{performance.engagement_improvement:.1f}%)")
                else:
                    # Poor performance - significant adjustments needed
                    adjustment_factor = 0.9
                    self.logger.warning(f"{enhancement_type} showing poor performance ({performance.engagement_improvement:.1f}%)")
                
                # Apply adjustments based on enhancement type
                current_params = self.optimal_parameters.get(enhancement_type, {})
                
                if enhancement_type == 'sound_effects':
                    if performance.engagement_improvement < 5:
                        # Try increasing volume slightly
                        enhancement_suggestions['volume_multiplier'] = min(
                            1.5, current_params.get('volume_multiplier', 1.0) * adjustment_factor
                        )
                        # Try longer fade for smoother integration
                        enhancement_suggestions['fade_duration'] = min(
                            0.1, current_params.get('fade_duration', 0.05) * 1.2
                        )
                    else:
                        # Performance is good, minor optimization
                        enhancement_suggestions['density_threshold'] = max(
                            0.1, current_params.get('density_threshold', 0.25) * 0.95
                        )
                
                elif enhancement_type == 'visual_effects':
                    if performance.engagement_improvement < 5:
                        # Try more subtle zoom
                        enhancement_suggestions['zoom_intensity'] = max(
                            1.02, current_params.get('zoom_intensity', 1.05) * 0.98
                        )
                        # Increase color grading
                        enhancement_suggestions['color_grade_intensity'] = min(
                            1.0, current_params.get('color_grade_intensity', 0.7) * adjustment_factor
                        )
                
                elif enhancement_type == 'text_overlays':
                    if performance.engagement_improvement < 5:
                        # Try larger text
                        enhancement_suggestions['font_size_multiplier'] = min(
                            1.2, current_params.get('font_size_multiplier', 1.0) * adjustment_factor
                        )
                        # Adjust display duration
                        enhancement_suggestions['display_duration'] = max(
                            1.0, current_params.get('display_duration', 2.5) * 0.9
                        )
            
            if enhancement_suggestions:
                suggestions[enhancement_type] = enhancement_suggestions
        
        return suggestions
    
    def apply_parameter_adjustments(self, adjustments: Dict[str, Dict[str, float]]) -> bool:
        """Apply parameter adjustments to optimal parameters"""
        try:
            for enhancement_type, params in adjustments.items():
                if enhancement_type not in self.optimal_parameters:
                    self.optimal_parameters[enhancement_type] = {}
                
                for param_name, new_value in params.items():
                    # Validate parameter is within acceptable range
                    if enhancement_type in self.parameter_ranges:
                        if param_name in self.parameter_ranges[enhancement_type]:
                            min_val, max_val = self.parameter_ranges[enhancement_type][param_name]
                            new_value = max(min_val, min(max_val, new_value))
                    
                    old_value = self.optimal_parameters[enhancement_type].get(param_name, 0)
                    self.optimal_parameters[enhancement_type][param_name] = new_value
                    
                    self.logger.info(f"Adjusted {enhancement_type}.{param_name}: {old_value:.3f} → {new_value:.3f}")
            
            self._save_optimal_parameters()
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying parameter adjustments: {e}")
            return False
    
    def optimize_parameters(self, auto_apply: bool = False) -> Dict[str, Any]:
        """Run parameter optimization analysis"""
        self.logger.info("Starting parameter optimization analysis...")
        
        # Get performance report
        report = self.monitor.analyzer.generate_performance_report()
        
        # Generate suggestions
        suggestions = self.suggest_parameter_adjustments()
        
        optimization_result = {
            'timestamp': datetime.now().isoformat(),
            'current_parameters': self.optimal_parameters.copy(),
            'suggested_adjustments': suggestions,
            'performance_report': report,
            'auto_applied': False
        }
        
        if suggestions:
            self.logger.info(f"Generated {len(suggestions)} parameter adjustment suggestions")
            
            if auto_apply:
                success = self.apply_parameter_adjustments(suggestions)
                optimization_result['auto_applied'] = success
                
                if success:
                    self.logger.info("Parameter adjustments applied automatically")
                    optimization_result['new_parameters'] = self.optimal_parameters.copy()
                else:
                    self.logger.error("Failed to apply parameter adjustments")
            else:
                self.logger.info("Parameter adjustments suggested (not applied automatically)")
        else:
            self.logger.info("No parameter adjustments suggested at this time")
        
        return optimization_result
    
    def update_config_with_optimal_parameters(self) -> bool:
        """Update system configuration with optimal parameters"""
        try:
            config_file = self.config.paths.base_dir / "config.yaml"
            
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
            else:
                config_data = {}
            
            # Update configuration with optimal parameters
            if 'enhancement_optimization' not in config_data:
                config_data['enhancement_optimization'] = {}
            
            config_data['enhancement_optimization']['optimal_parameters'] = self.optimal_parameters
            config_data['enhancement_optimization']['last_updated'] = datetime.now().isoformat()
            
            # Apply sound effects parameters
            if 'sound_effects' in self.optimal_parameters:
                if 'audio' not in config_data:
                    config_data['audio'] = {}
                
                se_params = self.optimal_parameters['sound_effects']
                config_data['audio']['sound_effects_volume_multiplier'] = se_params.get('volume_multiplier', 1.0)
                config_data['audio']['sound_effects_fade_duration'] = se_params.get('fade_duration', 0.05)
            
            # Apply visual effects parameters
            if 'visual_effects' in self.optimal_parameters:
                if 'effects' not in config_data:
                    config_data['effects'] = {}
                
                ve_params = self.optimal_parameters['visual_effects']
                config_data['effects']['max_zoom'] = ve_params.get('zoom_intensity', 1.05)
                config_data['effects']['color_grade_intensity'] = ve_params.get('color_grade_intensity', 0.7)
            
            # Apply text overlay parameters
            if 'text_overlays' in self.optimal_parameters:
                if 'text_overlay' not in config_data:
                    config_data['text_overlay'] = {}
                
                to_params = self.optimal_parameters['text_overlays']
                config_data['text_overlay']['font_size_multiplier'] = to_params.get('font_size_multiplier', 1.0)
                config_data['text_overlay']['default_duration'] = to_params.get('display_duration', 2.5)
            
            # Save updated configuration
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration updated with optimal parameters: {config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            return False


def main():
    """Main optimizer function"""
    parser = argparse.ArgumentParser(description="YouTube Video Enhancement Optimizer")
    parser.add_argument('--days-back', type=int, default=14, 
                       help='Number of days to analyze (default: 14)')
    parser.add_argument('--auto-apply', action='store_true',
                       help='Automatically apply suggested parameter adjustments')
    parser.add_argument('--update-config', action='store_true',
                       help='Update system configuration with optimal parameters')
    parser.add_argument('--output', type=Path,
                       help='Output file for optimization results (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize optimizer
        optimizer = ParameterOptimizer()
        
        # Run optimization
        logger.info("Starting enhancement parameter optimization...")
        result = optimizer.optimize_parameters(auto_apply=args.auto_apply)
        
        # Update configuration if requested
        if args.update_config:
            logger.info("Updating system configuration...")
            success = optimizer.update_config_with_optimal_parameters()
            result['config_updated'] = success
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Optimization results saved to {args.output}")
        
        # Print summary
        print("\n" + "="*60)
        print("ENHANCEMENT OPTIMIZATION SUMMARY")
        print("="*60)
        
        if result['suggested_adjustments']:
            print(f"Adjustments suggested for {len(result['suggested_adjustments'])} enhancement types:")
            for enhancement_type, adjustments in result['suggested_adjustments'].items():
                print(f"\n{enhancement_type.upper()}:")
                for param, value in adjustments.items():
                    current = result['current_parameters'].get(enhancement_type, {}).get(param, 'N/A')
                    print(f"  {param}: {current} → {value:.3f}")
        else:
            print("No parameter adjustments suggested at this time.")
        
        if result.get('auto_applied'):
            print(f"\n[SUCCESS] Adjustments applied automatically")
        elif result['suggested_adjustments']:
            print(f"\n[INFO] Adjustments suggested (use --auto-apply to apply)")
        
        if result.get('config_updated'):
            print(f"[SUCCESS] System configuration updated")
        
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()