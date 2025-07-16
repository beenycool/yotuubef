#!/usr/bin/env python3
"""
Configuration Validator and Optimizer
Validates configuration settings and optimizes them for better video generation
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import yaml
import json
from dataclasses import dataclass, field
from enum import Enum


class ConfigIssueType(Enum):
    """Types of configuration issues"""
    MISSING_REQUIRED = "missing_required"
    INVALID_VALUE = "invalid_value"
    SUBOPTIMAL_SETTING = "suboptimal_setting"
    DEPRECATED_SETTING = "deprecated_setting"
    COMPATIBILITY_ISSUE = "compatibility_issue"


@dataclass
class ConfigIssue:
    """Represents a configuration issue"""
    type: ConfigIssueType
    severity: str  # "critical", "warning", "info"
    section: str
    key: str
    current_value: Any
    recommended_value: Any
    description: str
    fix_suggestion: str


class ConfigValidator:
    """Validates and optimizes configuration settings"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.issues: List[ConfigIssue] = []
        
        # Define validation rules
        self.validation_rules = {
            'api': {
                'required_keys': ['reddit_client_id', 'reddit_client_secret', 'gemini_api_key'],
                'optional_keys': ['reddit_user_agent', 'youtube_api_key', 'elevenlabs_api_key'],
                'validations': {
                    'gemini_rate_limit_rpm': {'type': int, 'min': 1, 'max': 60, 'recommended': 10},
                    'gemini_rate_limit_daily': {'type': int, 'min': 1, 'max': 10000, 'recommended': 500},
                }
            },
            'video_processing': {
                'required_keys': ['target_fps', 'default_output_resolution'],
                'validations': {
                    'target_fps': {'type': int, 'min': 24, 'max': 60, 'recommended': 30},
                    'video_bitrate': {'type': str, 'pattern': r'^\d+[KkMm]$', 'recommended': '5M'},
                    'audio_bitrate': {'type': str, 'pattern': r'^\d+[Kk]$', 'recommended': '128k'},
                    'max_video_duration': {'type': int, 'min': 30, 'max': 3600, 'recommended': 600},
                    'min_video_duration': {'type': int, 'min': 1, 'max': 30, 'recommended': 5},
                }
            },
            'ai_features': {
                'required_keys': [],
                'validations': {
                    'enable_cinematic_editing': {'type': bool, 'recommended': True},
                    'enable_advanced_audio': {'type': bool, 'recommended': True},
                    'enable_ab_testing': {'type': bool, 'recommended': True},
                    'enable_auto_optimization': {'type': bool, 'recommended': True},
                    'fallback_confidence_threshold': {'type': float, 'min': 0.0, 'max': 1.0, 'recommended': 0.6},
                }
            },
            'long_form_video': {
                'required_keys': ['enable_long_form_generation'],
                'validations': {
                    'default_duration_minutes': {'type': int, 'min': 1, 'max': 60, 'recommended': 5},
                    'max_duration_minutes': {'type': int, 'min': 1, 'max': 180, 'recommended': 60},
                    'words_per_minute': {'type': int, 'min': 100, 'max': 200, 'recommended': 150},
                    'pause_between_sections': {'type': float, 'min': 0.5, 'max': 5.0, 'recommended': 2.0},
                }
            },
            'performance': {
                'required_keys': [],
                'validations': {
                    'max_concurrent_videos': {'type': int, 'min': 1, 'max': 10, 'recommended': 3},
                    'max_vram_usage': {'type': float, 'min': 0.1, 'max': 0.95, 'recommended': 0.7},
                    'chunk_size_mb': {'type': int, 'min': 10, 'max': 1000, 'recommended': 100},
                }
            }
        }
        
        # Performance optimization recommendations
        self.performance_recommendations = {
            'low_memory': {
                'max_concurrent_videos': 1,
                'chunk_size_mb': 50,
                'max_vram_usage': 0.5,
                'aggressive_memory_cleanup': True,
                'description': 'Optimized for systems with limited memory'
            },
            'balanced': {
                'max_concurrent_videos': 3,
                'chunk_size_mb': 100,
                'max_vram_usage': 0.7,
                'aggressive_memory_cleanup': True,
                'description': 'Balanced performance and resource usage'
            },
            'high_performance': {
                'max_concurrent_videos': 5,
                'chunk_size_mb': 200,
                'max_vram_usage': 0.85,
                'aggressive_memory_cleanup': False,
                'description': 'Maximum performance for high-end systems'
            }
        }
        
        # Quality optimization recommendations
        self.quality_recommendations = {
            'maximum_quality': {
                'video_bitrate': '15M',
                'audio_bitrate': '320k',
                'target_fps': 30,
                'enable_cinematic_editing': True,
                'enable_advanced_audio': True,
                'video_quality_profile': 'maximum',
                'description': 'Maximum quality settings'
            },
            'balanced_quality': {
                'video_bitrate': '5M',
                'audio_bitrate': '128k',
                'target_fps': 30,
                'enable_cinematic_editing': True,
                'enable_advanced_audio': True,
                'video_quality_profile': 'high',
                'description': 'Balanced quality and file size'
            },
            'fast_processing': {
                'video_bitrate': '3M',
                'audio_bitrate': '96k',
                'target_fps': 30,
                'enable_cinematic_editing': False,
                'enable_advanced_audio': False,
                'video_quality_profile': 'speed',
                'description': 'Fast processing with acceptable quality'
            }
        }
    
    def validate_config(self, config_dict: Dict[str, Any]) -> List[ConfigIssue]:
        """
        Validate configuration dictionary
        
        Args:
            config_dict: Configuration dictionary to validate
            
        Returns:
            List of configuration issues found
        """
        self.issues = []
        
        # Validate each section
        for section_name, section_rules in self.validation_rules.items():
            if section_name not in config_dict:
                # Section is missing entirely
                self.issues.append(ConfigIssue(
                    type=ConfigIssueType.MISSING_REQUIRED,
                    severity="warning",
                    section=section_name,
                    key="",
                    current_value=None,
                    recommended_value={},
                    description=f"Configuration section '{section_name}' is missing",
                    fix_suggestion=f"Add '{section_name}' section to configuration"
                ))
                continue
            
            section_config = config_dict[section_name]
            
            # Check required keys
            for required_key in section_rules.get('required_keys', []):
                if required_key not in section_config:
                    self.issues.append(ConfigIssue(
                        type=ConfigIssueType.MISSING_REQUIRED,
                        severity="critical",
                        section=section_name,
                        key=required_key,
                        current_value=None,
                        recommended_value="<required>",
                        description=f"Required key '{required_key}' is missing",
                        fix_suggestion=f"Add '{required_key}' to {section_name} section"
                    ))
            
            # Validate existing keys
            validations = section_rules.get('validations', {})
            for key, validation_rule in validations.items():
                if key in section_config:
                    self._validate_key(section_name, key, section_config[key], validation_rule)
        
        # Check for deprecated settings
        self._check_deprecated_settings(config_dict)
        
        # Check for compatibility issues
        self._check_compatibility_issues(config_dict)
        
        return self.issues
    
    def _validate_key(self, section: str, key: str, value: Any, rule: Dict[str, Any]):
        """Validate a single configuration key"""
        
        # Type validation
        expected_type = rule.get('type')
        if expected_type and not isinstance(value, expected_type):
            self.issues.append(ConfigIssue(
                type=ConfigIssueType.INVALID_VALUE,
                severity="critical",
                section=section,
                key=key,
                current_value=value,
                recommended_value=f"<{expected_type.__name__}>",
                description=f"Expected {expected_type.__name__}, got {type(value).__name__}",
                fix_suggestion=f"Change {key} to {expected_type.__name__} type"
            ))
            return
        
        # Range validation
        if 'min' in rule and isinstance(value, (int, float)) and value < rule['min']:
            self.issues.append(ConfigIssue(
                type=ConfigIssueType.INVALID_VALUE,
                severity="warning",
                section=section,
                key=key,
                current_value=value,
                recommended_value=rule['min'],
                description=f"Value {value} is below minimum {rule['min']}",
                fix_suggestion=f"Increase {key} to at least {rule['min']}"
            ))
        
        if 'max' in rule and isinstance(value, (int, float)) and value > rule['max']:
            self.issues.append(ConfigIssue(
                type=ConfigIssueType.INVALID_VALUE,
                severity="warning",
                section=section,
                key=key,
                current_value=value,
                recommended_value=rule['max'],
                description=f"Value {value} is above maximum {rule['max']}",
                fix_suggestion=f"Decrease {key} to at most {rule['max']}"
            ))
        
        # Pattern validation
        if 'pattern' in rule and isinstance(value, str):
            import re
            if not re.match(rule['pattern'], value):
                self.issues.append(ConfigIssue(
                    type=ConfigIssueType.INVALID_VALUE,
                    severity="warning",
                    section=section,
                    key=key,
                    current_value=value,
                    recommended_value=rule.get('recommended', '<valid pattern>'),
                    description=f"Value does not match expected pattern",
                    fix_suggestion=f"Update {key} to match pattern: {rule['pattern']}"
                ))
        
        # Recommendation check
        if 'recommended' in rule and value != rule['recommended']:
            self.issues.append(ConfigIssue(
                type=ConfigIssueType.SUBOPTIMAL_SETTING,
                severity="info",
                section=section,
                key=key,
                current_value=value,
                recommended_value=rule['recommended'],
                description=f"Non-optimal value, recommended: {rule['recommended']}",
                fix_suggestion=f"Consider changing {key} to {rule['recommended']} for better performance"
            ))
    
    def _check_deprecated_settings(self, config_dict: Dict[str, Any]):
        """Check for deprecated configuration settings"""
        deprecated_settings = {
            'api': {
                'openai_api_key': 'Use gemini_api_key instead',
                'openai_model': 'Use gemini_model instead',
            },
            'video_processing': {
                'use_gpu': 'Use enable_gpu_acceleration instead',
                'output_format': 'Format is now automatically determined',
            }
        }
        
        for section_name, deprecated_keys in deprecated_settings.items():
            if section_name in config_dict:
                for key, message in deprecated_keys.items():
                    if key in config_dict[section_name]:
                        self.issues.append(ConfigIssue(
                            type=ConfigIssueType.DEPRECATED_SETTING,
                            severity="warning",
                            section=section_name,
                            key=key,
                            current_value=config_dict[section_name][key],
                            recommended_value=None,
                            description=f"Setting '{key}' is deprecated",
                            fix_suggestion=message
                        ))
    
    def _check_compatibility_issues(self, config_dict: Dict[str, Any]):
        """Check for compatibility issues between settings"""
        
        # Check GPU settings compatibility
        if 'performance' in config_dict and 'video_processing' in config_dict:
            perf_config = config_dict['performance']
            video_config = config_dict['video_processing']
            
            # High concurrent videos with high quality might cause issues
            if (perf_config.get('max_concurrent_videos', 1) > 3 and 
                video_config.get('video_quality_profile') == 'maximum'):
                self.issues.append(ConfigIssue(
                    type=ConfigIssueType.COMPATIBILITY_ISSUE,
                    severity="warning",
                    section="performance",
                    key="max_concurrent_videos",
                    current_value=perf_config.get('max_concurrent_videos'),
                    recommended_value=3,
                    description="High concurrent processing with maximum quality may cause memory issues",
                    fix_suggestion="Reduce concurrent videos or use lower quality profile"
                ))
            
            # VRAM usage vs GPU acceleration
            if (perf_config.get('max_vram_usage', 0.7) > 0.85 and 
                video_config.get('enable_gpu_acceleration', True)):
                self.issues.append(ConfigIssue(
                    type=ConfigIssueType.COMPATIBILITY_ISSUE,
                    severity="warning",
                    section="performance",
                    key="max_vram_usage",
                    current_value=perf_config.get('max_vram_usage'),
                    recommended_value=0.7,
                    description="High VRAM usage may cause GPU acceleration issues",
                    fix_suggestion="Reduce VRAM usage or disable GPU acceleration"
                ))
    
    def optimize_config(self, config_dict: Dict[str, Any], optimization_target: str = "balanced") -> Dict[str, Any]:
        """
        Optimize configuration for specific target
        
        Args:
            config_dict: Current configuration
            optimization_target: "quality", "performance", "memory", "balanced"
            
        Returns:
            Optimized configuration
        """
        optimized = config_dict.copy()
        
        if optimization_target == "quality":
            # Apply quality optimizations
            if 'video_processing' not in optimized:
                optimized['video_processing'] = {}
            optimized['video_processing'].update(self.quality_recommendations['maximum_quality'])
            
            if 'performance' not in optimized:
                optimized['performance'] = {}
            optimized['performance'].update({
                'max_concurrent_videos': 1,  # Single video for best quality
                'aggressive_memory_cleanup': True
            })
        
        elif optimization_target == "performance":
            # Apply performance optimizations
            if 'video_processing' not in optimized:
                optimized['video_processing'] = {}
            optimized['video_processing'].update(self.quality_recommendations['fast_processing'])
            
            if 'performance' not in optimized:
                optimized['performance'] = {}
            optimized['performance'].update(self.performance_recommendations['high_performance'])
        
        elif optimization_target == "memory":
            # Apply memory optimizations
            if 'performance' not in optimized:
                optimized['performance'] = {}
            optimized['performance'].update(self.performance_recommendations['low_memory'])
            
            if 'video_processing' not in optimized:
                optimized['video_processing'] = {}
            optimized['video_processing'].update({
                'video_bitrate': '3M',
                'audio_bitrate': '96k',
                'chunk_size_mb': 50
            })
        
        else:  # balanced
            # Apply balanced optimizations
            if 'performance' not in optimized:
                optimized['performance'] = {}
            optimized['performance'].update(self.performance_recommendations['balanced'])
            
            if 'video_processing' not in optimized:
                optimized['video_processing'] = {}
            optimized['video_processing'].update(self.quality_recommendations['balanced_quality'])
        
        return optimized
    
    def generate_validation_report(self, config_dict: Dict[str, Any]) -> str:
        """Generate a comprehensive validation report"""
        issues = self.validate_config(config_dict)
        
        # Group issues by severity
        critical_issues = [i for i in issues if i.severity == "critical"]
        warning_issues = [i for i in issues if i.severity == "warning"]
        info_issues = [i for i in issues if i.severity == "info"]
        
        report = f"""
⚙️ Configuration Validation Report
{'='*50}

📊 Issue Summary:
   Critical Issues: {len(critical_issues)}
   Warnings: {len(warning_issues)}
   Info/Recommendations: {len(info_issues)}
   Total Issues: {len(issues)}

"""
        
        if critical_issues:
            report += "🚨 Critical Issues (Must Fix):\n"
            for issue in critical_issues:
                report += f"   • {issue.section}.{issue.key}: {issue.description}\n"
                report += f"     Fix: {issue.fix_suggestion}\n"
        
        if warning_issues:
            report += "\n⚠️  Warnings (Should Fix):\n"
            for issue in warning_issues:
                report += f"   • {issue.section}.{issue.key}: {issue.description}\n"
                report += f"     Current: {issue.current_value} → Recommended: {issue.recommended_value}\n"
        
        if info_issues:
            report += "\n💡 Recommendations (Consider):\n"
            for issue in info_issues:
                report += f"   • {issue.section}.{issue.key}: {issue.description}\n"
                report += f"     Current: {issue.current_value} → Recommended: {issue.recommended_value}\n"
        
        report += f"""
🎯 Optimization Targets Available:
   • quality: Maximum video quality (slower processing)
   • performance: Fast processing (lower quality)
   • memory: Low memory usage (for limited systems)
   • balanced: Good balance of quality and performance

🛠️ Usage:
   validator = ConfigValidator()
   optimized_config = validator.optimize_config(config, "balanced")

{'='*50}
"""
        
        return report


def main():
    """Demonstration of configuration validation and optimization"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("⚙️ Configuration Validator and Optimizer Demo")
    print("=" * 50)
    
    # Load existing configuration
    config_file = Path("config.yaml")
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Loaded configuration from {config_file}")
    else:
        # Create sample configuration
        config = {
            'api': {
                'reddit_client_id': 'your_client_id',
                'reddit_client_secret': 'your_client_secret',
                'gemini_api_key': 'your_gemini_key',
                'gemini_rate_limit_rpm': 15,  # Suboptimal value
                'gemini_rate_limit_daily': 1000,  # Suboptimal value
            },
            'video_processing': {
                'target_fps': 25,  # Suboptimal value
                'video_bitrate': '8M',
                'audio_bitrate': '128k',
                'max_video_duration': 300,
                'enable_gpu_acceleration': True,
            },
            'performance': {
                'max_concurrent_videos': 5,  # Might be too high
                'max_vram_usage': 0.9,  # Too high
                'chunk_size_mb': 200,
            },
            'ai_features': {
                'enable_cinematic_editing': False,  # Suboptimal
                'enable_advanced_audio': True,
                'enable_ab_testing': True,
            }
        }
        print("📄 Using sample configuration for demonstration")
    
    # Initialize validator
    validator = ConfigValidator()
    
    # Validate configuration
    print("\n🔍 Validating configuration...")
    report = validator.generate_validation_report(config)
    print(report)
    
    # Show optimization options
    print("\n🎯 Optimization Examples:")
    
    optimization_targets = ["quality", "performance", "memory", "balanced"]
    for target in optimization_targets:
        print(f"\n📊 {target.upper()} Optimization:")
        optimized = validator.optimize_config(config, target)
        
        # Show key differences
        if 'performance' in optimized:
            perf = optimized['performance']
            print(f"   • Concurrent videos: {perf.get('max_concurrent_videos', 'N/A')}")
            print(f"   • VRAM usage: {perf.get('max_vram_usage', 'N/A')}")
        
        if 'video_processing' in optimized:
            video = optimized['video_processing']
            print(f"   • Video bitrate: {video.get('video_bitrate', 'N/A')}")
            print(f"   • Quality profile: {video.get('video_quality_profile', 'N/A')}")
    
    # Save optimized configuration
    print("\n💾 Saving optimized configurations...")
    for target in optimization_targets:
        optimized = validator.optimize_config(config, target)
        output_file = Path(f"config_optimized_{target}.yaml")
        with open(output_file, 'w') as f:
            yaml.dump(optimized, f, default_flow_style=False, sort_keys=False)
        print(f"   • {target}: {output_file}")
    
    # Save validation report
    report_file = Path("config_validation_report.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"\n📄 Validation report saved to: {report_file}")


if __name__ == "__main__":
    main()