"""
Comprehensive test runner for video enhancement system.
Tests sound effects, visual effects, engagement monitoring, and parameter optimization.
"""

import logging
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processing.sound_effects_manager import SoundEffectsManager
from src.monitoring.engagement_metrics import EngagementMonitor, VideoMetrics
from src.config.settings import get_config
from scripts.enhancement_optimizer import ParameterOptimizer


class EnhancementTestRunner:
    """Comprehensive test runner for enhancement system"""
    
    def __init__(self, verbose: bool = False):
        self.config = get_config()
        self.logger = self._setup_logging(verbose)
        self.test_results: Dict[str, Dict] = {}
        
        # Create temporary test environment
        self.temp_dir = None
        self._setup_test_environment()
    
    def _setup_logging(self, verbose: bool) -> logging.Logger:
        """Setup logging configuration"""
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _setup_test_environment(self):
        """Setup temporary test environment"""
        try:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="enhancement_test_"))
            self.logger.info(f"Created test environment: {self.temp_dir}")
            
            # Create sound effects test files
            self._create_test_sound_files()
            
        except Exception as e:
            self.logger.error(f"Failed to setup test environment: {e}")
            raise
    
    def _create_test_sound_files(self):
        """Create test sound effect files"""
        try:
            # Create directory structure
            sound_effects_dir = self.temp_dir / "sound_effects"
            categories = ['impact', 'transition', 'liquid', 'mechanical', 'notification', 'dramatic']
            
            for category in categories:
                (sound_effects_dir / category).mkdir(parents=True, exist_ok=True)
            
            # Create minimal WAV file headers (for testing purposes)
            wav_header = b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
            
            # Create test sound files
            test_sounds = {
                'impact/impact.wav': wav_header,
                'impact/hit.wav': wav_header,
                'impact/thud.wav': wav_header,
                'transition/whoosh.wav': wav_header,
                'transition/swoosh.wav': wav_header,
                'transition/zip.wav': wav_header,
                'liquid/splash.wav': wav_header,
                'liquid/pour.wav': wav_header,
                'mechanical/click.wav': wav_header,
                'mechanical/pop.wav': wav_header,
                'notification/ding.wav': wav_header,
                'notification/chime.wav': wav_header,
                'dramatic/boom.wav': wav_header,
                'dramatic/thunder.wav': wav_header,
            }
            
            for file_path, content in test_sounds.items():
                full_path = sound_effects_dir / file_path
                full_path.write_bytes(content)
            
            self.logger.info(f"Created {len(test_sounds)} test sound files")
            
        except Exception as e:
            self.logger.error(f"Failed to create test sound files: {e}")
            raise
    
    def test_sound_effects_manager(self) -> Dict[str, Any]:
        """Test SoundEffectsManager functionality"""
        test_name = "sound_effects_manager"
        self.logger.info(f"Testing {test_name}...")
        
        result = {
            'test_name': test_name,
            'status': 'UNKNOWN',
            'details': {},
            'errors': []
        }
        
        try:
            # Patch config to use test directory
            original_sound_dir = self.config.paths.sound_effects_folder
            self.config.paths.sound_effects_folder = self.temp_dir / "sound_effects"
            
            # Initialize manager
            manager = SoundEffectsManager()
            
            # Test 1: Cache initialization
            cache_status = manager.get_cache_status()
            result['details']['cache_total_effects'] = cache_status['total_effects']
            result['details']['cache_categories'] = cache_status['categories']
            
            if cache_status['total_effects'] == 0:
                result['errors'].append("No sound effects found in cache")
            
            # Test 2: Sound effect discovery
            test_effects = ['whoosh', 'splash', 'click', 'impact', 'ding', 'boom']
            found_effects = {}
            
            for effect in test_effects:
                found_file = manager.find_sound_effect(effect)
                found_effects[effect] = found_file is not None
                
                if found_file:
                    # Test validation
                    try:
                        from unittest.mock import patch, Mock
                        with patch('src.processing.sound_effects_manager.AudioFileClip') as mock_clip:
                            mock_audio = Mock()
                            mock_audio.duration = 2.0
                            mock_clip.return_value.__enter__.return_value = mock_audio
                            
                            is_valid, message = manager.validate_sound_effect(found_file)
                            if not is_valid:
                                result['errors'].append(f"Validation failed for {effect}: {message}")
                    except Exception as e:
                        result['errors'].append(f"Validation error for {effect}: {e}")
            
            result['details']['found_effects'] = found_effects
            found_count = sum(found_effects.values())
            result['details']['found_count'] = found_count
            
            # Test 3: Category mapping
            available_effects = manager.get_available_effects()
            result['details']['available_by_category'] = {
                cat: len(effects) for cat, effects in available_effects.items()
            }
            
            # Test 4: Fallback mechanism
            nonexistent_effect = manager.find_sound_effect('nonexistent_super_rare_effect')
            result['details']['fallback_works'] = nonexistent_effect is not None
            
            # Determine overall status
            if result['errors']:
                result['status'] = 'FAILED'
            elif found_count >= len(test_effects) * 0.8:  # 80% success rate
                result['status'] = 'PASSED'
            else:
                result['status'] = 'PARTIAL'
                result['errors'].append(f"Only found {found_count}/{len(test_effects)} test effects")
            
            # Restore original config
            self.config.paths.sound_effects_folder = original_sound_dir
            
        except Exception as e:
            result['status'] = 'ERROR'
            result['errors'].append(f"Test execution failed: {e}")
            self.logger.error(f"Sound effects manager test failed: {e}")
        
        self.test_results[test_name] = result
        return result
    
    def test_engagement_monitoring(self) -> Dict[str, Any]:
        """Test engagement monitoring system"""
        test_name = "engagement_monitoring"
        self.logger.info(f"Testing {test_name}...")
        
        result = {
            'test_name': test_name,
            'status': 'UNKNOWN',
            'details': {},
            'errors': []
        }
        
        try:
            # Use temporary database
            test_db_path = self.temp_dir / "test_engagement.db"
            
            from src.monitoring.engagement_metrics import EngagementMetricsDB
            
            # Test 1: Database initialization
            db = EngagementMetricsDB(test_db_path)
            result['details']['database_created'] = test_db_path.exists()
            
            if not test_db_path.exists():
                result['errors'].append("Database file not created")
            
            # Test 2: Store and retrieve metrics
            test_metrics = VideoMetrics(
                video_id="test_video_001",
                title="Test Video for Monitoring",
                upload_date=datetime.now(),
                duration_seconds=30.0,
                views=1000,
                likes=50,
                comments=10,
                enhancements_used=['sound_effects', 'visual_effects']
            )
            
            store_success = db.store_video_metrics(test_metrics)
            result['details']['store_metrics_success'] = store_success
            
            if not store_success:
                result['errors'].append("Failed to store video metrics")
            
            # Retrieve and verify
            retrieved_metrics = db.get_video_metrics("test_video_001")
            result['details']['retrieve_metrics_success'] = retrieved_metrics is not None
            
            if retrieved_metrics:
                result['details']['metrics_match'] = (
                    retrieved_metrics.video_id == test_metrics.video_id and
                    retrieved_metrics.views == test_metrics.views and
                    len(retrieved_metrics.enhancements_used) == len(test_metrics.enhancements_used)
                )
            else:
                result['errors'].append("Failed to retrieve video metrics")
            
            # Test 3: Enhancement tracking
            db.track_enhancement("test_video_001", "sound_effects", {"volume": 0.8})
            result['details']['enhancement_tracking_completed'] = True
            
            # Test 4: Monitor initialization
            monitor = EngagementMonitor()
            monitor.db = db  # Use test database
            
            # Test performance analysis (may return None if insufficient data)
            performance = monitor.analyzer.analyze_enhancement_performance('sound_effects')
            result['details']['performance_analysis_completed'] = True
            result['details']['performance_analysis_result'] = performance is not None
            
            # Test report generation
            report = monitor.analyzer.generate_performance_report()
            result['details']['report_generated'] = bool(report)
            result['details']['report_keys'] = list(report.keys()) if report else []
            
            # Determine status
            if result['errors']:
                result['status'] = 'FAILED'
            elif all([
                result['details'].get('database_created', False),
                result['details'].get('store_metrics_success', False),
                result['details'].get('retrieve_metrics_success', False),
                result['details'].get('metrics_match', False)
            ]):
                result['status'] = 'PASSED'
            else:
                result['status'] = 'PARTIAL'
        
        except Exception as e:
            result['status'] = 'ERROR'
            result['errors'].append(f"Test execution failed: {e}")
            self.logger.error(f"Engagement monitoring test failed: {e}")
        
        self.test_results[test_name] = result
        return result
    
    def test_parameter_optimization(self) -> Dict[str, Any]:
        """Test parameter optimization system"""
        test_name = "parameter_optimization"
        self.logger.info(f"Testing {test_name}...")
        
        result = {
            'test_name': test_name,
            'status': 'UNKNOWN',
            'details': {},
            'errors': []
        }
        
        try:
            # Test 1: Optimizer initialization
            optimizer = ParameterOptimizer()
            result['details']['optimizer_initialized'] = True
            
            # Test 2: Load/save optimal parameters
            original_params = optimizer.optimal_parameters.copy()
            result['details']['initial_parameters_loaded'] = len(original_params) > 0
            
            # Test parameter adjustment suggestion
            suggestions = optimizer.suggest_parameter_adjustments()
            result['details']['suggestions_generated'] = True
            result['details']['suggestion_count'] = len(suggestions)
            result['details']['suggestion_types'] = list(suggestions.keys())
            
            # Test 3: Parameter validation and ranges
            test_adjustments = {
                'sound_effects': {
                    'volume_multiplier': 1.2,
                    'fade_duration': 0.08
                },
                'visual_effects': {
                    'zoom_intensity': 1.08
                }
            }
            
            apply_success = optimizer.apply_parameter_adjustments(test_adjustments)
            result['details']['apply_adjustments_success'] = apply_success
            
            if apply_success:
                # Verify parameters were applied
                new_params = optimizer.optimal_parameters
                volume_updated = new_params.get('sound_effects', {}).get('volume_multiplier') == 1.2
                zoom_updated = new_params.get('visual_effects', {}).get('zoom_intensity') == 1.08
                
                result['details']['parameters_updated_correctly'] = volume_updated and zoom_updated
            
            # Test 4: Configuration update
            try:
                config_update_success = optimizer.update_config_with_optimal_parameters()
                result['details']['config_update_attempted'] = True
                result['details']['config_update_success'] = config_update_success
            except Exception as e:
                result['details']['config_update_error'] = str(e)
            
            # Restore original parameters
            optimizer.optimal_parameters = original_params
            optimizer._save_optimal_parameters()
            
            # Determine status
            if result['errors']:
                result['status'] = 'FAILED'
            elif all([
                result['details'].get('optimizer_initialized', False),
                result['details'].get('initial_parameters_loaded', False),
                result['details'].get('apply_adjustments_success', False)
            ]):
                result['status'] = 'PASSED'
            else:
                result['status'] = 'PARTIAL'
        
        except Exception as e:
            result['status'] = 'ERROR'
            result['errors'].append(f"Test execution failed: {e}")
            self.logger.error(f"Parameter optimization test failed: {e}")
        
        self.test_results[test_name] = result
        return result
    
    def test_system_integration(self) -> Dict[str, Any]:
        """Test system integration and configuration"""
        test_name = "system_integration"
        self.logger.info(f"Testing {test_name}...")
        
        result = {
            'test_name': test_name,
            'status': 'UNKNOWN',
            'details': {},
            'errors': []
        }
        
        try:
            # Test 1: Configuration loading
            config = get_config()
            result['details']['config_loaded'] = config is not None
            
            if config:
                # Check critical paths
                paths_exist = {
                    'sound_effects_folder': config.paths.sound_effects_folder.exists(),
                    'music_folder': config.paths.music_folder.exists(),
                    'fonts_folder': config.paths.fonts_folder.exists(),
                    'temp_dir': config.paths.temp_dir.exists()
                }
                result['details']['paths_exist'] = paths_exist
                result['details']['all_paths_exist'] = all(paths_exist.values())
                
                # Check configuration sections
                config_sections = {
                    'video_config': hasattr(config, 'video'),
                    'audio_config': hasattr(config, 'audio'),
                    'effects_config': hasattr(config, 'effects'),
                    'text_overlay_config': hasattr(config, 'text_overlay')
                }
                result['details']['config_sections'] = config_sections
                result['details']['all_sections_present'] = all(config_sections.values())
            
            # Test 2: Sound effects directory structure
            sound_dir = config.paths.sound_effects_folder
            if sound_dir.exists():
                expected_categories = ['impact', 'transition', 'liquid', 'mechanical', 'notification', 'dramatic']
                category_status = {}
                
                for category in expected_categories:
                    category_dir = sound_dir / category
                    category_status[category] = category_dir.exists()
                
                result['details']['sound_categories_exist'] = category_status
                result['details']['all_sound_categories_exist'] = all(category_status.values())
            else:
                result['errors'].append("Sound effects directory does not exist")
            
            # Test 3: Import all enhancement modules
            try:
                from src.processing.sound_effects_manager import SoundEffectsManager
                from src.monitoring.engagement_metrics import EngagementMonitor
                from scripts.enhancement_optimizer import ParameterOptimizer
                
                result['details']['all_modules_importable'] = True
            except ImportError as e:
                result['errors'].append(f"Module import failed: {e}")
                result['details']['all_modules_importable'] = False
            
            # Test 4: Database connectivity
            try:
                from src.database.db_manager import DatabaseManager
                db_manager = DatabaseManager()
                result['details']['database_manager_initialized'] = True
            except Exception as e:
                result['details']['database_manager_error'] = str(e)
            
            # Determine status
            if result['errors']:
                result['status'] = 'FAILED'
            elif all([
                result['details'].get('config_loaded', False),
                result['details'].get('all_sections_present', False),
                result['details'].get('all_modules_importable', False)
            ]):
                result['status'] = 'PASSED'
            else:
                result['status'] = 'PARTIAL'
        
        except Exception as e:
            result['status'] = 'ERROR'
            result['errors'].append(f"Test execution failed: {e}")
            self.logger.error(f"System integration test failed: {e}")
        
        self.test_results[test_name] = result
        return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all enhancement tests"""
        self.logger.info("Starting comprehensive enhancement testing...")
        
        start_time = datetime.now()
        
        # Run individual tests
        tests = [
            self.test_sound_effects_manager,
            self.test_engagement_monitoring,
            self.test_parameter_optimization,
            self.test_system_integration
        ]
        
        for test_func in tests:
            try:
                test_func()
            except Exception as e:
                self.logger.error(f"Test {test_func.__name__} failed with exception: {e}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Compile overall results
        overall_result = {
            'test_run_info': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'total_tests': len(self.test_results)
            },
            'test_results': self.test_results,
            'summary': self._generate_test_summary()
        }
        
        return overall_result
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate test summary statistics"""
        if not self.test_results:
            return {'status': 'NO_TESTS_RUN'}
        
        status_counts = {}
        for test_result in self.test_results.values():
            status = test_result['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        total_tests = len(self.test_results)
        passed_tests = status_counts.get('PASSED', 0)
        failed_tests = status_counts.get('FAILED', 0) + status_counts.get('ERROR', 0)
        partial_tests = status_counts.get('PARTIAL', 0)
        
        overall_status = 'PASSED'
        if failed_tests > 0:
            overall_status = 'FAILED'
        elif partial_tests > 0:
            overall_status = 'PARTIAL'
        
        return {
            'overall_status': overall_status,
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'partial': partial_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'status_breakdown': status_counts
        }
    
    def cleanup(self):
        """Cleanup test environment"""
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.info(f"Cleaned up test environment: {self.temp_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup test environment: {e}")


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Video Enhancement System Test Runner")
    parser.add_argument('--output', type=Path, 
                       help='Output file for test results (JSON)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Skip cleanup of test environment (for debugging)')
    
    args = parser.parse_args()
    
    test_runner = None
    
    try:
        # Initialize test runner
        test_runner = EnhancementTestRunner(verbose=args.verbose)
        
        # Run all tests
        results = test_runner.run_all_tests()
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Test results saved to: {args.output}")
        
        # Print summary
        summary = results['summary']
        print("\n" + "="*60)
        print("ENHANCEMENT SYSTEM TEST SUMMARY")
        print("="*60)
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Partial: {summary['partial']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Duration: {results['test_run_info']['duration_seconds']:.1f} seconds")
        
        if summary['overall_status'] == 'FAILED':
            print("\n❌ SOME TESTS FAILED")
            for test_name, test_result in results['test_results'].items():
                if test_result['status'] in ['FAILED', 'ERROR']:
                    print(f"\n{test_name.upper()} ERRORS:")
                    for error in test_result['errors']:
                        print(f"  • {error}")
        elif summary['overall_status'] == 'PARTIAL':
            print("\n[WARNING] SOME TESTS HAD ISSUES")
        else:
            print("\n[SUCCESS] ALL TESTS PASSED")
        
        print("="*60)
        
        # Exit with appropriate code
        if summary['overall_status'] == 'FAILED':
            sys.exit(1)
        elif summary['overall_status'] == 'PARTIAL':
            sys.exit(2)
        else:
            sys.exit(0)
    
    except Exception as e:
        print(f"Test runner failed: {e}")
        sys.exit(1)
    
    finally:
        # Cleanup unless requested not to
        if test_runner and not args.no_cleanup:
            test_runner.cleanup()


if __name__ == "__main__":
    main()