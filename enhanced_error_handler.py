#!/usr/bin/env python3
"""
Enhanced Error Handling and Logging System
Provides comprehensive error handling, logging, and debugging capabilities
"""

import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
from functools import wraps
import os


class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


class EnhancedErrorHandler:
    """Enhanced error handling with detailed logging and recovery strategies"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Error statistics
        self.error_stats = {
            'total_errors': 0,
            'error_types': {},
            'error_history': []
        }
        
        # Recovery strategies
        self.recovery_strategies = {
            'FileNotFoundError': self._handle_file_not_found,
            'ImportError': self._handle_import_error,
            'ModuleNotFoundError': self._handle_module_not_found,
            'ConnectionError': self._handle_connection_error,
            'TimeoutError': self._handle_timeout_error,
            'PermissionError': self._handle_permission_error,
            'MemoryError': self._handle_memory_error,
        }
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup comprehensive logging system"""
        # Create main logger
        self.logger = logging.getLogger('enhanced_youtube_generator')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for all logs
        log_file = self.log_dir / f"youtube_generator_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Error-specific handler
        error_log_file = self.log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
        
        # Debug handler for detailed debugging
        debug_log_file = self.log_dir / f"debug_{datetime.now().strftime('%Y%m%d')}.log"
        debug_handler = logging.FileHandler(debug_log_file, encoding='utf-8')
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(file_formatter)
        self.logger.addHandler(debug_handler)
    
    def handle_error(self, error: Exception, context: str = "", recovery_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Handle errors with recovery strategies and detailed logging
        
        Args:
            error: The exception that occurred
            context: Context information about where the error occurred
            recovery_data: Additional data for recovery strategies
            
        Returns:
            Dictionary with error information and recovery results
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Update error statistics
        self.error_stats['total_errors'] += 1
        self.error_stats['error_types'][error_type] = self.error_stats['error_types'].get(error_type, 0) + 1
        
        # Create error record
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_msg,
            'context': context,
            'traceback': traceback.format_exc(),
            'recovery_attempted': False,
            'recovery_successful': False,
            'recovery_message': None
        }
        
        # Log the error
        self.logger.error(f"Error in {context}: {error_type}: {error_msg}")
        self.logger.debug(f"Full traceback: {traceback.format_exc()}")
        
        # Attempt recovery if strategy exists
        if error_type in self.recovery_strategies:
            try:
                recovery_result = self.recovery_strategies[error_type](error, context, recovery_data)
                error_record['recovery_attempted'] = True
                error_record['recovery_successful'] = recovery_result.get('success', False)
                error_record['recovery_message'] = recovery_result.get('message', '')
                
                if recovery_result.get('success'):
                    self.logger.info(f"Recovery successful for {error_type}: {recovery_result.get('message')}")
                else:
                    self.logger.warning(f"Recovery failed for {error_type}: {recovery_result.get('message')}")
                    
            except Exception as recovery_error:
                self.logger.error(f"Recovery strategy failed: {recovery_error}")
                error_record['recovery_message'] = f"Recovery strategy failed: {recovery_error}"
        
        # Add to error history
        self.error_stats['error_history'].append(error_record)
        
        # Save error record to file
        error_file = self.log_dir / f"error_details_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(error_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(error_record) + '\n')
        
        return error_record
    
    def _handle_file_not_found(self, error: Exception, context: str, recovery_data: Optional[Dict]) -> Dict[str, Any]:
        """Handle file not found errors"""
        if recovery_data and 'alternative_paths' in recovery_data:
            for alt_path in recovery_data['alternative_paths']:
                if Path(alt_path).exists():
                    return {
                        'success': True,
                        'message': f"Found alternative file: {alt_path}",
                        'alternative_path': alt_path
                    }
        
        # Try to create directory if it's a directory issue
        if recovery_data and 'create_directory' in recovery_data:
            try:
                Path(recovery_data['create_directory']).mkdir(parents=True, exist_ok=True)
                return {
                    'success': True,
                    'message': f"Created directory: {recovery_data['create_directory']}"
                }
            except Exception as e:
                return {
                    'success': False,
                    'message': f"Failed to create directory: {e}"
                }
        
        return {
            'success': False,
            'message': f"No recovery strategy available for missing file: {error}"
        }
    
    def _handle_import_error(self, error: Exception, context: str, recovery_data: Optional[Dict]) -> Dict[str, Any]:
        """Handle import errors"""
        missing_module = str(error).replace("No module named '", "").replace("'", "")
        
        suggestions = {
            'numpy': 'pip install numpy',
            'cv2': 'pip install opencv-python',
            'moviepy': 'pip install moviepy',
            'psutil': 'pip install psutil',
            'asyncpraw': 'pip install asyncpraw',
            'google-generativeai': 'pip install google-generativeai',
            'elevenlabs': 'pip install elevenlabs',
            'yt_dlp': 'pip install yt-dlp',
            'pydub': 'pip install pydub',
            'scipy': 'pip install scipy',
            'PIL': 'pip install Pillow',
            'torch': 'pip install torch',
            'transformers': 'pip install transformers',
        }
        
        if missing_module in suggestions:
            return {
                'success': False,  # Can't auto-install, but provide guidance
                'message': f"To fix this error, run: {suggestions[missing_module]}",
                'install_command': suggestions[missing_module]
            }
        
        return {
            'success': False,
            'message': f"Unknown module '{missing_module}'. Please install manually or check the module name."
        }
    
    def _handle_module_not_found(self, error: Exception, context: str, recovery_data: Optional[Dict]) -> Dict[str, Any]:
        """Handle module not found errors (similar to import errors)"""
        return self._handle_import_error(error, context, recovery_data)
    
    def _handle_connection_error(self, error: Exception, context: str, recovery_data: Optional[Dict]) -> Dict[str, Any]:
        """Handle connection errors"""
        return {
            'success': False,
            'message': "Connection failed. Please check your internet connection and try again later.",
            'suggestion': "Consider using offline mode or cached data if available."
        }
    
    def _handle_timeout_error(self, error: Exception, context: str, recovery_data: Optional[Dict]) -> Dict[str, Any]:
        """Handle timeout errors"""
        return {
            'success': False,
            'message': "Operation timed out. Try increasing timeout duration or checking network connection.",
            'suggestion': "Consider breaking large operations into smaller chunks."
        }
    
    def _handle_permission_error(self, error: Exception, context: str, recovery_data: Optional[Dict]) -> Dict[str, Any]:
        """Handle permission errors"""
        return {
            'success': False,
            'message': "Permission denied. Please check file/directory permissions.",
            'suggestion': "Try running with appropriate permissions or change file ownership."
        }
    
    def _handle_memory_error(self, error: Exception, context: str, recovery_data: Optional[Dict]) -> Dict[str, Any]:
        """Handle memory errors"""
        return {
            'success': False,
            'message': "Out of memory. Try reducing batch size or closing other applications.",
            'suggestion': "Consider using streaming processing or lower resolution settings."
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        return {
            'total_errors': self.error_stats['total_errors'],
            'error_types': self.error_stats['error_types'],
            'most_common_error': max(self.error_stats['error_types'].items(), key=lambda x: x[1])[0] if self.error_stats['error_types'] else None,
            'recent_errors': self.error_stats['error_history'][-10:] if self.error_stats['error_history'] else [],
            'error_rate': len([e for e in self.error_stats['error_history'] if 
                             datetime.fromisoformat(e['timestamp']) > datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)])
        }
    
    def generate_error_report(self) -> str:
        """Generate a comprehensive error report"""
        stats = self.get_error_statistics()
        
        report = f"""
🚨 Error Analysis Report
{'='*50}

📊 Error Statistics:
   Total Errors: {stats['total_errors']}
   Today's Errors: {stats['error_rate']}
   Most Common Error: {stats['most_common_error'] or 'None'}

📈 Error Types:
"""
        
        for error_type, count in sorted(stats['error_types'].items(), key=lambda x: x[1], reverse=True):
            report += f"   • {error_type}: {count} occurrences\n"
        
        report += f"""
🔄 Recovery Success Rate:
"""
        
        total_with_recovery = len([e for e in self.error_stats['error_history'] if e['recovery_attempted']])
        successful_recoveries = len([e for e in self.error_stats['error_history'] if e['recovery_successful']])
        
        if total_with_recovery > 0:
            success_rate = (successful_recoveries / total_with_recovery) * 100
            report += f"   • Recovery Attempts: {total_with_recovery}\n"
            report += f"   • Successful Recoveries: {successful_recoveries}\n"
            report += f"   • Success Rate: {success_rate:.1f}%\n"
        else:
            report += "   • No recovery attempts made\n"
        
        report += f"""
📋 Recent Errors:
"""
        
        for error in stats['recent_errors']:
            timestamp = datetime.fromisoformat(error['timestamp']).strftime('%H:%M:%S')
            report += f"   • {timestamp} - {error['error_type']}: {error['error_message'][:50]}...\n"
        
        report += f"""
💡 Recommendations:
   • Install missing dependencies (see individual error details)
   • Check file permissions and paths
   • Verify internet connection for API calls
   • Monitor system resources (RAM, disk space)
   • Review configuration files for correctness

📁 Log Files:
   • Main Log: {self.log_dir}/youtube_generator_{datetime.now().strftime('%Y%m%d')}.log
   • Error Log: {self.log_dir}/errors_{datetime.now().strftime('%Y%m%d')}.log
   • Debug Log: {self.log_dir}/debug_{datetime.now().strftime('%Y%m%d')}.log

{'='*50}
"""
        
        return report


def error_handler(context: str = ""):
    """Decorator for automatic error handling"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if hasattr(wrapper, '_error_handler'):
                    wrapper._error_handler.handle_error(e, context or func.__name__)
                else:
                    # Create a default error handler if none exists
                    default_handler = EnhancedErrorHandler()
                    default_handler.handle_error(e, context or func.__name__)
                raise
        return wrapper
    return decorator


def setup_global_error_handler():
    """Setup global error handler for the application"""
    handler = EnhancedErrorHandler()
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        handler.handle_error(exc_value, "Global Exception Handler")
    
    sys.excepthook = handle_exception
    return handler


def main():
    """Demonstration of enhanced error handling"""
    
    # Setup global error handler
    error_handler = setup_global_error_handler()
    
    print("🚀 Enhanced Error Handling System Demo")
    print("=" * 50)
    
    # Test various error scenarios
    test_errors = [
        (FileNotFoundError("test_file.txt"), "File operation", {'alternative_paths': ['./test_file.txt', './data/test_file.txt']}),
        (ImportError("No module named 'numpy'"), "Module import", None),
        (ConnectionError("Connection failed"), "API call", None),
        (PermissionError("Permission denied"), "File access", None),
    ]
    
    print("🧪 Testing error handling scenarios...")
    for error, context, recovery_data in test_errors:
        print(f"\n📋 Testing: {type(error).__name__}")
        result = error_handler.handle_error(error, context, recovery_data)
        print(f"   Recovery attempted: {result['recovery_attempted']}")
        if result['recovery_attempted']:
            print(f"   Recovery successful: {result['recovery_successful']}")
            print(f"   Message: {result['recovery_message']}")
    
    # Generate and display error report
    print("\n📊 Error Report:")
    report = error_handler.generate_error_report()
    print(report)
    
    # Save report to file
    report_file = Path("error_analysis_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 Error report saved to: {report_file}")


if __name__ == "__main__":
    main()