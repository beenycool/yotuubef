"""
Robust Error Handling and Task Queue System
Implements retry mechanisms, persistent queues, and improved configuration management
"""

import asyncio
import logging
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import time
import traceback
import functools

# Optional dependencies with fallbacks
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

from src.config.settings import get_config


class TaskStatus(Enum):
    """Status of tasks in the queue"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class RetryPolicy:
    """Retry policy configuration"""
    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 300.0  # Maximum delay in seconds
    exponential_backoff: bool = True
    jitter: bool = True  # Add random jitter to delay


@dataclass
class Task:
    """Task definition for the queue"""
    task_id: str
    task_type: str
    task_name: str
    task_data: Dict[str, Any]
    priority: TaskPriority
    status: TaskStatus
    retry_policy: RetryPolicy
    created_at: datetime
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    attempts: int = 0
    last_error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class PersistentTaskQueue:
    """
    Persistent task queue with SQLite backend for reliable task processing
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("data/task_queue.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self._init_database()
        
        # Task processors registry
        self.task_processors: Dict[str, Callable] = {}
        
        # Queue processing state
        self.is_processing = False
        self.processing_tasks = {}
        
        self.logger.info("Persistent Task Queue initialized")
    
    def _init_database(self):
        """Initialize SQLite database for task storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS tasks (
                        task_id TEXT PRIMARY KEY,
                        task_type TEXT NOT NULL,
                        task_name TEXT NOT NULL,
                        task_data TEXT NOT NULL,
                        priority INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        retry_policy TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL,
                        scheduled_at TIMESTAMP,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        attempts INTEGER DEFAULT 0,
                        last_error TEXT,
                        result TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_task_status ON tasks(status)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_task_priority ON tasks(priority, scheduled_at)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_task_type ON tasks(task_type)
                """)
                
                self.logger.info("Task queue database initialized")
                
        except Exception as e:
            self.logger.error(f"Task queue database initialization failed: {e}")
    
    def register_processor(self, task_type: str, processor_func: Callable):
        """Register a task processor function"""
        self.task_processors[task_type] = processor_func
        self.logger.info(f"Registered processor for task type: {task_type}")
    
    async def add_task(self, 
                      task_type: str,
                      task_name: str,
                      task_data: Dict[str, Any],
                      priority: TaskPriority = TaskPriority.NORMAL,
                      retry_policy: Optional[RetryPolicy] = None,
                      scheduled_at: Optional[datetime] = None) -> str:
        """
        Add a task to the queue
        
        Args:
            task_type: Type of task for processor selection
            task_name: Human-readable task name
            task_data: Task data payload
            priority: Task priority
            retry_policy: Custom retry policy (uses default if None)
            scheduled_at: When to execute the task (None for immediate)
            
        Returns:
            Task ID
        """
        try:
            task_id = f"{task_type}_{int(time.time() * 1000)}"
            
            if retry_policy is None:
                retry_policy = RetryPolicy()
            
            task = Task(
                task_id=task_id,
                task_type=task_type,
                task_name=task_name,
                task_data=task_data,
                priority=priority,
                status=TaskStatus.PENDING,
                retry_policy=retry_policy,
                created_at=datetime.now(),
                scheduled_at=scheduled_at
            )
            
            # Save to database
            await self._save_task(task)
            
            self.logger.info(f"Added task to queue: {task_id}")
            return task_id
            
        except Exception as e:
            self.logger.error(f"Failed to add task to queue: {e}")
            raise
    
    async def _save_task(self, task: Task):
        """Save task to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO tasks
                    (task_id, task_type, task_name, task_data, priority, status,
                     retry_policy, created_at, scheduled_at, started_at, completed_at,
                     attempts, last_error, result)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task.task_id,
                    task.task_type,
                    task.task_name,
                    json.dumps(task.task_data),
                    task.priority.value,
                    task.status.value,
                    json.dumps(asdict(task.retry_policy)),
                    task.created_at.isoformat(),
                    task.scheduled_at.isoformat() if task.scheduled_at else None,
                    task.started_at.isoformat() if task.started_at else None,
                    task.completed_at.isoformat() if task.completed_at else None,
                    task.attempts,
                    task.last_error,
                    json.dumps(task.result) if task.result else None
                ))
                
        except Exception as e:
            self.logger.error(f"Failed to save task: {e}")
    
    async def get_next_task(self) -> Optional[Task]:
        """Get the next task to process"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get highest priority pending task that's ready to run
                cursor.execute("""
                    SELECT task_id, task_type, task_name, task_data, priority, status,
                           retry_policy, created_at, scheduled_at, started_at, completed_at,
                           attempts, last_error, result
                    FROM tasks
                    WHERE status IN ('pending', 'retrying')
                    AND (scheduled_at IS NULL OR scheduled_at <= ?)
                    ORDER BY priority DESC, created_at ASC
                    LIMIT 1
                """, (datetime.now().isoformat(),))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Parse task data
                task = Task(
                    task_id=row[0],
                    task_type=row[1],
                    task_name=row[2],
                    task_data=json.loads(row[3]),
                    priority=TaskPriority(row[4]),
                    status=TaskStatus(row[5]),
                    retry_policy=RetryPolicy(**json.loads(row[6])),
                    created_at=datetime.fromisoformat(row[7]),
                    scheduled_at=datetime.fromisoformat(row[8]) if row[8] else None,
                    started_at=datetime.fromisoformat(row[9]) if row[9] else None,
                    completed_at=datetime.fromisoformat(row[10]) if row[10] else None,
                    attempts=row[11],
                    last_error=row[12],
                    result=json.loads(row[13]) if row[13] else None
                )
                
                return task
                
        except Exception as e:
            self.logger.error(f"Failed to get next task: {e}")
            return None
    
    async def process_task(self, task: Task) -> bool:
        """
        Process a single task with error handling and retries
        
        Args:
            task: Task to process
            
        Returns:
            Success status
        """
        try:
            self.logger.info(f"Processing task: {task.task_id}")
            
            # Check if processor exists
            if task.task_type not in self.task_processors:
                self.logger.error(f"No processor registered for task type: {task.task_type}")
                await self._mark_task_failed(task, "No processor registered")
                return False
            
            # Update task status
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            task.attempts += 1
            await self._save_task(task)
            
            # Track processing task
            self.processing_tasks[task.task_id] = task
            
            try:
                # Execute task processor
                processor = self.task_processors[task.task_type]
                
                if asyncio.iscoroutinefunction(processor):
                    result = await processor(task.task_data)
                else:
                    result = processor(task.task_data)
                
                # Mark task as completed
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.result = result
                await self._save_task(task)
                
                self.logger.info(f"Task completed successfully: {task.task_id}")
                return True
                
            except Exception as e:
                # Handle task execution error
                error_msg = f"Task execution failed: {str(e)}"
                self.logger.error(f"Task {task.task_id} failed: {error_msg}")
                
                # Check if we should retry
                if task.attempts < task.retry_policy.max_attempts:
                    await self._schedule_retry(task, error_msg)
                    return False
                else:
                    await self._mark_task_failed(task, error_msg)
                    return False
            
            finally:
                # Remove from processing tasks
                self.processing_tasks.pop(task.task_id, None)
                
        except Exception as e:
            self.logger.error(f"Task processing system error: {e}")
            return False
    
    async def _schedule_retry(self, task: Task, error_msg: str):
        """Schedule a task for retry with exponential backoff"""
        try:
            task.status = TaskStatus.RETRYING
            task.last_error = error_msg
            
            # Calculate retry delay
            delay = self._calculate_retry_delay(task)
            
            # Schedule next attempt
            task.scheduled_at = datetime.now() + timedelta(seconds=delay)
            
            await self._save_task(task)
            
            self.logger.info(f"Scheduled retry for task {task.task_id} in {delay:.1f}s (attempt {task.attempts}/{task.retry_policy.max_attempts})")
            
        except Exception as e:
            self.logger.error(f"Failed to schedule retry: {e}")
    
    def _calculate_retry_delay(self, task: Task) -> float:
        """Calculate retry delay with exponential backoff and jitter"""
        retry_policy = task.retry_policy
        
        if retry_policy.exponential_backoff:
            # Exponential backoff: base_delay * (2 ^ (attempts - 1))
            delay = retry_policy.base_delay * (2 ** (task.attempts - 1))
        else:
            # Fixed delay
            delay = retry_policy.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, retry_policy.max_delay)
        
        # Add jitter to prevent thundering herd
        if retry_policy.jitter:
            import random
            jitter = random.uniform(0.8, 1.2)
            delay *= jitter
        
        return delay
    
    async def _mark_task_failed(self, task: Task, error_msg: str):
        """Mark a task as permanently failed"""
        task.status = TaskStatus.FAILED
        task.completed_at = datetime.now()
        task.last_error = error_msg
        await self._save_task(task)
        
        self.logger.error(f"Task permanently failed: {task.task_id} - {error_msg}")
    
    async def start_processing(self, max_concurrent: int = 5):
        """Start processing tasks from the queue"""
        self.is_processing = True
        self.logger.info(f"Starting task queue processing (max_concurrent: {max_concurrent})")
        
        try:
            while self.is_processing:
                # Get current number of processing tasks
                current_processing = len(self.processing_tasks)
                
                if current_processing < max_concurrent:
                    # Get next task
                    task = await self.get_next_task()
                    
                    if task:
                        # Process task asynchronously
                        asyncio.create_task(self.process_task(task))
                    else:
                        # No tasks available, wait a bit
                        await asyncio.sleep(1)
                else:
                    # Too many concurrent tasks, wait
                    await asyncio.sleep(0.5)
                    
        except Exception as e:
            self.logger.error(f"Task processing loop error: {e}")
        finally:
            self.is_processing = False
    
    def stop_processing(self):
        """Stop processing tasks"""
        self.is_processing = False
        self.logger.info("Stopping task queue processing")
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status and statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count tasks by status
                cursor.execute("""
                    SELECT status, COUNT(*) 
                    FROM tasks 
                    GROUP BY status
                """)
                
                status_counts = {}
                for status, count in cursor.fetchall():
                    status_counts[status] = count
                
                # Get queue length (pending tasks)
                pending_count = status_counts.get('pending', 0) + status_counts.get('retrying', 0)
                
                # Get oldest pending task
                cursor.execute("""
                    SELECT MIN(created_at) 
                    FROM tasks 
                    WHERE status IN ('pending', 'retrying')
                """)
                oldest_pending = cursor.fetchone()[0]
                
                return {
                    'queue_length': pending_count,
                    'status_counts': status_counts,
                    'processing_count': len(self.processing_tasks),
                    'is_processing': self.is_processing,
                    'oldest_pending': oldest_pending,
                    'registered_processors': list(self.task_processors.keys())
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get queue status: {e}")
            return {}
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE tasks 
                    SET status = 'cancelled', completed_at = ?
                    WHERE task_id = ? AND status IN ('pending', 'retrying')
                """, (datetime.now().isoformat(), task_id))
                
                if cursor.rowcount > 0:
                    self.logger.info(f"Cancelled task: {task_id}")
                    return True
                else:
                    self.logger.warning(f"Task not found or not cancellable: {task_id}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to cancel task: {e}")
            return False


class RobustRetryHandler:
    """
    Decorator and utility for adding robust retry logic to functions
    """
    
    @staticmethod
    def with_retry(
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 300.0,
        exponential_backoff: bool = True,
        jitter: bool = True,
        exceptions: Tuple = (Exception,),
        on_retry: Optional[Callable] = None
    ):
        """
        Decorator to add retry logic to functions
        
        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay between retries
            max_delay: Maximum delay between retries
            exponential_backoff: Use exponential backoff
            jitter: Add random jitter to delays
            exceptions: Tuple of exception types to retry on
            on_retry: Callback function called on each retry
        """
        def decorator(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_attempts):
                    try:
                        if asyncio.iscoroutinefunction(func):
                            return await func(*args, **kwargs)
                        else:
                            return func(*args, **kwargs)
                            
                    except exceptions as e:
                        last_exception = e
                        
                        if attempt == max_attempts - 1:
                            # Last attempt, raise the exception
                            raise e
                        
                        # Calculate delay
                        if exponential_backoff:
                            delay = base_delay * (2 ** attempt)
                        else:
                            delay = base_delay
                        
                        delay = min(delay, max_delay)
                        
                        if jitter:
                            import random
                            delay *= random.uniform(0.8, 1.2)
                        
                        # Call retry callback if provided
                        if on_retry:
                            try:
                                if asyncio.iscoroutinefunction(on_retry):
                                    await on_retry(attempt + 1, e, delay)
                                else:
                                    on_retry(attempt + 1, e, delay)
                            except Exception:
                                pass  # Ignore callback errors
                        
                        # Wait before retrying
                        await asyncio.sleep(delay)
                
                # This should never be reached, but just in case
                raise last_exception
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                        
                    except exceptions as e:
                        last_exception = e
                        
                        if attempt == max_attempts - 1:
                            raise e
                        
                        # Calculate delay
                        if exponential_backoff:
                            delay = base_delay * (2 ** attempt)
                        else:
                            delay = base_delay
                        
                        delay = min(delay, max_delay)
                        
                        if jitter:
                            import random
                            delay *= random.uniform(0.8, 1.2)
                        
                        # Call retry callback if provided
                        if on_retry:
                            try:
                                on_retry(attempt + 1, e, delay)
                            except Exception:
                                pass
                        
                        # Wait before retrying
                        time.sleep(delay)
                
                raise last_exception
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator


class EnhancedConfigManager:
    """
    Enhanced configuration manager with environment variable support and validation
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path("config.yaml")
        self.env_file = Path(".env")
        
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        if DOTENV_AVAILABLE and self.env_file.exists():
            load_dotenv(self.env_file)
            self.logger.info("Loaded environment variables from .env file")
        
        # Configuration cache
        self._config_cache = {}
        self._last_loaded = None
        
        # Configuration validation rules
        self.validation_rules = {
            'YOUTUBE_API_KEY': {'required': True, 'type': str, 'min_length': 10},
            'REDDIT_CLIENT_ID': {'required': True, 'type': str, 'min_length': 5},
            'REDDIT_CLIENT_SECRET': {'required': True, 'type': str, 'min_length': 10},
            'GEMINI_API_KEY': {'required': False, 'type': str, 'min_length': 10},
            'PEXELS_API_KEY': {'required': False, 'type': str, 'min_length': 10},
            'UNSPLASH_API_KEY': {'required': False, 'type': str, 'min_length': 10},
            'PIXABAY_API_KEY': {'required': False, 'type': str, 'min_length': 10},
            'MAX_VIDEOS_PER_DAY': {'required': False, 'type': int, 'min_value': 1, 'max_value': 50},
            'VIDEO_GENERATION_INTERVAL': {'required': False, 'type': int, 'min_value': 300},
        }
        
        self.logger.info("Enhanced Config Manager initialized")
    
    def get_config(self, reload: bool = False) -> Dict[str, Any]:
        """Get configuration with caching and validation"""
        try:
            # Check if we need to reload
            if reload or not self._config_cache or self._needs_reload():
                self._load_config()
            
            return self._config_cache.copy()
            
        except Exception as e:
            self.logger.error(f"Failed to get configuration: {e}")
            return {}
    
    def _needs_reload(self) -> bool:
        """Check if configuration needs to be reloaded"""
        if not self._last_loaded:
            return True
        
        # Check if config file has been modified
        if self.config_file.exists():
            file_mtime = datetime.fromtimestamp(self.config_file.stat().st_mtime)
            return file_mtime > self._last_loaded
        
        return False
    
    def _load_config(self):
        """Load configuration from file and environment"""
        try:
            config = {}
            
            # Load from YAML file if exists
            if self.config_file.exists():
                try:
                    import yaml
                    with open(self.config_file, 'r') as f:
                        file_config = yaml.safe_load(f) or {}
                    config.update(file_config)
                    self.logger.info(f"Loaded configuration from {self.config_file}")
                except ImportError:
                    self.logger.warning("PyYAML not available, skipping YAML config file")
                except Exception as e:
                    self.logger.warning(f"Failed to load YAML config: {e}")
            
            # Override with environment variables
            import os
            for key in self.validation_rules.keys():
                env_value = os.getenv(key)
                if env_value:
                    config[key] = env_value
            
            # Validate configuration
            validation_result = self._validate_config(config)
            if not validation_result['valid']:
                self.logger.warning(f"Configuration validation warnings: {validation_result['warnings']}")
                if validation_result['errors']:
                    self.logger.error(f"Configuration validation errors: {validation_result['errors']}")
            
            self._config_cache = config
            self._last_loaded = datetime.now()
            
            self.logger.info(f"Configuration loaded with {len(config)} settings")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            # Use empty config as fallback
            self._config_cache = {}
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration against rules"""
        errors = []
        warnings = []
        
        for key, rules in self.validation_rules.items():
            value = config.get(key)
            
            # Check required fields
            if rules.get('required', False) and not value:
                errors.append(f"Required configuration '{key}' is missing")
                continue
            
            if value is None:
                continue
            
            # Type validation
            expected_type = rules.get('type')
            if expected_type and not isinstance(value, expected_type):
                try:
                    # Try to convert
                    if expected_type == int:
                        config[key] = int(value)
                    elif expected_type == float:
                        config[key] = float(value)
                    elif expected_type == bool:
                        config[key] = str(value).lower() in ('true', '1', 'yes', 'on')
                    elif expected_type == str:
                        config[key] = str(value)
                except (ValueError, TypeError):
                    warnings.append(f"Configuration '{key}' has incorrect type, expected {expected_type.__name__}")
            
            # Value validation
            if 'min_length' in rules and isinstance(value, str):
                if len(value) < rules['min_length']:
                    warnings.append(f"Configuration '{key}' is too short (minimum {rules['min_length']} characters)")
            
            if 'min_value' in rules and isinstance(value, (int, float)):
                if value < rules['min_value']:
                    warnings.append(f"Configuration '{key}' is too small (minimum {rules['min_value']})")
            
            if 'max_value' in rules and isinstance(value, (int, float)):
                if value > rules['max_value']:
                    warnings.append(f"Configuration '{key}' is too large (maximum {rules['max_value']})")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a secret configuration value"""
        config = self.get_config()
        value = config.get(key, default)
        
        if value:
            # Don't log the actual value for security
            self.logger.debug(f"Retrieved secret configuration: {key}")
        else:
            self.logger.warning(f"Secret configuration not found: {key}")
        
        return value
    
    def set_config(self, key: str, value: Any, persist: bool = True):
        """Set a configuration value"""
        try:
            self._config_cache[key] = value
            
            if persist:
                self._save_config()
            
            self.logger.info(f"Updated configuration: {key}")
            
        except Exception as e:
            self.logger.error(f"Failed to set configuration: {e}")
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            if not self.config_file.parent.exists():
                self.config_file.parent.mkdir(parents=True)
            
            # Only save non-sensitive values
            safe_config = {}
            sensitive_keys = {'API_KEY', 'SECRET', 'TOKEN', 'PASSWORD'}
            
            for key, value in self._config_cache.items():
                # Skip sensitive values (they should be in environment variables)
                if not any(sensitive in key.upper() for sensitive in sensitive_keys):
                    safe_config[key] = value
            
            try:
                import yaml
                with open(self.config_file, 'w') as f:
                    yaml.dump(safe_config, f, default_flow_style=False)
                self.logger.info(f"Saved configuration to {self.config_file}")
            except ImportError:
                # Fallback to JSON if PyYAML not available
                with open(self.config_file.with_suffix('.json'), 'w') as f:
                    json.dump(safe_config, f, indent=2)
                self.logger.info(f"Saved configuration to {self.config_file.with_suffix('.json')}")
                
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
    
    def create_env_template(self):
        """Create a .env template file with required variables"""
        try:
            env_template_path = Path(".env.example")
            
            template_content = [
                "# YouTube Video Generator Environment Variables",
                "# Copy this file to .env and fill in your actual values",
                "",
                "# Required API Keys",
                "YOUTUBE_API_KEY=your_youtube_api_key_here",
                "REDDIT_CLIENT_ID=your_reddit_client_id_here",
                "REDDIT_CLIENT_SECRET=your_reddit_client_secret_here",
                "",
                "# Optional API Keys",
                "GEMINI_API_KEY=your_gemini_api_key_here",
                "PEXELS_API_KEY=your_pexels_api_key_here",
                "UNSPLASH_API_KEY=your_unsplash_api_key_here",
                "PIXABAY_API_KEY=your_pixabay_api_key_here",
                "",
                "# System Configuration",
                "MAX_VIDEOS_PER_DAY=8",
                "VIDEO_GENERATION_INTERVAL=3600",
                "",
                "# Database URLs (optional)",
                "DATABASE_URL=sqlite:///data/app.db",
                "",
                "# Logging Level",
                "LOG_LEVEL=INFO"
            ]
            
            with open(env_template_path, 'w') as f:
                f.write('\n'.join(template_content))
            
            self.logger.info(f"Created environment template: {env_template_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create environment template: {e}")


# Global instances
global_task_queue = PersistentTaskQueue()
global_config_manager = EnhancedConfigManager()