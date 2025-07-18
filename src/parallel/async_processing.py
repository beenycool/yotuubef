"""
Parallel Processing System
Implements asynchronous operations and decoupled generation/uploading for efficiency
"""

import asyncio
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import concurrent.futures
import multiprocessing
import threading
from queue import Queue, Empty

# Optional dependencies with fallbacks
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

try:
    import aiodns
    AIODNS_AVAILABLE = True
except ImportError:
    AIODNS_AVAILABLE = False

from src.config.settings import get_config


class WorkerType(Enum):
    """Types of workers in the parallel system"""
    CONTENT_ANALYZER = "content_analyzer"
    VIDEO_GENERATOR = "video_generator"
    VIDEO_PROCESSOR = "video_processor"
    UPLOADER = "uploader"
    ASSET_DOWNLOADER = "asset_downloader"


class WorkerStatus(Enum):
    """Status of workers"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class WorkItem:
    """Work item for processing"""
    item_id: str
    worker_type: WorkerType
    priority: int
    data: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class Worker:
    """Worker definition"""
    worker_id: str
    worker_type: WorkerType
    status: WorkerStatus
    current_item: Optional[WorkItem] = None
    items_processed: int = 0
    last_activity: Optional[datetime] = None
    performance_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}


class AsyncWorkQueue:
    """
    Asynchronous work queue for parallel processing
    """
    
    def __init__(self, max_size: int = 1000):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.priority_queue = asyncio.PriorityQueue(maxsize=max_size)
        self.results = {}
        self.logger = logging.getLogger(__name__)
    
    async def put(self, item: WorkItem, use_priority: bool = True):
        """Add work item to queue"""
        try:
            if use_priority:
                # Higher priority = lower number (processed first)
                priority = -item.priority
                await self.priority_queue.put((priority, item.item_id, item))
            else:
                await self.queue.put(item)
            
            self.logger.debug(f"Added work item to queue: {item.item_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to add work item to queue: {e}")
    
    async def get(self, use_priority: bool = True) -> Optional[WorkItem]:
        """Get work item from queue"""
        try:
            if use_priority:
                if not self.priority_queue.empty():
                    priority, item_id, item = await self.priority_queue.get()
                    return item
            else:
                if not self.queue.empty():
                    return await self.queue.get()
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get work item from queue: {e}")
            return None
    
    async def get_with_timeout(self, timeout: float = 1.0, use_priority: bool = True) -> Optional[WorkItem]:
        """Get work item with timeout"""
        try:
            if use_priority:
                priority, item_id, item = await asyncio.wait_for(
                    self.priority_queue.get(), timeout=timeout
                )
                return item
            else:
                return await asyncio.wait_for(
                    self.queue.get(), timeout=timeout
                )
                
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            self.logger.error(f"Failed to get work item with timeout: {e}")
            return None
    
    def put_result(self, item_id: str, result: Any):
        """Store result for an item"""
        self.results[item_id] = result
    
    def get_result(self, item_id: str) -> Any:
        """Get result for an item"""
        return self.results.get(item_id)
    
    def qsize(self, use_priority: bool = True) -> int:
        """Get queue size"""
        if use_priority:
            return self.priority_queue.qsize()
        else:
            return self.queue.qsize()


class ParallelProcessingManager:
    """
    Manages parallel processing with multiple worker types and intelligent load balancing
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Work queues for different worker types
        self.work_queues = {
            worker_type: AsyncWorkQueue() for worker_type in WorkerType
        }
        
        # Worker pools
        self.workers = {}
        self.worker_tasks = {}
        
        # Processing control
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Performance tracking
        self.performance_stats = {
            'items_processed': 0,
            'items_failed': 0,
            'average_processing_time': 0.0,
            'throughput_per_minute': 0.0,
            'start_time': datetime.now()
        }
        
        # Worker processors registry
        self.worker_processors = {}
        
        self.logger.info("Parallel Processing Manager initialized")
    
    def register_processor(self, worker_type: WorkerType, processor_func: Callable):
        """Register a processor function for a worker type"""
        self.worker_processors[worker_type] = processor_func
        self.logger.info(f"Registered processor for worker type: {worker_type.value}")
    
    async def start_workers(self, worker_configs: Dict[WorkerType, int]):
        """
        Start worker pools
        
        Args:
            worker_configs: Dictionary mapping worker types to number of workers
        """
        try:
            self.is_running = True
            self.logger.info("Starting parallel processing workers")
            
            for worker_type, count in worker_configs.items():
                if worker_type not in self.worker_processors:
                    self.logger.warning(f"No processor registered for {worker_type.value}, skipping")
                    continue
                
                # Create workers for this type
                workers = []
                for i in range(count):
                    worker_id = f"{worker_type.value}_{i}"
                    worker = Worker(
                        worker_id=worker_id,
                        worker_type=worker_type,
                        status=WorkerStatus.IDLE
                    )
                    workers.append(worker)
                
                self.workers[worker_type] = workers
                
                # Start worker tasks
                tasks = []
                for worker in workers:
                    task = asyncio.create_task(self._worker_loop(worker))
                    tasks.append(task)
                
                self.worker_tasks[worker_type] = tasks
                
                self.logger.info(f"Started {count} workers for {worker_type.value}")
            
            # Start performance monitoring
            asyncio.create_task(self._performance_monitor_loop())
            
        except Exception as e:
            self.logger.error(f"Failed to start workers: {e}")
            raise
    
    async def _worker_loop(self, worker: Worker):
        """Main loop for a worker"""
        try:
            self.logger.info(f"Worker {worker.worker_id} started")
            
            while self.is_running and not self.shutdown_event.is_set():
                try:
                    # Get work item from queue
                    work_item = await self.work_queues[worker.worker_type].get_with_timeout(
                        timeout=1.0
                    )
                    
                    if work_item:
                        await self._process_work_item(worker, work_item)
                    else:
                        # No work available, update status
                        worker.status = WorkerStatus.IDLE
                        worker.last_activity = datetime.now()
                
                except Exception as e:
                    self.logger.error(f"Worker {worker.worker_id} error: {e}")
                    worker.status = WorkerStatus.ERROR
                    await asyncio.sleep(5)  # Error recovery delay
            
            worker.status = WorkerStatus.STOPPED
            self.logger.info(f"Worker {worker.worker_id} stopped")
            
        except Exception as e:
            self.logger.error(f"Worker {worker.worker_id} crashed: {e}")
            worker.status = WorkerStatus.ERROR
    
    async def _process_work_item(self, worker: Worker, work_item: WorkItem):
        """Process a work item with a worker"""
        try:
            worker.status = WorkerStatus.BUSY
            worker.current_item = work_item
            worker.last_activity = datetime.now()
            
            work_item.started_at = datetime.now()
            
            self.logger.debug(f"Worker {worker.worker_id} processing item {work_item.item_id}")
            
            # Get processor function
            processor = self.worker_processors[worker.worker_type]
            
            # Execute processor
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(processor):
                result = await processor(work_item.data)
            else:
                # Run synchronous function in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, processor, work_item.data)
            
            processing_time = time.time() - start_time
            
            # Update work item
            work_item.completed_at = datetime.now()
            work_item.result = result
            
            # Store result
            self.work_queues[worker.worker_type].put_result(work_item.item_id, result)
            
            # Update worker stats
            worker.items_processed += 1
            worker.current_item = None
            worker.status = WorkerStatus.IDLE
            
            # Update performance metrics
            if 'processing_times' not in worker.performance_metrics:
                worker.performance_metrics['processing_times'] = []
            
            worker.performance_metrics['processing_times'].append(processing_time)
            
            # Keep only last 100 processing times for performance
            if len(worker.performance_metrics['processing_times']) > 100:
                worker.performance_metrics['processing_times'] = worker.performance_metrics['processing_times'][-100:]
            
            # Update global stats
            self.performance_stats['items_processed'] += 1
            
            self.logger.debug(f"Worker {worker.worker_id} completed item {work_item.item_id} in {processing_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Worker {worker.worker_id} failed to process item {work_item.item_id}: {e}")
            
            # Update work item with error
            work_item.completed_at = datetime.now()
            work_item.error = str(e)
            
            # Update worker
            worker.current_item = None
            worker.status = WorkerStatus.IDLE
            
            # Update global stats
            self.performance_stats['items_failed'] += 1
    
    async def submit_work(self, 
                         worker_type: WorkerType,
                         item_data: Dict[str, Any],
                         priority: int = 0,
                         item_id: Optional[str] = None) -> str:
        """
        Submit work item for processing
        
        Args:
            worker_type: Type of worker to process the item
            item_data: Data for processing
            priority: Priority level (higher = more important)
            item_id: Optional custom item ID
            
        Returns:
            Item ID for result retrieval
        """
        try:
            if not item_id:
                item_id = f"{worker_type.value}_{int(time.time() * 1000)}"
            
            work_item = WorkItem(
                item_id=item_id,
                worker_type=worker_type,
                priority=priority,
                data=item_data,
                created_at=datetime.now()
            )
            
            await self.work_queues[worker_type].put(work_item)
            
            self.logger.debug(f"Submitted work item: {item_id}")
            return item_id
            
        except Exception as e:
            self.logger.error(f"Failed to submit work item: {e}")
            raise
    
    async def get_result(self, item_id: str, worker_type: WorkerType, timeout: float = 30.0) -> Optional[Any]:
        """
        Get result for a work item
        
        Args:
            item_id: Item ID
            worker_type: Worker type that processed the item
            timeout: Timeout in seconds
            
        Returns:
            Result or None if not available
        """
        try:
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                result = self.work_queues[worker_type].get_result(item_id)
                if result is not None:
                    return result
                
                await asyncio.sleep(0.1)
            
            self.logger.warning(f"Timeout waiting for result: {item_id}")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get result: {e}")
            return None
    
    async def submit_and_wait(self,
                            worker_type: WorkerType,
                            item_data: Dict[str, Any],
                            priority: int = 0,
                            timeout: float = 30.0) -> Optional[Any]:
        """Submit work and wait for result"""
        item_id = await self.submit_work(worker_type, item_data, priority)
        return await self.get_result(item_id, worker_type, timeout)
    
    async def submit_batch(self,
                          worker_type: WorkerType,
                          items_data: List[Dict[str, Any]],
                          priority: int = 0) -> List[str]:
        """Submit multiple work items as a batch"""
        item_ids = []
        
        for item_data in items_data:
            item_id = await self.submit_work(worker_type, item_data, priority)
            item_ids.append(item_id)
        
        return item_ids
    
    async def get_batch_results(self,
                              item_ids: List[str],
                              worker_type: WorkerType,
                              timeout: float = 60.0) -> List[Any]:
        """Get results for multiple work items"""
        tasks = []
        
        for item_id in item_ids:
            task = asyncio.create_task(
                self.get_result(item_id, worker_type, timeout)
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def process_batch_parallel(self,
                                   worker_type: WorkerType,
                                   items_data: List[Dict[str, Any]],
                                   priority: int = 0,
                                   timeout: float = 60.0) -> List[Any]:
        """Process multiple items in parallel and return results"""
        item_ids = await self.submit_batch(worker_type, items_data, priority)
        return await self.get_batch_results(item_ids, worker_type, timeout)
    
    async def _performance_monitor_loop(self):
        """Monitor and update performance statistics"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Update every minute
                
                # Calculate throughput
                elapsed_minutes = (datetime.now() - self.performance_stats['start_time']).total_seconds() / 60
                if elapsed_minutes > 0:
                    self.performance_stats['throughput_per_minute'] = (
                        self.performance_stats['items_processed'] / elapsed_minutes
                    )
                
                # Calculate average processing time
                all_processing_times = []
                for worker_type, workers in self.workers.items():
                    for worker in workers:
                        times = worker.performance_metrics.get('processing_times', [])
                        all_processing_times.extend(times)
                
                if all_processing_times:
                    self.performance_stats['average_processing_time'] = sum(all_processing_times) / len(all_processing_times)
                
                self.logger.debug(f"Performance: {self.performance_stats['throughput_per_minute']:.1f} items/min, "
                                f"avg time: {self.performance_stats['average_processing_time']:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Performance monitor error: {e}")
    
    def get_worker_status(self) -> Dict[str, Any]:
        """Get status of all workers"""
        status = {}
        
        for worker_type, workers in self.workers.items():
            worker_status = []
            
            for worker in workers:
                worker_info = {
                    'worker_id': worker.worker_id,
                    'status': worker.status.value,
                    'items_processed': worker.items_processed,
                    'current_item': worker.current_item.item_id if worker.current_item else None,
                    'last_activity': worker.last_activity.isoformat() if worker.last_activity else None
                }
                
                # Add performance metrics
                if worker.performance_metrics.get('processing_times'):
                    times = worker.performance_metrics['processing_times']
                    worker_info['avg_processing_time'] = sum(times) / len(times)
                    worker_info['min_processing_time'] = min(times)
                    worker_info['max_processing_time'] = max(times)
                
                worker_status.append(worker_info)
            
            status[worker_type.value] = {
                'workers': worker_status,
                'queue_size': self.work_queues[worker_type].qsize(),
                'total_workers': len(workers),
                'active_workers': len([w for w in workers if w.status == WorkerStatus.BUSY]),
                'idle_workers': len([w for w in workers if w.status == WorkerStatus.IDLE])
            }
        
        return status
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics"""
        return {
            **self.performance_stats,
            'uptime_minutes': (datetime.now() - self.performance_stats['start_time']).total_seconds() / 60,
            'success_rate': (
                self.performance_stats['items_processed'] / 
                (self.performance_stats['items_processed'] + self.performance_stats['items_failed'])
                if (self.performance_stats['items_processed'] + self.performance_stats['items_failed']) > 0 else 1.0
            )
        }
    
    async def shutdown(self):
        """Shutdown all workers gracefully"""
        try:
            self.logger.info("Shutting down parallel processing manager")
            
            self.is_running = False
            self.shutdown_event.set()
            
            # Wait for all worker tasks to complete
            all_tasks = []
            for tasks in self.worker_tasks.values():
                all_tasks.extend(tasks)
            
            if all_tasks:
                await asyncio.gather(*all_tasks, return_exceptions=True)
            
            self.logger.info("Parallel processing manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


class AsyncPipeline:
    """
    Asynchronous processing pipeline for chaining operations
    """
    
    def __init__(self, name: str):
        self.name = name
        self.stages = []
        self.logger = logging.getLogger(__name__)
    
    def add_stage(self, stage_func: Callable, stage_name: str = None):
        """Add a processing stage to the pipeline"""
        if not stage_name:
            stage_name = getattr(stage_func, '__name__', f'stage_{len(self.stages)}')
        
        self.stages.append({
            'name': stage_name,
            'function': stage_func,
            'is_async': asyncio.iscoroutinefunction(stage_func)
        })
        
        self.logger.info(f"Added stage '{stage_name}' to pipeline '{self.name}'")
    
    async def process(self, input_data: Any) -> Any:
        """Process data through the pipeline"""
        try:
            data = input_data
            
            for i, stage in enumerate(self.stages):
                start_time = time.time()
                
                try:
                    if stage['is_async']:
                        data = await stage['function'](data)
                    else:
                        # Run sync function in thread pool
                        loop = asyncio.get_event_loop()
                        data = await loop.run_in_executor(None, stage['function'], data)
                    
                    processing_time = time.time() - start_time
                    self.logger.debug(f"Pipeline '{self.name}' stage '{stage['name']}' completed in {processing_time:.2f}s")
                    
                except Exception as e:
                    self.logger.error(f"Pipeline '{self.name}' stage '{stage['name']}' failed: {e}")
                    raise
            
            return data
            
        except Exception as e:
            self.logger.error(f"Pipeline '{self.name}' processing failed: {e}")
            raise
    
    async def process_batch(self, input_items: List[Any], max_concurrent: int = 5) -> List[Any]:
        """Process multiple items through the pipeline concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_item(item):
            async with semaphore:
                return await self.process(item)
        
        tasks = [asyncio.create_task(process_item(item)) for item in input_items]
        return await asyncio.gather(*tasks, return_exceptions=True)


class ContentGenerationPipeline:
    """
    Specialized pipeline for content generation with decoupled generation and uploading
    """
    
    def __init__(self, parallel_manager: ParallelProcessingManager):
        self.parallel_manager = parallel_manager
        self.logger = logging.getLogger(__name__)
        
        # Pipeline queues
        self.content_queue = asyncio.Queue()
        self.generation_queue = asyncio.Queue()
        self.upload_queue = asyncio.Queue()
        
        # Processing control
        self.is_running = False
    
    async def start_pipeline(self):
        """Start the content generation pipeline"""
        self.is_running = True
        self.logger.info("Starting content generation pipeline")
        
        # Start pipeline stages
        asyncio.create_task(self._content_analysis_stage())
        asyncio.create_task(self._video_generation_stage())
        asyncio.create_task(self._upload_stage())
    
    async def _content_analysis_stage(self):
        """Content analysis stage - analyzes and filters content"""
        while self.is_running:
            try:
                # Get content from content queue
                content_item = await asyncio.wait_for(
                    self.content_queue.get(), timeout=1.0
                )
                
                # Analyze content in parallel
                analysis_result = await self.parallel_manager.submit_and_wait(
                    WorkerType.CONTENT_ANALYZER,
                    content_item,
                    priority=1
                )
                
                if analysis_result and analysis_result.get('score', 0) > 60:
                    # Content passed analysis, move to generation
                    await self.generation_queue.put({
                        'content': content_item,
                        'analysis': analysis_result
                    })
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Content analysis stage error: {e}")
    
    async def _video_generation_stage(self):
        """Video generation stage - generates videos from analyzed content"""
        while self.is_running:
            try:
                # Get analyzed content
                generation_item = await asyncio.wait_for(
                    self.generation_queue.get(), timeout=1.0
                )
                
                # Generate video in parallel
                video_result = await self.parallel_manager.submit_and_wait(
                    WorkerType.VIDEO_GENERATOR,
                    generation_item,
                    priority=2
                )
                
                if video_result and video_result.get('success'):
                    # Video generated successfully, queue for upload
                    await self.upload_queue.put({
                        'video_data': video_result,
                        'original_content': generation_item['content'],
                        'analysis': generation_item['analysis']
                    })
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Video generation stage error: {e}")
    
    async def _upload_stage(self):
        """Upload stage - uploads generated videos"""
        while self.is_running:
            try:
                # Get generated video
                upload_item = await asyncio.wait_for(
                    self.upload_queue.get(), timeout=1.0
                )
                
                # Upload video in parallel
                upload_result = await self.parallel_manager.submit_and_wait(
                    WorkerType.UPLOADER,
                    upload_item,
                    priority=3
                )
                
                if upload_result and upload_result.get('success'):
                    self.logger.info(f"Video uploaded successfully: {upload_result.get('video_id')}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Upload stage error: {e}")
    
    async def add_content(self, content_item: Dict[str, Any]):
        """Add content to the pipeline"""
        await self.content_queue.put(content_item)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        return {
            'content_queue_size': self.content_queue.qsize(),
            'generation_queue_size': self.generation_queue.qsize(),
            'upload_queue_size': self.upload_queue.qsize(),
            'is_running': self.is_running
        }
    
    def stop_pipeline(self):
        """Stop the pipeline"""
        self.is_running = False
        self.logger.info("Stopping content generation pipeline")


# Global parallel processing manager instance
global_parallel_manager = ParallelProcessingManager()