"""
PipelineManager class responsible for coordinating parallel processing stages
"""

import logging
import asyncio
from typing import Dict, List, Any

# Import TaskPriority for proper task queue integration
try:
    from src.robustness.robust_system import TaskPriority
    ROBUST_SYSTEM_AVAILABLE = True
except ImportError:
    # Fallback enum if robust system is not available
    from enum import Enum
    class TaskPriority(Enum):
        LOW = 1
        NORMAL = 2  
        HIGH = 3
    ROBUST_SYSTEM_AVAILABLE = False

class PipelineManager:
    """
    Manages the video generation pipeline through parallel processing stages.
    Single responsibility: Coordinate processing stages and manage task flow.
    """
    
    def __init__(self, task_queue=None, parallel_manager=None):
        self.logger = logging.getLogger(__name__)
        self.task_queue = task_queue
        self.parallel_manager = parallel_manager
        
        # Pipeline stages
        self.pipeline_stages = [
            'content_analysis',
            'video_generation', 
            'video_processing',
            'video_upload'
        ]
        
        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'successful_completions': 0,
            'failed_tasks': 0,
            'stage_performance': {stage: {'count': 0, 'avg_time': 0} for stage in self.pipeline_stages}
        }
        
    async def process_content_through_pipeline(self, content_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a content item through the complete video generation pipeline.
        
        Args:
            content_item: Content to process through pipeline
            
        Returns:
            Dict containing the final processing result
        """
        try:
            self.logger.info(f"Starting pipeline processing for content: {content_item.get('content', {}).get('title', 'Unknown')}")
            
            # Initialize processing context
            processing_context = {
                'content': content_item,
                'pipeline_id': f"pipeline_{asyncio.current_task().get_name() if asyncio.current_task() else 'unknown'}",
                'started_at': asyncio.get_event_loop().time(),
                'stage_results': {}
            }
            
            # Process through each pipeline stage
            for stage in self.pipeline_stages:
                try:
                    stage_result = await self._process_pipeline_stage(stage, processing_context)
                    processing_context['stage_results'][stage] = stage_result
                    
                    # Check if stage failed
                    if not stage_result.get('success', False):
                        self.logger.error(f"Pipeline stage '{stage}' failed for content")
                        self.stats['failed_tasks'] += 1
                        return {
                            'success': False,
                            'failed_at_stage': stage,
                            'error': stage_result.get('error', 'Unknown error'),
                            'processing_context': processing_context
                        }
                        
                except Exception as e:
                    self.logger.error(f"Exception in pipeline stage '{stage}': {e}")
                    self.stats['failed_tasks'] += 1
                    return {
                        'success': False,
                        'failed_at_stage': stage,
                        'error': str(e),
                        'processing_context': processing_context
                    }
            
            # Pipeline completed successfully
            processing_time = asyncio.get_event_loop().time() - processing_context['started_at']
            self.stats['successful_completions'] += 1
            self.stats['total_processed'] += 1
            
            self.logger.info(f"Pipeline processing completed successfully in {processing_time:.2f}s")
            
            return {
                'success': True,
                'processing_time': processing_time,
                'stage_results': processing_context['stage_results'],
                'final_result': processing_context['stage_results'].get('video_upload', {})
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {e}")
            self.stats['failed_tasks'] += 1
            return {
                'success': False,
                'error': str(e),
                'failed_at_stage': 'initialization'
            }
            
    async def _process_pipeline_stage(self, stage: str, processing_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single pipeline stage.
        
        Args:
            stage: Name of the pipeline stage
            processing_context: Current processing context
            
        Returns:
            Dict containing stage processing result
        """
        stage_start_time = asyncio.get_event_loop().time()
        
        try:
            self.logger.debug(f"Processing pipeline stage: {stage}")
            
            # Prepare stage input data
            stage_input = self._prepare_stage_input(stage, processing_context)
            
            # Process through appropriate handler
            if self.parallel_manager and hasattr(self.parallel_manager, 'process_task'):
                # Use parallel manager if available
                result = await self.parallel_manager.process_task(stage, stage_input)
            else:
                # Use fallback processing
                result = await self._fallback_stage_processing(stage, stage_input)
            
            # Update performance statistics
            stage_time = asyncio.get_event_loop().time() - stage_start_time
            self._update_stage_stats(stage, stage_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Stage '{stage}' processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'stage': stage
            }
            
    def _prepare_stage_input(self, stage: str, processing_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare input data for a specific pipeline stage.
        
        Args:
            stage: Name of the pipeline stage
            processing_context: Current processing context
            
        Returns:
            Dict containing stage input data
        """
        base_input = {
            'content': processing_context['content'],
            'pipeline_id': processing_context['pipeline_id'],
            'stage': stage
        }
        
        # Add previous stage results as needed
        if stage == 'video_generation' and 'content_analysis' in processing_context['stage_results']:
            base_input['analysis'] = processing_context['stage_results']['content_analysis'].get('analysis')
            
        elif stage == 'video_processing' and 'video_generation' in processing_context['stage_results']:
            base_input['video_data'] = processing_context['stage_results']['video_generation'].get('video_data')
            
        elif stage == 'video_upload' and 'video_processing' in processing_context['stage_results']:
            base_input['processed_video'] = processing_context['stage_results']['video_processing'].get('processed_video')
            
        return base_input
        
    async def _fallback_stage_processing(self, stage: str, stage_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback processing for pipeline stages when parallel manager is not available.
        
        Args:
            stage: Name of the pipeline stage
            stage_input: Input data for the stage
            
        Returns:
            Dict containing stage processing result
        """
        # Simulate processing for each stage
        await asyncio.sleep(0.5)  # Simulate processing time
        
        if stage == 'content_analysis':
            return {
                'success': True,
                'analysis': {
                    'overall_score': 0.75,
                    'sentiment_score': 0.6,
                    'keywords': ['content', 'analysis'],
                    'processing_stage': stage
                }
            }
            
        elif stage == 'video_generation':
            return {
                'success': True,
                'video_data': {
                    'video_path': f"/tmp/generated_video_{stage_input.get('pipeline_id', 'unknown')}.mp4",
                    'duration': 30,
                    'processing_stage': stage
                }
            }
            
        elif stage == 'video_processing':
            return {
                'success': True,
                'processed_video': {
                    'final_path': f"/tmp/processed_video_{stage_input.get('pipeline_id', 'unknown')}.mp4",
                    'size_mb': 5.2,
                    'processing_stage': stage
                }
            }
            
        elif stage == 'video_upload':
            return {
                'success': True,
                'upload_result': {
                    'video_id': f"video_{stage_input.get('pipeline_id', 'unknown')}",
                    'url': f"https://youtube.com/watch?v=sample_{stage_input.get('pipeline_id', 'unknown')}",
                    'processing_stage': stage
                }
            }
            
        else:
            return {
                'success': False,
                'error': f"Unknown pipeline stage: {stage}"
            }
            
    def _update_stage_stats(self, stage: str, processing_time: float):
        """
        Update performance statistics for a pipeline stage.
        
        Args:
            stage: Name of the pipeline stage
            processing_time: Time taken to process the stage
        """
        if stage in self.stats['stage_performance']:
            stage_stats = self.stats['stage_performance'][stage]
            stage_stats['count'] += 1
            
            # Update running average
            if stage_stats['count'] == 1:
                stage_stats['avg_time'] = processing_time
            else:
                # Running average formula: new_avg = old_avg + (new_value - old_avg) / count
                stage_stats['avg_time'] += (processing_time - stage_stats['avg_time']) / stage_stats['count']
                
    async def queue_content_for_processing(self, content_items: List[Dict[str, Any]]) -> List[str]:
        """
        Queue multiple content items for pipeline processing.
        
        Args:
            content_items: List of content items to queue
            
        Returns:
            List of task IDs for queued items
        """
        task_ids = []
        
        for content_item in content_items:
            try:
                if self.task_queue and hasattr(self.task_queue, 'add_task'):
                    task_id = await self.task_queue.add_task(
                        task_type='pipeline_processing',
                        task_name=f"Process content: {content_item.get('content', {}).get('title', 'Unknown')[:50]}",
                        task_data=content_item,
                        priority=TaskPriority.NORMAL
                    )
                    task_ids.append(task_id)
                else:
                    # Create async task for immediate processing
                    task = asyncio.create_task(self.process_content_through_pipeline(content_item))
                    task_ids.append(f"async_task_{id(task)}")
                    
            except Exception as e:
                self.logger.error(f"Failed to queue content item for processing: {e}")
                
        self.logger.info(f"Queued {len(task_ids)} content items for pipeline processing")
        return task_ids
        
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get current pipeline performance statistics.
        
        Returns:
            Dict containing pipeline statistics
        """
        success_rate = (self.stats['successful_completions'] / max(self.stats['total_processed'], 1)) * 100
        
        return {
            'total_processed': self.stats['total_processed'],
            'successful_completions': self.stats['successful_completions'],
            'failed_tasks': self.stats['failed_tasks'],
            'success_rate_percent': round(success_rate, 2),
            'stage_performance': self.stats['stage_performance'],
            'pipeline_stages': self.pipeline_stages
        }