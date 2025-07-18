"""
Advanced Features Demo
Demonstrates the new advanced systems and improvements
"""

import asyncio
import logging
import json
from pathlib import Path
from datetime import datetime
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.advanced_content_analyzer import AdvancedContentAnalyzer
from src.templates.dynamic_video_templates import DynamicVideoTemplateManager, VideoTemplateType
from src.optimization.smart_optimization_engine import SmartOptimizationEngine, TestType
from src.robustness.robust_system import global_task_queue, RobustRetryHandler, TaskPriority
from src.parallel.async_processing import global_parallel_manager, WorkerType


async def demo_advanced_content_analysis():
    """Demonstrate advanced content analysis features"""
    print("\n🧠 ADVANCED CONTENT ANALYSIS DEMO")
    print("=" * 50)
    
    try:
        analyzer = AdvancedContentAnalyzer()
        
        # Sample content for analysis
        sample_contents = [
            {
                'title': 'Amazing AI Technology That Will Blow Your Mind',
                'description': 'This incredible new AI technology is revolutionizing how we work and live. You won\'t believe what it can do!',
                'metadata': {
                    'subreddit': 'technology',
                    'score': 1250,
                    'num_comments': 89
                }
            },
            {
                'title': 'Funny Cat Video Compilation',
                'description': 'Hilarious cats doing silly things. Perfect for a good laugh!',
                'metadata': {
                    'subreddit': 'funny',
                    'score': 800,
                    'num_comments': 45
                }
            },
            {
                'title': 'How to Learn Python Programming in 10 Minutes',
                'description': 'Quick tutorial for beginners to get started with Python programming language.',
                'metadata': {
                    'subreddit': 'learnpython',
                    'score': 650,
                    'num_comments': 127
                }
            }
        ]
        
        print("Analyzing sample content with advanced features:")
        print("- Sentiment Analysis (NLTK/TextBlob)")
        print("- Keyword Trend Analysis (Google Trends)")
        print("- Uniqueness Checking (Database)")
        print("- Engagement Prediction (ML Model)")
        print()
        
        for i, content in enumerate(sample_contents, 1):
            print(f"📝 Content {i}: {content['title'][:50]}...")
            
            # Perform analysis
            result = await analyzer.analyze_content(
                title=content['title'],
                description=content['description'],
                metadata=content['metadata']
            )
            
            print(f"   📊 Overall Score: {result.score:.1f}/100")
            print(f"   😊 Sentiment: {result.sentiment_score:.2f} (-1 to 1)")
            print(f"   📈 Trend Relevance: {result.trend_relevance:.1f}/100")
            print(f"   🆕 Uniqueness: {result.uniqueness_score:.1f}/100")
            print(f"   🎯 Engagement Potential: {result.engagement_potential:.1f}/100")
            print(f"   🔑 Keywords: {', '.join(result.keywords[:5])}")
            print(f"   📂 Topics: {', '.join(result.topics[:3])}")
            print(f"   💡 Reasons: {result.reasons[0] if result.reasons else 'No specific reasons'}")
            print()
        
        print("✅ Advanced Content Analysis Demo Complete!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


async def demo_dynamic_video_templates():
    """Demonstrate dynamic video templates and asset sourcing"""
    print("\n🎬 DYNAMIC VIDEO TEMPLATES DEMO")
    print("=" * 50)
    
    try:
        template_manager = DynamicVideoTemplateManager()
        
        # Sample content analysis
        mock_analysis = {
            'keywords': ['technology', 'amazing', 'tutorial', 'guide'],
            'topics': ['technology', 'education'],
            'sentiment_score': 0.7,
            'trend_relevance': 85.0,
            'engagement_potential': 78.0
        }
        
        print("Available Video Template Types:")
        for template_type in VideoTemplateType:
            print(f"   📹 {template_type.value.replace('_', ' ').title()}")
        print()
        
        print("Selecting optimal template based on content analysis...")
        
        # Select optimal template
        template = await template_manager.select_optimal_template(
            mock_analysis,
            preferences={'duration_preference': 45}
        )
        
        print(f"🎯 Selected Template: {template.name}")
        print(f"   📝 Description: {template.description}")
        print(f"   ⏱️ Duration Range: {template.duration_range[0]}-{template.duration_range[1]} seconds")
        print(f"   🎨 Style: {template.style_config}")
        print(f"   📊 Compatibility Score: {template.compatibility_score}")
        print()
        
        print("Template Structure:")
        for i, stage in enumerate(template.structure, 1):
            print(f"   {i}. {stage['type'].title()} ({stage['duration']}s) - {stage['content']}")
        print()
        
        print("Asset Requirements:")
        for asset_type, count in template.asset_requirements.items():
            print(f"   📦 {asset_type.replace('_', ' ').title()}: {count} items")
        print()
        
        # Demonstrate asset sourcing (will use fallbacks)
        print("Sourcing assets for template...")
        assets = await template_manager.source_assets_for_template(
            template,
            mock_analysis['keywords'],
            mock_analysis['topics']
        )
        
        total_assets = sum(len(asset_list) for asset_list in assets.values())
        print(f"📥 Sourced {total_assets} assets from various sources")
        
        for asset_type, asset_list in assets.items():
            if asset_list:
                print(f"   {asset_type}: {len(asset_list)} items")
                for asset in asset_list[:2]:  # Show first 2
                    print(f"     - {asset.description} ({asset.source})")
        
        print("\n✅ Dynamic Video Templates Demo Complete!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


async def demo_smart_optimization():
    """Demonstrate A/B testing and smart optimization"""
    print("\n🧪 SMART OPTIMIZATION & A/B TESTING DEMO")
    print("=" * 50)
    
    try:
        optimization_engine = SmartOptimizationEngine()
        
        print("Creating A/B Tests for different video elements...")
        print()
        
        # Create thumbnail A/B test
        thumbnail_variants = [
            "thumbnail_bright_colors.jpg",
            "thumbnail_minimal_design.jpg", 
            "thumbnail_action_shot.jpg"
        ]
        
        thumbnail_test = await optimization_engine.create_thumbnail_ab_test(
            thumbnail_variants,
            "Thumbnail Style Comparison"
        )
        
        print(f"🖼️ Created Thumbnail A/B Test: {thumbnail_test.test_id}")
        print(f"   Variants: {len(thumbnail_test.variants)}")
        for variant in thumbnail_test.variants:
            print(f"     - {variant.variant_name}: {variant.configuration}")
        print()
        
        # Create title A/B test
        title_variants = [
            "How to Master This Skill in 60 Seconds",
            "AMAZING Skill Mastery - You Won't Believe This!",
            "Quick Tutorial: Master This Essential Skill"
        ]
        
        title_test = await optimization_engine.create_title_ab_test(
            title_variants,
            "Title Format Effectiveness"
        )
        
        print(f"📝 Created Title A/B Test: {title_test.test_id}")
        print(f"   Variants: {len(title_test.variants)}")
        for variant in title_test.variants:
            print(f"     - {variant.variant_name}: {variant.configuration}")
        print()
        
        # Create upload time A/B test
        time_variants = ["09:00", "12:00", "16:00", "19:00", "21:00"]
        
        time_test = await optimization_engine.create_upload_time_ab_test(
            time_variants,
            "Optimal Upload Time Analysis"
        )
        
        print(f"⏰ Created Upload Time A/B Test: {time_test.test_id}")
        print(f"   Time Slots: {[v.configuration['upload_time'] for v in time_test.variants]}")
        print()
        
        # Start tests
        await optimization_engine.start_test(thumbnail_test.test_id)
        await optimization_engine.start_test(title_test.test_id)
        await optimization_engine.start_test(time_test.test_id)
        print("🚀 Started all A/B tests")
        
        # Simulate collecting test data
        print("\n📊 Simulating test data collection...")
        import random
        
        for test in [thumbnail_test, title_test, time_test]:
            for variant in test.variants:
                # Simulate metrics for each variant
                metrics = {
                    'views': random.randint(500, 2000),
                    'clicks': random.randint(25, 200),
                    'watch_time': random.uniform(45, 120),
                    'engagement_rate': random.uniform(0.08, 0.30),
                    'conversion_rate': random.uniform(0.02, 0.15),
                    'retention_rate': random.uniform(0.50, 0.85)
                }
                
                await optimization_engine.collect_test_data(
                    variant.variant_id,
                    f"video_{random.randint(1000, 9999)}",
                    metrics
                )
        
        print("✅ Test data collected for all variants")
        
        # Get optimization recommendations
        recommendations = await optimization_engine.get_optimization_recommendations()
        
        print(f"\n💡 Generated {len(recommendations)} optimization recommendations:")
        for rec in recommendations[:3]:  # Show top 3
            print(f"   📈 {rec['type'].title()}: {rec['description']}")
            print(f"      Priority: {rec['priority_score']:.1f}, Impact: {rec['expected_impact']:.1%}")
        
        # Get analytics
        analytics = await optimization_engine.get_test_analytics()
        print(f"\n📊 Test Analytics:")
        print(f"   Active Tests: {analytics.get('active_tests', 0)}")
        print(f"   Completed Tests: {analytics.get('completed_tests', 0)}")
        print(f"   Recent Winners: {len(analytics.get('recent_winners', []))}")
        
        print("\n✅ Smart Optimization Demo Complete!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


async def demo_robust_error_handling():
    """Demonstrate robust error handling and task queues"""
    print("\n🛡️ ROBUST ERROR HANDLING & TASK QUEUES DEMO")
    print("=" * 50)
    
    try:
        # Register a demo task processor
        async def demo_task_processor(task_data):
            task_name = task_data.get('name', 'unknown')
            
            # Simulate processing time
            await asyncio.sleep(1)
            
            # Simulate occasional failures for demonstration
            import random
            if random.random() < 0.3:  # 30% failure rate for demo
                raise Exception(f"Simulated processing error for {task_name}")
            
            return {
                'success': True,
                'processed_item': task_name,
                'timestamp': datetime.now().isoformat()
            }
        
        global_task_queue.register_processor('demo_task', demo_task_processor)
        
        print("🔧 Registered demo task processor with simulated failures")
        print("📋 Adding tasks to persistent queue...")
        
        # Add sample tasks
        task_ids = []
        for i in range(5):
            task_id = await global_task_queue.add_task(
                task_type='demo_task',
                task_name=f'Demo Task {i+1}',
                task_data={'name': f'task_{i+1}', 'priority': i},
                priority=TaskPriority.NORMAL
            )
            task_ids.append(task_id)
            print(f"   ✅ Added: {task_id}")
        
        print()
        
        # Start processing (simulate for a short time)
        print("🚀 Starting task queue processing...")
        processing_task = asyncio.create_task(
            global_task_queue.start_processing(max_concurrent=2)
        )
        
        # Let it run for a bit
        await asyncio.sleep(8)
        
        # Stop processing
        global_task_queue.stop_processing()
        processing_task.cancel()
        
        # Get queue status
        status = await global_task_queue.get_queue_status()
        print(f"📊 Queue Status:")
        print(f"   Pending Tasks: {status.get('queue_length', 0)}")
        print(f"   Processing: {status.get('processing_count', 0)}")
        print(f"   Registered Processors: {len(status.get('registered_processors', []))}")
        
        # Demonstrate retry decorator
        print("\n🔄 Demonstrating retry decorator...")
        
        @RobustRetryHandler.with_retry(
            max_attempts=3,
            base_delay=0.5,
            exponential_backoff=True
        )
        async def unreliable_function():
            import random
            if random.random() < 0.7:  # 70% failure rate
                raise Exception("Simulated network error")
            return "Success after retries!"
        
        try:
            result = await unreliable_function()
            print(f"   ✅ {result}")
        except Exception as e:
            print(f"   ❌ Failed after all retries: {e}")
        
        print("\n✅ Robust Error Handling Demo Complete!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


async def demo_parallel_processing():
    """Demonstrate parallel processing capabilities"""
    print("\n⚡ PARALLEL PROCESSING DEMO")
    print("=" * 50)
    
    try:
        # Register demo processors
        async def content_processor(data):
            await asyncio.sleep(1)  # Simulate processing time
            return {
                'processed': True,
                'content_id': data.get('id'),
                'analysis_score': data.get('score', 0) * 1.2  # Boost score
            }
        
        async def video_processor(data):
            await asyncio.sleep(2)  # Simulate longer processing
            return {
                'video_generated': True,
                'video_id': f"video_{data.get('content_id', 'unknown')}",
                'duration': 45
            }
        
        global_parallel_manager.register_processor(WorkerType.CONTENT_ANALYZER, content_processor)
        global_parallel_manager.register_processor(WorkerType.VIDEO_GENERATOR, video_processor)
        
        # Start workers
        worker_configs = {
            WorkerType.CONTENT_ANALYZER: 2,
            WorkerType.VIDEO_GENERATOR: 1
        }
        
        print("🚀 Starting parallel workers...")
        await global_parallel_manager.start_workers(worker_configs)
        
        # Submit batch work
        content_items = [
            {'id': f'content_{i}', 'score': 70 + i * 5}
            for i in range(5)
        ]
        
        print(f"📤 Submitting {len(content_items)} content items for analysis...")
        
        # Process content in parallel
        analysis_results = await global_parallel_manager.process_batch_parallel(
            WorkerType.CONTENT_ANALYZER,
            content_items,
            priority=1,
            timeout=10.0
        )
        
        successful_analyses = [r for r in analysis_results if isinstance(r, dict) and r.get('processed')]
        print(f"✅ Completed {len(successful_analyses)}/{len(content_items)} content analyses")
        
        # Process videos for successful analyses
        if successful_analyses:
            print("🎬 Generating videos for analyzed content...")
            
            video_results = await global_parallel_manager.process_batch_parallel(
                WorkerType.VIDEO_GENERATOR,
                successful_analyses[:3],  # Process first 3
                priority=2,
                timeout=15.0
            )
            
            successful_videos = [r for r in video_results if isinstance(r, dict) and r.get('video_generated')]
            print(f"✅ Generated {len(successful_videos)} videos")
            
            for video in successful_videos:
                print(f"   🎥 {video['video_id']} ({video['duration']}s)")
        
        # Get performance stats
        stats = global_parallel_manager.get_performance_stats()
        worker_status = global_parallel_manager.get_worker_status()
        
        print(f"\n📊 Parallel Processing Stats:")
        print(f"   Items Processed: {stats.get('items_processed', 0)}")
        print(f"   Success Rate: {stats.get('success_rate', 0):.1%}")
        print(f"   Throughput: {stats.get('throughput_per_minute', 0):.1f} items/min")
        print(f"   Average Processing Time: {stats.get('average_processing_time', 0):.2f}s")
        
        print(f"\n👷 Worker Status:")
        for worker_type, status in worker_status.items():
            print(f"   {worker_type}: {status['active_workers']}/{status['total_workers']} active")
        
        # Shutdown workers
        await global_parallel_manager.shutdown()
        
        print("\n✅ Parallel Processing Demo Complete!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


async def main():
    """Run all demos"""
    print("🚀 ADVANCED YOUTUBE VIDEO GENERATOR - FEATURE DEMOS")
    print("=" * 60)
    print("Demonstrating all the advanced improvements:")
    print("1. 🧠 Advanced Content Analysis")
    print("2. 🎬 Dynamic Video Templates") 
    print("3. 🧪 Smart Optimization & A/B Testing")
    print("4. 🛡️ Robust Error Handling")
    print("5. ⚡ Parallel Processing")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during demo
    
    demos = [
        ("Advanced Content Analysis", demo_advanced_content_analysis),
        ("Dynamic Video Templates", demo_dynamic_video_templates),
        ("Smart Optimization", demo_smart_optimization),
        ("Robust Error Handling", demo_robust_error_handling),
        ("Parallel Processing", demo_parallel_processing)
    ]
    
    results = []
    start_time = datetime.now()
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n{'='*20} {demo_name.upper()} {'='*20}")
            success = await demo_func()
            results.append((demo_name, success))
            
            if success:
                print(f"✅ {demo_name} demo completed successfully!")
            else:
                print(f"❌ {demo_name} demo failed!")
                
        except Exception as e:
            print(f"🚨 {demo_name} demo crashed: {e}")
            results.append((demo_name, False))
    
    # Summary
    elapsed_time = datetime.now() - start_time
    successful_demos = sum(1 for _, success in results if success)
    
    print(f"\n{'='*60}")
    print("📊 DEMO SUMMARY")
    print("=" * 60)
    print(f"⏱️ Total Time: {elapsed_time.total_seconds():.1f} seconds")
    print(f"✅ Successful: {successful_demos}/{len(demos)} demos")
    print(f"📈 Success Rate: {successful_demos/len(demos):.1%}")
    print()
    
    for demo_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {status} {demo_name}")
    
    print("\n🎉 All demos completed!")
    print("🚀 The YouTube Video Generator is now equipped with:")
    print("   • Advanced AI-powered content analysis")
    print("   • Dynamic video templates with asset sourcing")
    print("   • Comprehensive A/B testing and optimization")
    print("   • Bulletproof error handling and recovery")
    print("   • High-performance parallel processing")
    print("\n💡 Ready for production use with professional-grade reliability!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n🚨 Demo failed: {e}")
        import traceback
        traceback.print_exc()