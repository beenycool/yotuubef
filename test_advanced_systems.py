"""
Simple test to verify the new advanced systems work
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_content_analyzer():
    """Test the advanced content analyzer"""
    try:
        from src.analysis.advanced_content_analyzer import AdvancedContentAnalyzer
        
        print("🧠 Testing Advanced Content Analyzer...")
        analyzer = AdvancedContentAnalyzer()
        
        result = await analyzer.analyze_content(
            title="Amazing AI Technology Breakthrough",
            description="Revolutionary AI system that changes everything",
            metadata={"subreddit": "technology", "score": 1500}
        )
        
        print(f"   ✅ Analysis Score: {result.score:.1f}/100")
        print(f"   ✅ Sentiment: {result.sentiment_score:.2f}")
        print(f"   ✅ Keywords: {result.keywords[:3]}")
        return True
        
    except Exception as e:
        print(f"   ❌ Content Analyzer failed: {e}")
        return False

async def test_template_manager():
    """Test the dynamic template manager"""
    try:
        from src.templates.dynamic_video_templates import DynamicVideoTemplateManager
        
        print("🎬 Testing Dynamic Template Manager...")
        manager = DynamicVideoTemplateManager()
        
        mock_analysis = {
            'keywords': ['technology', 'tutorial'],
            'topics': ['technology', 'education'],
            'score': 85.0
        }
        
        template = await manager.select_optimal_template(mock_analysis)
        print(f"   ✅ Selected Template: {template.name}")
        print(f"   ✅ Duration Range: {template.duration_range}")
        return True
        
    except Exception as e:
        print(f"   ❌ Template Manager failed: {e}")
        return False

async def test_optimization_engine():
    """Test the optimization engine"""
    try:
        from src.optimization.smart_optimization_engine import SmartOptimizationEngine, TestType
        
        print("🧪 Testing Smart Optimization Engine...")
        engine = SmartOptimizationEngine()
        
        # Create a simple A/B test
        variants_config = [
            {'name': 'Variant A', 'configuration': {'title': 'Title A'}},
            {'name': 'Variant B', 'configuration': {'title': 'Title B'}}
        ]
        
        test = await engine.create_ab_test(
            TestType.TITLE,
            "Test Title Variants",
            "Testing different title formats",
            variants_config
        )
        
        print(f"   ✅ Created A/B Test: {test.test_id}")
        print(f"   ✅ Variants: {len(test.variants)}")
        return True
        
    except Exception as e:
        print(f"   ❌ Optimization Engine failed: {e}")
        return False

async def test_task_queue():
    """Test the robust task queue system"""
    try:
        from src.robustness.robust_system import global_task_queue, TaskPriority
        
        print("🛡️ Testing Robust Task Queue...")
        
        # Register a simple processor
        def simple_processor(data):
            return {'processed': True, 'data': data}
        
        global_task_queue.register_processor('test_task', simple_processor)
        
        # Add a task
        task_id = await global_task_queue.add_task(
            task_type='test_task',
            task_name='Test Task',
            task_data={'test': 'data'},
            priority=TaskPriority.NORMAL
        )
        
        print(f"   ✅ Added Task: {task_id}")
        
        # Get queue status
        status = await global_task_queue.get_queue_status()
        print(f"   ✅ Queue Length: {status.get('queue_length', 0)}")
        return True
        
    except Exception as e:
        print(f"   ❌ Task Queue failed: {e}")
        return False

async def test_parallel_processing():
    """Test the parallel processing system"""
    try:
        from src.parallel.async_processing import global_parallel_manager, WorkerType
        
        print("⚡ Testing Parallel Processing...")
        
        # Register a simple processor
        async def test_processor(data):
            await asyncio.sleep(0.1)  # Simulate work
            return {'result': data.get('input', 'no input') + '_processed'}
        
        global_parallel_manager.register_processor(WorkerType.CONTENT_ANALYZER, test_processor)
        
        # Start minimal workers
        await global_parallel_manager.start_workers({
            WorkerType.CONTENT_ANALYZER: 1
        })
        
        # Submit work
        result = await global_parallel_manager.submit_and_wait(
            WorkerType.CONTENT_ANALYZER,
            {'input': 'test_data'},
            timeout=5.0
        )
        
        print(f"   ✅ Processing Result: {result}")
        
        # Shutdown
        await global_parallel_manager.shutdown()
        return True
        
    except Exception as e:
        print(f"   ❌ Parallel Processing failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 TESTING ADVANCED SYSTEMS")
    print("=" * 40)
    
    tests = [
        ("Content Analyzer", test_content_analyzer),
        ("Template Manager", test_template_manager),
        ("Optimization Engine", test_optimization_engine),
        ("Task Queue", test_task_queue),
        ("Parallel Processing", test_parallel_processing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"   🚨 {test_name} crashed: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    successful = sum(1 for _, success in results if success)
    print("📊 TEST SUMMARY")
    print("=" * 40)
    print(f"✅ Passed: {successful}/{len(tests)}")
    print(f"📈 Success Rate: {successful/len(tests):.1%}")
    print()
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    if successful == len(tests):
        print("\n🎉 All advanced systems are working correctly!")
        print("🚀 Ready for professional-grade video generation!")
    else:
        print(f"\n⚠️ {len(tests) - successful} system(s) need attention")

if __name__ == "__main__":
    asyncio.run(main())