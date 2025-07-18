"""
Smart Optimization Loop with A/B Testing and Data-Driven Decisions
Implements comprehensive testing and optimization for video performance
"""

import asyncio
import logging
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import statistics

# Optional dependencies with fallbacks
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from src.config.settings import get_config


class TestType(Enum):
    """Types of A/B tests available"""
    THUMBNAIL = "thumbnail"
    TITLE = "title"
    UPLOAD_TIME = "upload_time"
    DESCRIPTION = "description"
    TAGS = "tags"
    MUSIC = "music"
    DURATION = "duration"


class TestStatus(Enum):
    """Status of A/B tests"""
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class TestVariant:
    """A/B test variant configuration"""
    variant_id: str
    test_id: str
    variant_name: str
    configuration: Dict[str, Any]
    expected_improvement: float  # Expected improvement percentage
    confidence_level: float = 0.95
    sample_size: int = 100


@dataclass
class TestResult:
    """A/B test result data"""
    variant_id: str
    views: int
    clicks: int
    watch_time: float
    engagement_rate: float
    conversion_rate: float
    retention_rate: float
    collected_at: datetime


@dataclass
class ABTest:
    """Complete A/B test configuration and results"""
    test_id: str
    test_type: TestType
    test_name: str
    description: str
    variants: List[TestVariant]
    status: TestStatus
    start_date: datetime
    end_date: Optional[datetime]
    results: List[TestResult]
    winner_variant_id: Optional[str] = None
    confidence_score: float = 0.0
    statistical_significance: bool = False


class SmartOptimizationEngine:
    """
    Advanced optimization engine with A/B testing and data-driven decisions
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize database for test tracking
        self.db_path = Path("data/optimization.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # Active tests tracking
        self.active_tests: Dict[str, ABTest] = {}
        self._load_active_tests()
        
        # Performance metrics configuration
        self.metrics_config = {
            'primary_metric': 'engagement_rate',
            'secondary_metrics': ['views', 'watch_time', 'retention_rate'],
            'minimum_sample_size': 50,
            'test_duration_days': 7,
            'confidence_threshold': 0.95,
            'minimum_improvement': 0.05  # 5% minimum improvement
        }
        
        # Test scheduling
        self.test_schedule = {}
        self._load_test_schedule()
        
        self.logger.info("Smart Optimization Engine initialized")
    
    def _init_database(self):
        """Initialize SQLite database for optimization tracking"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # A/B tests table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS ab_tests (
                        test_id TEXT PRIMARY KEY,
                        test_type TEXT,
                        test_name TEXT,
                        description TEXT,
                        status TEXT,
                        start_date TIMESTAMP,
                        end_date TIMESTAMP,
                        winner_variant_id TEXT,
                        confidence_score REAL,
                        statistical_significance BOOLEAN,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Test variants table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS test_variants (
                        variant_id TEXT PRIMARY KEY,
                        test_id TEXT,
                        variant_name TEXT,
                        configuration TEXT,
                        expected_improvement REAL,
                        confidence_level REAL,
                        sample_size INTEGER,
                        FOREIGN KEY (test_id) REFERENCES ab_tests (test_id)
                    )
                """)
                
                # Test results table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS test_results (
                        result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        variant_id TEXT,
                        video_id TEXT,
                        views INTEGER,
                        clicks INTEGER,
                        watch_time REAL,
                        engagement_rate REAL,
                        conversion_rate REAL,
                        retention_rate REAL,
                        collected_at TIMESTAMP,
                        FOREIGN KEY (variant_id) REFERENCES test_variants (variant_id)
                    )
                """)
                
                # Performance metrics table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        video_id TEXT,
                        metric_name TEXT,
                        metric_value REAL,
                        baseline_value REAL,
                        improvement_percentage REAL,
                        measurement_date TIMESTAMP,
                        metadata TEXT
                    )
                """)
                
                # Optimization recommendations table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS optimization_recommendations (
                        recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        recommendation_type TEXT,
                        priority_score REAL,
                        description TEXT,
                        expected_impact REAL,
                        implementation_complexity TEXT,
                        status TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        applied_at TIMESTAMP
                    )
                """)
                
                self.logger.info("Optimization database initialized")
                
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
    
    def _load_active_tests(self):
        """Load active tests from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT test_id, test_type, test_name, description, status, 
                           start_date, end_date, winner_variant_id, confidence_score,
                           statistical_significance
                    FROM ab_tests 
                    WHERE status IN ('planned', 'running')
                """)
                
                for row in cursor.fetchall():
                    test_id = row[0]
                    
                    # Load variants for this test
                    cursor.execute("""
                        SELECT variant_id, variant_name, configuration, 
                               expected_improvement, confidence_level, sample_size
                        FROM test_variants 
                        WHERE test_id = ?
                    """, (test_id,))
                    
                    variants = []
                    for variant_row in cursor.fetchall():
                        variant = TestVariant(
                            variant_id=variant_row[0],
                            test_id=test_id,
                            variant_name=variant_row[1],
                            configuration=json.loads(variant_row[2]),
                            expected_improvement=variant_row[3],
                            confidence_level=variant_row[4],
                            sample_size=variant_row[5]
                        )
                        variants.append(variant)
                    
                    # Load results for this test
                    cursor.execute("""
                        SELECT variant_id, views, clicks, watch_time, engagement_rate,
                               conversion_rate, retention_rate, collected_at
                        FROM test_results 
                        WHERE variant_id IN (SELECT variant_id FROM test_variants WHERE test_id = ?)
                    """, (test_id,))
                    
                    results = []
                    for result_row in cursor.fetchall():
                        result = TestResult(
                            variant_id=result_row[0],
                            views=result_row[1],
                            clicks=result_row[2],
                            watch_time=result_row[3],
                            engagement_rate=result_row[4],
                            conversion_rate=result_row[5],
                            retention_rate=result_row[6],
                            collected_at=datetime.fromisoformat(result_row[7])
                        )
                        results.append(result)
                    
                    # Create AB test object
                    ab_test = ABTest(
                        test_id=test_id,
                        test_type=TestType(row[1]),
                        test_name=row[2],
                        description=row[3],
                        variants=variants,
                        status=TestStatus(row[4]),
                        start_date=datetime.fromisoformat(row[5]),
                        end_date=datetime.fromisoformat(row[6]) if row[6] else None,
                        results=results,
                        winner_variant_id=row[7],
                        confidence_score=row[8] or 0.0,
                        statistical_significance=bool(row[9])
                    )
                    
                    self.active_tests[test_id] = ab_test
                
                self.logger.info(f"Loaded {len(self.active_tests)} active tests")
                
        except Exception as e:
            self.logger.warning(f"Failed to load active tests: {e}")
    
    def _load_test_schedule(self):
        """Load test scheduling configuration"""
        try:
            schedule_file = Path("data/test_schedule.json")
            if schedule_file.exists():
                with open(schedule_file, 'r') as f:
                    self.test_schedule = json.load(f)
            else:
                # Default test schedule
                self.test_schedule = {
                    'thumbnail_tests': {'frequency': 'weekly', 'priority': 'high'},
                    'title_tests': {'frequency': 'bi-weekly', 'priority': 'medium'},
                    'upload_time_tests': {'frequency': 'monthly', 'priority': 'medium'},
                    'description_tests': {'frequency': 'monthly', 'priority': 'low'},
                    'music_tests': {'frequency': 'bi-weekly', 'priority': 'medium'},
                    'duration_tests': {'frequency': 'monthly', 'priority': 'low'}
                }
                self._save_test_schedule()
                
        except Exception as e:
            self.logger.warning(f"Failed to load test schedule: {e}")
            self.test_schedule = {}
    
    def _save_test_schedule(self):
        """Save test scheduling configuration"""
        try:
            schedule_file = Path("data/test_schedule.json")
            schedule_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(schedule_file, 'w') as f:
                json.dump(self.test_schedule, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Failed to save test schedule: {e}")
    
    async def create_ab_test(self, 
                           test_type: TestType,
                           test_name: str,
                           description: str,
                           variants_config: List[Dict[str, Any]]) -> ABTest:
        """
        Create a new A/B test
        
        Args:
            test_type: Type of test to create
            test_name: Human-readable test name
            description: Test description
            variants_config: List of variant configurations
            
        Returns:
            Created ABTest object
        """
        try:
            self.logger.info(f"Creating A/B test: {test_name}")
            
            test_id = self._generate_test_id(test_type, test_name)
            
            # Create test variants
            variants = []
            for i, config in enumerate(variants_config):
                variant = TestVariant(
                    variant_id=f"{test_id}_variant_{i}",
                    test_id=test_id,
                    variant_name=config.get('name', f'Variant {i+1}'),
                    configuration=config.get('configuration', {}),
                    expected_improvement=config.get('expected_improvement', 0.1),
                    confidence_level=config.get('confidence_level', 0.95),
                    sample_size=config.get('sample_size', self.metrics_config['minimum_sample_size'])
                )
                variants.append(variant)
            
            # Create AB test
            ab_test = ABTest(
                test_id=test_id,
                test_type=test_type,
                test_name=test_name,
                description=description,
                variants=variants,
                status=TestStatus.PLANNED,
                start_date=datetime.now(),
                end_date=None,
                results=[]
            )
            
            # Save to database
            await self._save_ab_test(ab_test)
            
            # Add to active tests
            self.active_tests[test_id] = ab_test
            
            self.logger.info(f"Created A/B test: {test_id} with {len(variants)} variants")
            return ab_test
            
        except Exception as e:
            self.logger.error(f"Failed to create A/B test: {e}")
            raise
    
    def _generate_test_id(self, test_type: TestType, test_name: str) -> str:
        """Generate unique test ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name_hash = hashlib.md5(test_name.encode()).hexdigest()[:8]
        return f"{test_type.value}_{timestamp}_{name_hash}"
    
    async def _save_ab_test(self, ab_test: ABTest):
        """Save A/B test to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Save test
                conn.execute("""
                    INSERT OR REPLACE INTO ab_tests 
                    (test_id, test_type, test_name, description, status, start_date, 
                     end_date, winner_variant_id, confidence_score, statistical_significance)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ab_test.test_id,
                    ab_test.test_type.value,
                    ab_test.test_name,
                    ab_test.description,
                    ab_test.status.value,
                    ab_test.start_date.isoformat(),
                    ab_test.end_date.isoformat() if ab_test.end_date else None,
                    ab_test.winner_variant_id,
                    ab_test.confidence_score,
                    ab_test.statistical_significance
                ))
                
                # Save variants
                for variant in ab_test.variants:
                    conn.execute("""
                        INSERT OR REPLACE INTO test_variants
                        (variant_id, test_id, variant_name, configuration, 
                         expected_improvement, confidence_level, sample_size)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        variant.variant_id,
                        variant.test_id,
                        variant.variant_name,
                        json.dumps(variant.configuration),
                        variant.expected_improvement,
                        variant.confidence_level,
                        variant.sample_size
                    ))
                
                # Save results
                for result in ab_test.results:
                    conn.execute("""
                        INSERT OR REPLACE INTO test_results
                        (variant_id, views, clicks, watch_time, engagement_rate,
                         conversion_rate, retention_rate, collected_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        result.variant_id,
                        result.views,
                        result.clicks,
                        result.watch_time,
                        result.engagement_rate,
                        result.conversion_rate,
                        result.retention_rate,
                        result.collected_at.isoformat()
                    ))
                
        except Exception as e:
            self.logger.error(f"Failed to save A/B test: {e}")
    
    async def start_test(self, test_id: str) -> bool:
        """Start an A/B test"""
        try:
            if test_id not in self.active_tests:
                self.logger.error(f"Test not found: {test_id}")
                return False
            
            test = self.active_tests[test_id]
            test.status = TestStatus.RUNNING
            test.start_date = datetime.now()
            
            await self._save_ab_test(test)
            
            self.logger.info(f"Started A/B test: {test_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start test: {e}")
            return False
    
    async def collect_test_data(self, 
                              variant_id: str,
                              video_id: str,
                              metrics: Dict[str, Any]) -> bool:
        """
        Collect performance data for a test variant
        
        Args:
            variant_id: Test variant ID
            video_id: Video ID for tracking
            metrics: Performance metrics
            
        Returns:
            Success status
        """
        try:
            # Create test result
            result = TestResult(
                variant_id=variant_id,
                views=metrics.get('views', 0),
                clicks=metrics.get('clicks', 0),
                watch_time=metrics.get('watch_time', 0.0),
                engagement_rate=metrics.get('engagement_rate', 0.0),
                conversion_rate=metrics.get('conversion_rate', 0.0),
                retention_rate=metrics.get('retention_rate', 0.0),
                collected_at=datetime.now()
            )
            
            # Find the test this variant belongs to
            test_id = None
            for tid, test in self.active_tests.items():
                if any(v.variant_id == variant_id for v in test.variants):
                    test_id = tid
                    break
            
            if test_id:
                test = self.active_tests[test_id]
                test.results.append(result)
                
                # Save to database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO test_results
                        (variant_id, video_id, views, clicks, watch_time, engagement_rate,
                         conversion_rate, retention_rate, collected_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        result.variant_id,
                        video_id,
                        result.views,
                        result.clicks,
                        result.watch_time,
                        result.engagement_rate,
                        result.conversion_rate,
                        result.retention_rate,
                        result.collected_at.isoformat()
                    ))
                
                self.logger.info(f"Collected test data for variant: {variant_id}")
                
                # Check if test should be analyzed
                await self._check_test_completion(test_id)
                
                return True
            else:
                self.logger.warning(f"Test not found for variant: {variant_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to collect test data: {e}")
            return False
    
    async def _check_test_completion(self, test_id: str):
        """Check if a test has enough data for analysis"""
        try:
            test = self.active_tests[test_id]
            
            if test.status != TestStatus.RUNNING:
                return
            
            # Check if we have enough samples for each variant
            variant_sample_counts = {}
            for result in test.results:
                variant_sample_counts[result.variant_id] = variant_sample_counts.get(result.variant_id, 0) + 1
            
            min_samples_met = all(
                variant_sample_counts.get(v.variant_id, 0) >= v.sample_size
                for v in test.variants
            )
            
            # Check if test duration has passed
            duration_passed = (datetime.now() - test.start_date).days >= self.metrics_config['test_duration_days']
            
            # If either condition is met, analyze the test
            if min_samples_met or duration_passed:
                await self._analyze_test_results(test_id)
                
        except Exception as e:
            self.logger.error(f"Failed to check test completion: {e}")
    
    async def _analyze_test_results(self, test_id: str):
        """Analyze A/B test results and determine winner"""
        try:
            test = self.active_tests[test_id]
            
            self.logger.info(f"Analyzing test results for: {test_id}")
            
            # Group results by variant
            variant_results = {}
            for result in test.results:
                if result.variant_id not in variant_results:
                    variant_results[result.variant_id] = []
                variant_results[result.variant_id].append(result)
            
            if len(variant_results) < 2:
                self.logger.warning(f"Not enough variants with data for test: {test_id}")
                return
            
            # Calculate metrics for each variant
            variant_metrics = {}
            for variant_id, results in variant_results.items():
                metrics = self._calculate_variant_metrics(results)
                variant_metrics[variant_id] = metrics
            
            # Determine statistical significance
            primary_metric = self.metrics_config['primary_metric']
            significance_result = self._calculate_statistical_significance(
                variant_metrics, primary_metric
            )
            
            # Update test with results
            test.statistical_significance = significance_result['significant']
            test.confidence_score = significance_result['confidence']
            test.status = TestStatus.COMPLETED
            test.end_date = datetime.now()
            
            if significance_result['significant']:
                test.winner_variant_id = significance_result['winner_variant_id']
                
                self.logger.info(f"Test {test_id} completed - Winner: {test.winner_variant_id}")
                
                # Generate recommendations based on results
                await self._generate_optimization_recommendations(test, variant_metrics)
            else:
                self.logger.info(f"Test {test_id} completed - No significant winner")
            
            # Save updated test
            await self._save_ab_test(test)
            
        except Exception as e:
            self.logger.error(f"Failed to analyze test results: {e}")
    
    def _calculate_variant_metrics(self, results: List[TestResult]) -> Dict[str, float]:
        """Calculate aggregated metrics for a variant"""
        if not results:
            return {}
        
        metrics = {
            'views': statistics.mean([r.views for r in results]),
            'clicks': statistics.mean([r.clicks for r in results]),
            'watch_time': statistics.mean([r.watch_time for r in results]),
            'engagement_rate': statistics.mean([r.engagement_rate for r in results]),
            'conversion_rate': statistics.mean([r.conversion_rate for r in results]),
            'retention_rate': statistics.mean([r.retention_rate for r in results]),
            'sample_size': len(results)
        }
        
        # Calculate additional derived metrics
        if metrics['views'] > 0:
            metrics['ctr'] = metrics['clicks'] / metrics['views']
        else:
            metrics['ctr'] = 0.0
        
        return metrics
    
    def _calculate_statistical_significance(self, 
                                          variant_metrics: Dict[str, Dict[str, float]],
                                          primary_metric: str) -> Dict[str, Any]:
        """Calculate statistical significance of test results"""
        try:
            # Simple statistical significance calculation
            # In a production system, this would use proper statistical tests
            
            variant_ids = list(variant_metrics.keys())
            if len(variant_ids) < 2:
                return {'significant': False, 'confidence': 0.0}
            
            # Get primary metric values
            metric_values = {}
            for variant_id in variant_ids:
                if primary_metric in variant_metrics[variant_id]:
                    metric_values[variant_id] = variant_metrics[variant_id][primary_metric]
            
            if len(metric_values) < 2:
                return {'significant': False, 'confidence': 0.0}
            
            # Find best performing variant
            best_variant = max(metric_values, key=metric_values.get)
            best_value = metric_values[best_variant]
            
            # Calculate improvement over other variants
            improvements = []
            for variant_id, value in metric_values.items():
                if variant_id != best_variant and value > 0:
                    improvement = (best_value - value) / value
                    improvements.append(improvement)
            
            if not improvements:
                return {'significant': False, 'confidence': 0.0}
            
            avg_improvement = statistics.mean(improvements)
            
            # Simple significance check based on improvement threshold
            minimum_improvement = self.metrics_config['minimum_improvement']
            is_significant = avg_improvement >= minimum_improvement
            
            # Calculate confidence score (simplified)
            confidence = min(avg_improvement * 10, 0.99)  # Scale to 0-0.99
            
            return {
                'significant': is_significant,
                'confidence': confidence,
                'winner_variant_id': best_variant if is_significant else None,
                'improvement': avg_improvement
            }
            
        except Exception as e:
            self.logger.error(f"Statistical significance calculation failed: {e}")
            return {'significant': False, 'confidence': 0.0}
    
    async def _generate_optimization_recommendations(self, 
                                                   test: ABTest,
                                                   variant_metrics: Dict[str, Dict[str, float]]):
        """Generate optimization recommendations based on test results"""
        try:
            if not test.winner_variant_id:
                return
            
            winner_variant = None
            for variant in test.variants:
                if variant.variant_id == test.winner_variant_id:
                    winner_variant = variant
                    break
            
            if not winner_variant:
                return
            
            # Create recommendation
            winner_metrics = variant_metrics[test.winner_variant_id]
            primary_metric = self.metrics_config['primary_metric']
            improvement = winner_metrics.get(primary_metric, 0)
            
            recommendation = {
                'recommendation_type': test.test_type.value,
                'priority_score': improvement * 100,  # Convert to percentage
                'description': f"Apply winning configuration from test {test.test_name}",
                'expected_impact': improvement,
                'implementation_complexity': 'low',
                'status': 'pending',
                'winning_configuration': winner_variant.configuration,
                'test_confidence': test.confidence_score
            }
            
            # Save recommendation to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO optimization_recommendations
                    (recommendation_type, priority_score, description, expected_impact,
                     implementation_complexity, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    recommendation['recommendation_type'],
                    recommendation['priority_score'],
                    recommendation['description'],
                    recommendation['expected_impact'],
                    recommendation['implementation_complexity'],
                    recommendation['status']
                ))
            
            self.logger.info(f"Generated optimization recommendation for test: {test.test_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate optimization recommendations: {e}")
    
    async def get_optimization_recommendations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get pending optimization recommendations"""
        try:
            recommendations = []
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT recommendation_type, priority_score, description, expected_impact,
                           implementation_complexity, status, created_at
                    FROM optimization_recommendations
                    WHERE status = 'pending'
                    ORDER BY priority_score DESC
                    LIMIT ?
                """, (limit,))
                
                for row in cursor.fetchall():
                    recommendation = {
                        'type': row[0],
                        'priority_score': row[1],
                        'description': row[2],
                        'expected_impact': row[3],
                        'complexity': row[4],
                        'status': row[5],
                        'created_at': row[6]
                    }
                    recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to get optimization recommendations: {e}")
            return []
    
    async def create_thumbnail_ab_test(self, 
                                     thumbnail_variants: List[str],
                                     test_name: str = "Thumbnail A/B Test") -> ABTest:
        """Create A/B test for thumbnails"""
        variants_config = []
        
        for i, thumbnail_path in enumerate(thumbnail_variants):
            variants_config.append({
                'name': f'Thumbnail {i+1}',
                'configuration': {'thumbnail_path': thumbnail_path},
                'expected_improvement': 0.1,
                'sample_size': 100
            })
        
        return await self.create_ab_test(
            TestType.THUMBNAIL,
            test_name,
            "Test different thumbnail designs for video performance",
            variants_config
        )
    
    async def create_title_ab_test(self, 
                                 title_variants: List[str],
                                 test_name: str = "Title A/B Test") -> ABTest:
        """Create A/B test for titles"""
        variants_config = []
        
        for i, title in enumerate(title_variants):
            variants_config.append({
                'name': f'Title {i+1}',
                'configuration': {'title': title},
                'expected_improvement': 0.08,
                'sample_size': 75
            })
        
        return await self.create_ab_test(
            TestType.TITLE,
            test_name,
            "Test different title formats for video performance",
            variants_config
        )
    
    async def create_upload_time_ab_test(self, 
                                       time_variants: List[str],
                                       test_name: str = "Upload Time A/B Test") -> ABTest:
        """Create A/B test for upload timing"""
        variants_config = []
        
        for i, upload_time in enumerate(time_variants):
            variants_config.append({
                'name': f'Time Slot {i+1}',
                'configuration': {'upload_time': upload_time},
                'expected_improvement': 0.15,
                'sample_size': 50
            })
        
        return await self.create_ab_test(
            TestType.UPLOAD_TIME,
            test_name,
            "Test different upload times for video performance",
            variants_config
        )
    
    async def get_test_analytics(self) -> Dict[str, Any]:
        """Get comprehensive A/B testing analytics"""
        try:
            analytics = {
                'active_tests': len([t for t in self.active_tests.values() if t.status == TestStatus.RUNNING]),
                'completed_tests': 0,
                'total_improvements': 0.0,
                'test_types_performance': {},
                'recent_winners': []
            }
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count completed tests
                cursor.execute("SELECT COUNT(*) FROM ab_tests WHERE status = 'completed'")
                analytics['completed_tests'] = cursor.fetchone()[0]
                
                # Get test type performance
                cursor.execute("""
                    SELECT test_type, AVG(confidence_score), COUNT(*) 
                    FROM ab_tests 
                    WHERE status = 'completed' AND statistical_significance = 1
                    GROUP BY test_type
                """)
                
                for row in cursor.fetchall():
                    analytics['test_types_performance'][row[0]] = {
                        'avg_confidence': row[1],
                        'successful_tests': row[2]
                    }
                
                # Get recent winners
                cursor.execute("""
                    SELECT test_name, test_type, confidence_score, end_date
                    FROM ab_tests 
                    WHERE status = 'completed' AND winner_variant_id IS NOT NULL
                    ORDER BY end_date DESC
                    LIMIT 5
                """)
                
                for row in cursor.fetchall():
                    analytics['recent_winners'].append({
                        'test_name': row[0],
                        'test_type': row[1],
                        'confidence': row[2],
                        'completed_date': row[3]
                    })
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Failed to get test analytics: {e}")
            return {}
    
    async def schedule_automated_tests(self):
        """Schedule automated A/B tests based on configuration"""
        try:
            current_time = datetime.now()
            
            for test_type_key, schedule_config in self.test_schedule.items():
                frequency = schedule_config.get('frequency', 'monthly')
                priority = schedule_config.get('priority', 'medium')
                
                # Check if it's time to create a new test
                should_create_test = await self._should_create_scheduled_test(
                    test_type_key, frequency, current_time
                )
                
                if should_create_test:
                    await self._create_automated_test(test_type_key, priority)
            
        except Exception as e:
            self.logger.error(f"Failed to schedule automated tests: {e}")
    
    async def _should_create_scheduled_test(self, 
                                          test_type_key: str,
                                          frequency: str,
                                          current_time: datetime) -> bool:
        """Check if a scheduled test should be created"""
        try:
            # Get last test of this type
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT MAX(start_date) FROM ab_tests 
                    WHERE test_type = ?
                """, (test_type_key.replace('_tests', ''),))
                
                result = cursor.fetchone()
                last_test_date = result[0] if result[0] else None
            
            if not last_test_date:
                return True  # No previous test, create one
            
            last_test_datetime = datetime.fromisoformat(last_test_date)
            
            # Calculate time since last test
            time_since_last = current_time - last_test_datetime
            
            # Check frequency requirements
            frequency_days = {
                'daily': 1,
                'weekly': 7,
                'bi-weekly': 14,
                'monthly': 30,
                'quarterly': 90
            }
            
            required_days = frequency_days.get(frequency, 30)
            
            return time_since_last.days >= required_days
            
        except Exception as e:
            self.logger.warning(f"Failed to check scheduled test timing: {e}")
            return False
    
    async def _create_automated_test(self, test_type_key: str, priority: str):
        """Create an automated test based on type and priority"""
        try:
            test_type_str = test_type_key.replace('_tests', '')
            test_type = TestType(test_type_str)
            
            self.logger.info(f"Creating automated {test_type.value} test")
            
            # Create appropriate test based on type
            if test_type == TestType.THUMBNAIL:
                # Generate thumbnail variants automatically
                variants = await self._generate_thumbnail_variants()
                if variants:
                    await self.create_thumbnail_ab_test(variants, f"Automated Thumbnail Test {datetime.now().strftime('%Y%m%d')}")
            
            elif test_type == TestType.TITLE:
                # Generate title variants automatically
                variants = await self._generate_title_variants()
                if variants:
                    await self.create_title_ab_test(variants, f"Automated Title Test {datetime.now().strftime('%Y%m%d')}")
            
            elif test_type == TestType.UPLOAD_TIME:
                # Generate upload time variants
                variants = await self._generate_upload_time_variants()
                if variants:
                    await self.create_upload_time_ab_test(variants, f"Automated Upload Time Test {datetime.now().strftime('%Y%m%d')}")
            
        except Exception as e:
            self.logger.error(f"Failed to create automated test: {e}")
    
    async def _generate_thumbnail_variants(self) -> List[str]:
        """Generate thumbnail variants for automated testing"""
        # This would integrate with the thumbnail generation system
        # For now, return placeholder variants
        return [
            "thumbnail_variant_1.jpg",
            "thumbnail_variant_2.jpg",
            "thumbnail_variant_3.jpg"
        ]
    
    async def _generate_title_variants(self) -> List[str]:
        """Generate title variants for automated testing"""
        # This would use AI to generate different title formats
        # For now, return placeholder variants
        return [
            "How to Master [Topic] in 60 Seconds",
            "[Topic] Mastery: The Ultimate Guide",
            "Amazing [Topic] Secrets Revealed"
        ]
    
    async def _generate_upload_time_variants(self) -> List[str]:
        """Generate upload time variants for automated testing"""
        return [
            "09:00",  # Morning
            "12:00",  # Lunch
            "16:00",  # Afternoon
            "19:00",  # Evening
            "21:00"   # Night
        ]