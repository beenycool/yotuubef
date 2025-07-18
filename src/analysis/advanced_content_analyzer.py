"""
Advanced Content Analysis System
Implements sophisticated content evaluation beyond simple trending analysis
"""

import asyncio
import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import sqlite3
import re

# Optional dependencies with fallbacks
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False

from src.config.settings import get_config


@dataclass
class ContentAnalysisResult:
    """Result of advanced content analysis"""
    score: float  # Overall content quality score (0-100)
    sentiment_score: float  # Sentiment analysis score (-1 to 1)
    trend_relevance: float  # Google Trends relevance score (0-100)
    uniqueness_score: float  # Uniqueness vs existing content (0-100)
    engagement_potential: float  # Predicted engagement score (0-100)
    keywords: List[str]  # Extracted keywords
    topics: List[str]  # Identified topics
    reasons: List[str]  # Explanation for score
    metadata: Dict[str, Any]  # Additional analysis data


class AdvancedContentAnalyzer:
    """
    Advanced content analysis system that evaluates content quality,
    sentiment, trending relevance, and uniqueness
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize database for uniqueness tracking
        self.db_path = Path("data/content_analysis.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = None
        self._init_sentiment_analyzer()
        
        # Initialize trend analyzer
        self.pytrends = None
        self._init_trend_analyzer()
        
        # Content scoring weights
        self.scoring_weights = {
            'sentiment': 0.25,      # 25% weight for sentiment
            'trends': 0.30,         # 30% weight for trend relevance
            'uniqueness': 0.25,     # 25% weight for uniqueness
            'engagement': 0.20      # 20% weight for engagement potential
        }
        
        self.logger.info("Advanced Content Analyzer initialized")
    
    def _init_database(self):
        """Initialize SQLite database for content tracking"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS content_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content_hash TEXT UNIQUE,
                        title TEXT,
                        description TEXT,
                        keywords TEXT,
                        topics TEXT,
                        analysis_score REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trending_keywords (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        keyword TEXT,
                        trend_score REAL,
                        search_volume INTEGER,
                        category TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_content_hash ON content_history(content_hash)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trending_keyword ON trending_keywords(keyword)
                """)
                
                self.logger.info("Content analysis database initialized")
                
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
    
    def _init_sentiment_analyzer(self):
        """Initialize sentiment analysis components"""
        try:
            if NLTK_AVAILABLE:
                # Download required NLTK data
                try:
                    nltk.data.find('vader_lexicon')
                except LookupError:
                    nltk.download('vader_lexicon', quiet=True)
                
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
                
                try:
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    nltk.download('stopwords', quiet=True)
                
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
                self.logger.info("NLTK sentiment analyzer initialized")
                
            elif TEXTBLOB_AVAILABLE:
                self.logger.info("TextBlob sentiment analyzer available")
            else:
                self.logger.warning("No sentiment analysis libraries available")
                
        except Exception as e:
            self.logger.warning(f"Sentiment analyzer initialization failed: {e}")
    
    def _init_trend_analyzer(self):
        """Initialize Google Trends analyzer"""
        try:
            if PYTRENDS_AVAILABLE:
                self.pytrends = TrendReq(hl='en-US', tz=360)
                self.logger.info("Google Trends analyzer initialized")
            else:
                self.logger.warning("PyTrends not available - trend analysis disabled")
                
        except Exception as e:
            self.logger.warning(f"Trend analyzer initialization failed: {e}")
    
    async def analyze_content(self, 
                            title: str, 
                            description: str = "", 
                            url: str = "",
                            metadata: Optional[Dict[str, Any]] = None) -> ContentAnalysisResult:
        """
        Perform comprehensive content analysis
        
        Args:
            title: Content title
            description: Content description
            url: Content URL
            metadata: Additional metadata (comments, subreddit, etc.)
            
        Returns:
            ContentAnalysisResult with detailed analysis
        """
        try:
            self.logger.info(f"Analyzing content: {title[:50]}...")
            
            # Combine text for analysis
            full_text = f"{title} {description}".strip()
            
            # Extract keywords and topics
            keywords = self._extract_keywords(full_text)
            topics = self._identify_topics(full_text, metadata)
            
            # Perform individual analyses
            sentiment_score = await self._analyze_sentiment(full_text)
            trend_relevance = await self._analyze_trend_relevance(keywords, topics)
            uniqueness_score = await self._analyze_uniqueness(full_text, title)
            engagement_potential = await self._predict_engagement(
                full_text, sentiment_score, trend_relevance, metadata
            )
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                sentiment_score, trend_relevance, uniqueness_score, engagement_potential
            )
            
            # Generate explanation
            reasons = self._generate_score_explanation(
                overall_score, sentiment_score, trend_relevance, 
                uniqueness_score, engagement_potential
            )
            
            # Store analysis in database
            await self._store_analysis(title, description, keywords, topics, overall_score)
            
            result = ContentAnalysisResult(
                score=overall_score,
                sentiment_score=sentiment_score,
                trend_relevance=trend_relevance,
                uniqueness_score=uniqueness_score,
                engagement_potential=engagement_potential,
                keywords=keywords,
                topics=topics,
                reasons=reasons,
                metadata={
                    'analysis_timestamp': datetime.now().isoformat(),
                    'analyzer_version': '1.0',
                    'full_text_length': len(full_text),
                    'keyword_count': len(keywords),
                    'topic_count': len(topics)
                }
            )
            
            self.logger.info(f"Content analysis complete - Score: {overall_score:.1f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Content analysis failed: {e}")
            # Return default result on failure
            return ContentAnalysisResult(
                score=50.0,
                sentiment_score=0.0,
                trend_relevance=50.0,
                uniqueness_score=50.0,
                engagement_potential=50.0,
                keywords=[],
                topics=[],
                reasons=["Analysis failed - using default scores"],
                metadata={'error': str(e)}
            )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        try:
            keywords = []
            
            if NLTK_AVAILABLE:
                # Use NLTK for keyword extraction
                tokens = word_tokenize(text.lower())
                stop_words = set(stopwords.words('english'))
                
                # Filter tokens
                keywords = [
                    token for token in tokens 
                    if (token.isalpha() and 
                        len(token) > 3 and 
                        token not in stop_words)
                ]
                
                # Get most frequent keywords
                from collections import Counter
                keyword_counts = Counter(keywords)
                keywords = [word for word, count in keyword_counts.most_common(10)]
                
            else:
                # Simple keyword extraction
                words = re.findall(r'\b\w{4,}\b', text.lower())
                common_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'than', 'more'}
                keywords = [word for word in set(words) if word not in common_words][:10]
            
            return keywords
            
        except Exception as e:
            self.logger.warning(f"Keyword extraction failed: {e}")
            return []
    
    def _identify_topics(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Identify topics from text and metadata"""
        try:
            topics = []
            
            # Define topic keywords
            topic_mapping = {
                'technology': ['tech', 'ai', 'software', 'computer', 'digital', 'app', 'code'],
                'entertainment': ['funny', 'comedy', 'movie', 'show', 'entertainment', 'celebrity'],
                'gaming': ['game', 'gaming', 'player', 'xbox', 'playstation', 'nintendo'],
                'science': ['science', 'research', 'study', 'experiment', 'discovery'],
                'sports': ['sport', 'football', 'basketball', 'soccer', 'tennis', 'athlete'],
                'lifestyle': ['life', 'lifestyle', 'health', 'fitness', 'food', 'travel'],
                'education': ['learn', 'education', 'tutorial', 'guide', 'how', 'teach'],
                'news': ['news', 'breaking', 'update', 'report', 'current', 'today'],
                'business': ['business', 'money', 'investment', 'market', 'finance', 'startup']
            }
            
            text_lower = text.lower()
            
            # Check for topic keywords
            for topic, keywords in topic_mapping.items():
                if any(keyword in text_lower for keyword in keywords):
                    topics.append(topic)
            
            # Add subreddit as topic if available
            if metadata and 'subreddit' in metadata:
                subreddit = metadata['subreddit'].lower()
                if subreddit not in topics:
                    topics.append(f"subreddit_{subreddit}")
            
            return topics[:5]  # Return top 5 topics
            
        except Exception as e:
            self.logger.warning(f"Topic identification failed: {e}")
            return []
    
    async def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of content (-1 to 1)"""
        try:
            if self.sentiment_analyzer and NLTK_AVAILABLE:
                # Use NLTK VADER sentiment analyzer
                scores = self.sentiment_analyzer.polarity_scores(text)
                return scores['compound']  # Returns value between -1 and 1
                
            elif TEXTBLOB_AVAILABLE:
                # Use TextBlob sentiment analyzer
                blob = TextBlob(text)
                return blob.sentiment.polarity  # Returns value between -1 and 1
                
            else:
                # Simple rule-based sentiment
                positive_words = ['good', 'great', 'excellent', 'amazing', 'awesome', 'best', 'love', 'perfect']
                negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'stupid', 'annoying']
                
                text_lower = text.lower()
                positive_count = sum(1 for word in positive_words if word in text_lower)
                negative_count = sum(1 for word in negative_words if word in text_lower)
                
                if positive_count + negative_count == 0:
                    return 0.0
                
                return (positive_count - negative_count) / (positive_count + negative_count)
                
        except Exception as e:
            self.logger.warning(f"Sentiment analysis failed: {e}")
            return 0.0  # Neutral sentiment on failure
    
    async def _analyze_trend_relevance(self, keywords: List[str], topics: List[str]) -> float:
        """Analyze trend relevance using Google Trends (0-100)"""
        try:
            if not self.pytrends or not keywords:
                return 50.0  # Default score when trends unavailable
            
            trend_scores = []
            
            # Analyze keywords in batches
            keyword_batches = [keywords[i:i+4] for i in range(0, len(keywords), 4)]
            
            for batch in keyword_batches:
                try:
                    # Query Google Trends
                    self.pytrends.build_payload(batch, cat=0, timeframe='today 3-m', geo='US')
                    interest_data = self.pytrends.interest_over_time()
                    
                    if not interest_data.empty:
                        # Calculate average interest for this batch
                        for keyword in batch:
                            if keyword in interest_data.columns:
                                avg_interest = interest_data[keyword].mean()
                                trend_scores.append(avg_interest)
                    
                    # Small delay to avoid rate limiting
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    self.logger.warning(f"Trend analysis batch failed: {e}")
                    continue
            
            if trend_scores:
                return min(max(sum(trend_scores) / len(trend_scores), 0), 100)
            else:
                return 50.0  # Default when no trend data
                
        except Exception as e:
            self.logger.warning(f"Trend relevance analysis failed: {e}")
            return 50.0
    
    async def _analyze_uniqueness(self, text: str, title: str) -> float:
        """Analyze content uniqueness against existing content (0-100)"""
        try:
            # Generate content hash
            content_hash = hashlib.md5(text.encode()).hexdigest()
            
            # Check against database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check for exact matches
                cursor.execute(
                    "SELECT COUNT(*) FROM content_history WHERE content_hash = ?",
                    (content_hash,)
                )
                exact_matches = cursor.fetchone()[0]
                
                if exact_matches > 0:
                    return 0.0  # Not unique at all
                
                # Check for similar titles
                cursor.execute(
                    "SELECT title FROM content_history WHERE created_at > ?",
                    (datetime.now() - timedelta(days=30),)
                )
                recent_titles = cursor.fetchall()
                
                # Simple similarity check
                similar_count = 0
                title_words = set(title.lower().split())
                
                for (stored_title,) in recent_titles:
                    stored_words = set(stored_title.lower().split())
                    similarity = len(title_words.intersection(stored_words)) / len(title_words.union(stored_words))
                    
                    if similarity > 0.7:  # 70% similarity threshold
                        similar_count += 1
                
                # Calculate uniqueness score
                if similar_count == 0:
                    return 100.0
                elif similar_count <= 2:
                    return 75.0
                elif similar_count <= 5:
                    return 50.0
                else:
                    return 25.0
                
        except Exception as e:
            self.logger.warning(f"Uniqueness analysis failed: {e}")
            return 75.0  # Default to high uniqueness on failure
    
    async def _predict_engagement(self, 
                                text: str, 
                                sentiment_score: float, 
                                trend_relevance: float,
                                metadata: Optional[Dict[str, Any]] = None) -> float:
        """Predict engagement potential (0-100)"""
        try:
            engagement_score = 50.0  # Base score
            
            # Sentiment impact
            if sentiment_score > 0.3:
                engagement_score += 15  # Positive content tends to engage
            elif sentiment_score < -0.3:
                engagement_score += 10  # Negative content can also engage (controversy)
            
            # Trend relevance impact
            engagement_score += (trend_relevance - 50) * 0.3
            
            # Content length impact
            text_length = len(text)
            if 50 <= text_length <= 200:
                engagement_score += 10  # Optimal length
            elif text_length > 500:
                engagement_score -= 5  # Too long
            
            # Keywords that drive engagement
            engagement_keywords = [
                'amazing', 'incredible', 'shocking', 'unbelievable', 'must', 'watch',
                'viral', 'trending', 'breaking', 'exclusive', 'secret', 'hidden'
            ]
            
            text_lower = text.lower()
            keyword_bonus = sum(5 for keyword in engagement_keywords if keyword in text_lower)
            engagement_score += min(keyword_bonus, 20)  # Max 20 points from keywords
            
            # Metadata impact
            if metadata:
                # Subreddit reputation
                popular_subreddits = ['funny', 'videos', 'gifs', 'nextlevel', 'oddlysatisfying']
                if metadata.get('subreddit', '').lower() in popular_subreddits:
                    engagement_score += 10
                
                # Comment count impact
                comment_count = metadata.get('num_comments', 0)
                if comment_count > 100:
                    engagement_score += 15
                elif comment_count > 50:
                    engagement_score += 10
                elif comment_count > 20:
                    engagement_score += 5
            
            return min(max(engagement_score, 0), 100)
            
        except Exception as e:
            self.logger.warning(f"Engagement prediction failed: {e}")
            return 50.0
    
    def _calculate_overall_score(self, 
                               sentiment: float, 
                               trends: float, 
                               uniqueness: float, 
                               engagement: float) -> float:
        """Calculate weighted overall content score"""
        # Normalize sentiment from [-1,1] to [0,100]
        sentiment_normalized = (sentiment + 1) * 50
        
        # Calculate weighted score
        overall_score = (
            sentiment_normalized * self.scoring_weights['sentiment'] +
            trends * self.scoring_weights['trends'] +
            uniqueness * self.scoring_weights['uniqueness'] +
            engagement * self.scoring_weights['engagement']
        )
        
        return min(max(overall_score, 0), 100)
    
    def _generate_score_explanation(self, 
                                  overall: float,
                                  sentiment: float,
                                  trends: float,
                                  uniqueness: float,
                                  engagement: float) -> List[str]:
        """Generate human-readable explanation for the score"""
        reasons = []
        
        # Overall assessment
        if overall >= 80:
            reasons.append("Excellent content with high potential")
        elif overall >= 60:
            reasons.append("Good content with solid potential")
        elif overall >= 40:
            reasons.append("Average content with moderate potential")
        else:
            reasons.append("Below average content")
        
        # Sentiment assessment
        if sentiment > 0.3:
            reasons.append("Positive sentiment boosts engagement")
        elif sentiment < -0.3:
            reasons.append("Negative sentiment may create controversy")
        else:
            reasons.append("Neutral sentiment")
        
        # Trend assessment
        if trends >= 70:
            reasons.append("Highly relevant to current trends")
        elif trends >= 50:
            reasons.append("Moderately trending topic")
        else:
            reasons.append("Low trend relevance")
        
        # Uniqueness assessment
        if uniqueness >= 80:
            reasons.append("Highly unique content")
        elif uniqueness >= 50:
            reasons.append("Moderately unique content")
        else:
            reasons.append("Similar to existing content")
        
        # Engagement assessment
        if engagement >= 70:
            reasons.append("High engagement potential")
        elif engagement >= 50:
            reasons.append("Moderate engagement potential")
        else:
            reasons.append("Low engagement potential")
        
        return reasons
    
    async def _store_analysis(self, 
                            title: str, 
                            description: str, 
                            keywords: List[str], 
                            topics: List[str], 
                            score: float):
        """Store analysis results in database"""
        try:
            content_hash = hashlib.md5(f"{title} {description}".encode()).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO content_history 
                    (content_hash, title, description, keywords, topics, analysis_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    content_hash,
                    title,
                    description,
                    json.dumps(keywords),
                    json.dumps(topics),
                    score
                ))
                
        except Exception as e:
            self.logger.warning(f"Failed to store analysis: {e}")
    
    async def get_content_recommendations(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get content recommendations based on analysis history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT title, keywords, topics, analysis_score, created_at
                    FROM content_history
                    WHERE analysis_score >= 60
                    ORDER BY analysis_score DESC, created_at DESC
                    LIMIT ?
                """, (count,))
                
                results = cursor.fetchall()
                
                recommendations = []
                for title, keywords_json, topics_json, score, created_at in results:
                    recommendations.append({
                        'title': title,
                        'keywords': json.loads(keywords_json),
                        'topics': json.loads(topics_json),
                        'score': score,
                        'created_at': created_at
                    })
                
                return recommendations
                
        except Exception as e:
            self.logger.error(f"Failed to get recommendations: {e}")
            return []
    
    async def analyze_batch_content(self, 
                                  content_list: List[Dict[str, Any]]) -> List[ContentAnalysisResult]:
        """Analyze multiple content items in batch"""
        tasks = []
        
        for content in content_list:
            task = self.analyze_content(
                title=content.get('title', ''),
                description=content.get('description', ''),
                url=content.get('url', ''),
                metadata=content.get('metadata', {})
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)