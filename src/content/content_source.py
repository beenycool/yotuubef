"""
ContentSource class responsible for finding and pre-analyzing content from Reddit
"""

import logging
import asyncio
import random
from typing import Dict, List, Any
from datetime import datetime


class ContentSource:
    """
    Handles content sourcing from Reddit.
    Single responsibility: Find and pre-analyze content for video generation.
    """
    
    def __init__(self, reddit_client=None, content_analyzer=None):
        self.logger = logging.getLogger(__name__)
        self.reddit_client = reddit_client
        self.content_analyzer = content_analyzer
        
        # Content sourcing settings
        self.subreddits = [
            'AskReddit', 'TrueOffMyChest', 'relationship_advice', 
            'AmItheAsshole', 'confession', 'unpopularopinion',
            'LifeProTips', 'todayilearned', 'Showerthoughts'
        ]
        self.min_content_score = 100
        self.max_content_age_hours = 24
        
    async def find_and_analyze_content(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Find and pre-analyze content from Reddit sources.
        
        Args:
            max_items: Maximum number of content items to return
            
        Returns:
            List of analyzed content items suitable for video generation
        """
        try:
            analyzed_content = []
            
            if not self.reddit_client:
                self.logger.warning("No Reddit client available, using simulated content")
                return await self._get_simulated_content(max_items)
                
            # Search across multiple subreddits
            for subreddit_name in self.subreddits[:3]:  # Limit to 3 subreddits for efficiency
                try:
                    content_items = await self._get_subreddit_content(subreddit_name, max_items // 3)
                    analyzed_content.extend(content_items)
                    
                    if len(analyzed_content) >= max_items:
                        break
                        
                except Exception as e:
                    self.logger.warning(f"Failed to get content from r/{subreddit_name}: {e}")
                    continue
                    
            # Sort by analysis score and return top items
            analyzed_content.sort(key=lambda x: x.get('analysis', {}).get('overall_score', 0), reverse=True)
            return analyzed_content[:max_items]
            
        except Exception as e:
            self.logger.error(f"Content finding and analysis failed: {e}")
            return await self._get_simulated_content(max_items)
            
    async def _get_subreddit_content(self, subreddit_name: str, max_items: int) -> List[Dict[str, Any]]:
        """
        Get content from a specific subreddit and analyze it.
        
        Args:
            subreddit_name: Name of the subreddit
            max_items: Maximum items to retrieve
            
        Returns:
            List of analyzed content items
        """
        try:
            # Get posts from Reddit
            posts = await self._fetch_reddit_posts(subreddit_name, max_items * 2)  # Get more to filter
            
            analyzed_posts = []
            for post in posts:
                # Basic filtering
                if not self._is_suitable_content(post):
                    continue
                    
                # Analyze content if analyzer is available
                analysis = None
                if self.content_analyzer:
                    try:
                        analysis = await self.content_analyzer.analyze_content(
                            title=post.get('title', ''),
                            description=post.get('selftext', ''),
                            metadata={'subreddit': subreddit_name, 'score': post.get('score', 0)}
                        )
                    except Exception as e:
                        self.logger.warning(f"Content analysis failed for post: {e}")
                        
                analyzed_posts.append({
                    'content': post,
                    'analysis': analysis,
                    'source': f"r/{subreddit_name}",
                    'retrieved_at': datetime.now().isoformat()
                })
                
                if len(analyzed_posts) >= max_items:
                    break
                    
            return analyzed_posts
            
        except Exception as e:
            self.logger.error(f"Failed to get content from r/{subreddit_name}: {e}")
            return []
            
    async def _fetch_reddit_posts(self, subreddit_name: str, max_items: int) -> List[Dict[str, Any]]:
        """
        Fetch posts from Reddit API.
        
        Args:
            subreddit_name: Name of the subreddit
            max_items: Maximum posts to fetch
            
        Returns:
            List of Reddit posts
        """
        if not self.reddit_client or not hasattr(self.reddit_client, 'get_posts'):
            # Simulate Reddit API call
            await asyncio.sleep(0.5)  # Simulate network delay
            return self._generate_sample_posts(subreddit_name, max_items)
            
        try:
            return await self.reddit_client.get_posts(subreddit_name, limit=max_items)
        except Exception as e:
            self.logger.warning(f"Reddit API call failed: {e}")
            return self._generate_sample_posts(subreddit_name, max_items)
            
    def _is_suitable_content(self, post: Dict[str, Any]) -> bool:
        """
        Check if a post is suitable for video generation.
        
        Args:
            post: Reddit post data
            
        Returns:
            bool: True if suitable for video generation
        """
        # Check score threshold
        score = post.get('score', 0)
        if score < self.min_content_score:
            return False
            
        # Check content length
        title = post.get('title', '')
        selftext = post.get('selftext', '')
        
        if len(title) < 10:  # Too short title
            return False
            
        # Check for suitable content type
        if post.get('is_video', False) or post.get('url', '').endswith(('.jpg', '.png', '.gif')):
            return False  # Skip media posts for text-based videos
            
        return True
        
    def _generate_sample_posts(self, subreddit_name: str, count: int) -> List[Dict[str, Any]]:
        """
        Generate sample posts for testing/simulation.
        
        Args:
            subreddit_name: Name of the subreddit
            count: Number of sample posts to generate
            
        Returns:
            List of sample Reddit posts
        """
        sample_titles = [
            "What's the most life-changing advice you've ever received?",
            "People who changed careers after 30, what's your story?",
            "What's a red flag in job interviews that candidates should watch for?",
            "What's something you wish you knew before starting college?",
            "What's the weirdest thing a guest has done at your house?",
            "What's a skill everyone should learn but most people don't?",
            "What's the best purchase you've made under $100?",
            "What's something that sounds fake but is actually true?",
        ]
        
        posts = []
        for i in range(count):
            posts.append({
                'title': random.choice(sample_titles),
                'selftext': f"Sample content for r/{subreddit_name} post {i+1}",
                'score': random.randint(self.min_content_score, 5000),
                'author': f"sample_user_{i}",
                'created_utc': datetime.now().timestamp(),
                'num_comments': random.randint(10, 500),
                'subreddit': subreddit_name,
                'id': f"sample_{subreddit_name}_{i}",
                'url': f"https://reddit.com/r/{subreddit_name}/comments/sample_{i}"
            })
            
        return posts
        
    async def _get_simulated_content(self, max_items: int) -> List[Dict[str, Any]]:
        """
        Get simulated content for testing when Reddit client is not available.
        
        Args:
            max_items: Maximum items to return
            
        Returns:
            List of simulated content items
        """
        await asyncio.sleep(0.1)  # Simulate processing time
        
        simulated_content = []
        for i in range(max_items):
            content = {
                'content': self._generate_sample_posts('simulated', 1)[0],
                'analysis': {
                    'overall_score': random.uniform(0.6, 0.9),
                    'sentiment_score': random.uniform(0.3, 0.7),
                    'keywords': ['advice', 'life', 'experience'],
                    'estimated_engagement': random.randint(50, 200)
                },
                'source': 'simulated',
                'retrieved_at': datetime.now().isoformat()
            }
            simulated_content.append(content)
            
        return simulated_content