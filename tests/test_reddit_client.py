from datetime import datetime, timezone
import uuid
import pytest

from src.config.settings import init_config
from src.integrations.reddit_client import ContentFilter, RedditPost


def _post_with_title(
    title: str, is_video: bool = True, duration: float = 30
) -> RedditPost:
    return RedditPost(
        id=str(uuid.uuid4()),
        title=title,
        url="https://example.com/video.mp4",
        subreddit="interesting",
        author="author",
        score=200,
        upvote_ratio=0.98,
        num_comments=10,
        created_utc=datetime.now(timezone.utc),
        is_video=is_video,
        selftext="",
        video_url="https://example.com/video.mp4" if is_video else None,
        duration=duration,
        nsfw=False,
        spoiler=False,
        width=1080,
        height=1920,
        fps=30,
    )


def test_content_filter_filter_posts_mixed_suitability(tmp_path):
    """Test filtering posts with a mix of suitable, blocked, and flagged content."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
content:
  hard_disallowed:
    - nsfw
    - banned
  demonetization_risk:
    - violence
  caution:
    - edge case
  forbidden_words: []
  unsuitable_content_types: []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    init_config(config_path)

    content_filter = ContentFilter()

    post_suitable = _post_with_title("a perfectly fine clip")
    post_blocked = _post_with_title("nsfw clip")
    post_flagged_demo = _post_with_title("violence in movie scene")
    post_flagged_caution = _post_with_title("edge case video")
    post_blocked_2 = _post_with_title("banned video content")

    posts = [
        post_suitable,
        post_blocked,
        post_flagged_demo,
        post_flagged_caution,
        post_blocked_2,
    ]
    filtered_posts = content_filter.filter_posts(posts)

    # Should keep suitable and flagged (demonetization_risk or caution flag, but don't block)
    assert len(filtered_posts) == 3
    assert post_suitable in filtered_posts
    assert post_flagged_demo in filtered_posts
    assert post_flagged_caution in filtered_posts

    # Should block hard_disallowed
    assert post_blocked not in filtered_posts
    assert post_blocked_2 not in filtered_posts


def test_content_filter_filter_posts_quality_blocks(tmp_path):
    """Test filtering posts that fail quality metrics."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
content:
  hard_disallowed: []
  demonetization_risk: []
  caution: []
  forbidden_words: []
  unsuitable_content_types: []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    init_config(config_path)

    content_filter = ContentFilter()

    post_good = _post_with_title("good video")

    # Low score
    post_low_score = _post_with_title("low score")
    post_low_score.score = 5

    # Low upvote ratio
    post_controversial = _post_with_title("controversial")
    post_controversial.upvote_ratio = 0.5

    # Too short
    post_short = _post_with_title("too short", duration=2)

    # Low res
    post_low_res = _post_with_title("low res")
    post_low_res.width = 480
    post_low_res.height = 360

    # Too long
    post_long = _post_with_title("too long", duration=301)

    # Low FPS
    post_low_fps = _post_with_title("low fps")
    post_low_fps.fps = 20

    posts = [post_good, post_low_score, post_controversial, post_short, post_low_res, post_long, post_low_fps]
    filtered_posts = content_filter.filter_posts(posts)

    assert len(filtered_posts) == 1
    assert filtered_posts[0] == post_good
