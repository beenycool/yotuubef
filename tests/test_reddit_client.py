from datetime import datetime, timezone
from types import SimpleNamespace
import uuid
import pytest

from src.config.settings import init_config
from src.integrations.reddit_client import RedditPost


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
    """Placeholder: ContentFilter was removed in hybrid-only restructure."""
    pass


def test_reddit_post_from_submission_extracts_fps_from_dict_media():
    submission = SimpleNamespace(
        id="abc123",
        title="A reddit hosted video",
        url="https://v.redd.it/example",
        subreddit=SimpleNamespace(display_name="interesting"),
        author="author",
        score=123,
        upvote_ratio=0.97,
        num_comments=9,
        created_utc=datetime.now(timezone.utc).timestamp(),
        is_video=True,
        media={
            "reddit_video": {
                "fallback_url": "https://v.redd.it/example/DASH_720.mp4",
                "duration": 12.0,
                "width": 1080,
                "height": 1920,
                "fps": 24.0,
            }
        },
        thumbnail="https://example.com/thumb.jpg",
        over_18=False,
        spoiler=False,
        selftext="",
    )

    post = RedditPost.from_submission(submission)

    assert post.is_video is True
    assert post.video_url == "https://v.redd.it/example/DASH_720.mp4"
    assert post.fps == 24.0
