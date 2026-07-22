from datetime import datetime, timezone
from types import SimpleNamespace
import uuid
import pytest

from src.integrations.reddit_client import RedditPost


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
