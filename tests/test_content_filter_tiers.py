from datetime import datetime, timezone

from src.config.settings import init_config
from src.integrations.reddit_client import ContentFilter, RedditPost


def _post_with_title(title: str) -> RedditPost:
    return RedditPost(
        id="abc123",
        title=title,
        url="https://example.com/video.mp4",
        subreddit="interesting",
        author="author",
        score=200,
        upvote_ratio=0.98,
        num_comments=10,
        created_utc=datetime.now(timezone.utc),
        is_video=True,
        selftext="",
        video_url="https://example.com/video.mp4",
        duration=30,
    )


def test_content_filter_blocks_hard_disallowed(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
content:
  hard_disallowed:
    - nsfw
  demonetization_risk: []
  caution: []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    init_config(config_path)

    content_filter = ContentFilter()
    result = content_filter.is_suitable_for_monetization(_post_with_title("nsfw clip"))

    assert result["is_suitable"] is False
    assert result["blocking_reasons"]


def test_content_filter_flags_demonetization_without_blocking(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
content:
  hard_disallowed: []
  forbidden_words: []
  unsuitable_content_types: []
  demonetization_risk:
    - violence
  caution: []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    init_config(config_path)

    content_filter = ContentFilter()
    result = content_filter.is_suitable_for_monetization(
        _post_with_title("violence in movie scene")
    )

    assert result["is_suitable"] is True
    assert result["demonetization_risk"]

def test_content_filter_filter_posts(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
content:
  hard_disallowed:
    - nsfw
    - banned
  demonetization_risk:
    - violence
  caution: []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    init_config(config_path)

    content_filter = ContentFilter()

    post1 = _post_with_title("a perfectly fine clip")
    post2 = _post_with_title("nsfw clip")
    post3 = _post_with_title("violence in movie scene")
    post4 = _post_with_title("banned video")

    posts = [post1, post2, post3, post4]
    filtered_posts = content_filter.filter_posts(posts)

    # Should keep post1 (fine) and post3 (demonetization_risk only flags, doesn't block)
    assert len(filtered_posts) == 2
    assert post1 in filtered_posts
    assert post3 in filtered_posts
    assert post2 not in filtered_posts
    assert post4 not in filtered_posts
