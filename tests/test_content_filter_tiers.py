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

def test_contains_forbidden_words(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
content:
  hard_disallowed: []
  forbidden_words:
    - badword
    - toxic
  unsuitable_content_types: []
  demonetization_risk: []
  caution: []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    init_config(config_path)

    content_filter = ContentFilter()

    # None and empty strings
    assert content_filter.contains_forbidden_words(None) is False
    assert content_filter.contains_forbidden_words("") is False
    assert content_filter.contains_forbidden_words("   ") is False

    # Clean strings
    assert content_filter.contains_forbidden_words("This is a clean post") is False
    assert content_filter.contains_forbidden_words("Just some regular text") is False

    # Exact match
    assert content_filter.contains_forbidden_words("This contains a badword") is True
    assert content_filter.contains_forbidden_words("toxic community") is True

    # Case insensitivity
    assert content_filter.contains_forbidden_words("This contains a BADWORD") is True
    assert content_filter.contains_forbidden_words("ToXiC player") is True

    # Substring boundaries (should not match if it's part of another word)
    assert content_filter.contains_forbidden_words("intoxicating perfume") is False
    assert content_filter.contains_forbidden_words("notabadwordhere") is False

    # Punctuation
    assert content_filter.contains_forbidden_words("badword!") is True
    assert content_filter.contains_forbidden_words("Wait, toxic?") is True
    assert content_filter.contains_forbidden_words(".badword.") is True
