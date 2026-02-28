from src.database.db_manager import DatabaseManager
from pathlib import Path

db_path = Path("artifacts.db")
db_manager = DatabaseManager(db_path=db_path)

db_manager.record_local_artifacts(
    reddit_url="https://reddit.com/r/test/comments/abc123/post",
    local_project_id="project-abc123",
    local_video_path="mapped_video.mp4",
    local_thumbnail_paths=["thumb.png"],
    youtube_video_id="yt_video_123",
)

artifacts = db_manager.get_local_artifacts_by_video_id("yt_video_123")
print("Artifacts:", artifacts)
