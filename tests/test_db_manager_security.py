import pytest
import sqlite3
from pathlib import Path
from src.database.db_manager import DatabaseManager


def test_get_processing_stats_sql_injection(tmp_path):
    """
    Test that get_processing_stats is immune to SQL injection.
    It should safely cast to integer or handle malicious input without executing it.
    """
    db_path = tmp_path / "test.db"
    db_manager = DatabaseManager(db_path)

    with pytest.raises(ValueError):
        db_manager.get_processing_stats(days="30') OR 1=1; DROP TABLE uploads; --")

    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='uploads'"
        )
        count = cursor.fetchone()[0]
        assert count == 1, "SQL Injection was successful: uploads table was dropped!"


def test_cleanup_old_records_sql_injection(tmp_path):
    """
    Test that cleanup_old_records is immune to SQL injection.
    """
    db_path = tmp_path / "test.db"
    db_manager = DatabaseManager(db_path)

    with pytest.raises(ValueError):
        db_manager.cleanup_old_records(
            days_to_keep="30') OR 1=1; DROP TABLE processing_history; --"
        )

    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='processing_history'"
        )
        count = cursor.fetchone()[0]
        assert count == 1, (
            "SQL Injection was successful: processing_history table was dropped!"
        )

def test_update_upload_status_security(tmp_path):
    """
    Test that update_upload_status is immune to SQL injection.
    It should securely update the values without executing injected SQL.
    """
    db_path = tmp_path / "test.db"
    db_manager = DatabaseManager(db_path)

    # Setup: create an initial upload record
    upload_id = db_manager.record_upload(
        reddit_url="https://reddit.com/r/test",
        reddit_post_id="test_post_1",
        title="Test Video",
        subreddit="test",
        original_score=100
    )
    assert upload_id is not None

    # Malicious input string designed to execute additional SQL commands
    malicious_input = "COMPLETED'; DROP TABLE uploads; --"

    # Attempt to update the upload with the malicious input as a status or error message
    db_manager.update_upload_status(
        upload_id=upload_id,
        status="FAILED",
        error_message=malicious_input,
        youtube_url=malicious_input,
        youtube_video_id=malicious_input
    )

    # Verify the table still exists and data is securely saved as literal strings
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()

        # Check if table still exists
        cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='uploads'")
        count = cursor.fetchone()[0]
        assert count == 1, "SQL Injection was successful: uploads table was dropped!"

        # Check that the data was actually saved as the literal malicious string
        cursor.execute("SELECT error_message, youtube_url, youtube_video_id FROM uploads WHERE id = ?", (upload_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == malicious_input
        assert row[1] == malicious_input
        assert row[2] == malicious_input
