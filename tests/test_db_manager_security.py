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
        original_score=100,
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
        youtube_video_id=malicious_input,
    )

    # Verify the table still exists and data is securely saved as literal strings
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()

        # Check if table still exists
        cursor.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='uploads'"
        )
        count = cursor.fetchone()[0]
        assert count == 1, "SQL Injection was successful: uploads table was dropped!"

        # Check that the data was actually saved as the literal malicious string
        cursor.execute(
            "SELECT error_message, youtube_url, youtube_video_id FROM uploads WHERE id = ?",
            (upload_id,),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == malicious_input
        assert row[1] == malicious_input
        assert row[2] == malicious_input


def test_cleanup_old_records_removes_stale_uploads_and_processing_history(tmp_path):
    db_path = tmp_path / "test.db"
    db_manager = DatabaseManager(db_path)

    upload_id = db_manager.record_upload(
        reddit_url="https://reddit.com/r/test/comments/cleanup/post",
        reddit_post_id="cleanup_post",
        title="Cleanup Video",
        subreddit="test",
        original_score=42,
    )
    assert upload_id is not None

    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE uploads SET upload_timestamp = datetime('now', '-120 days') WHERE id = ?",
            (upload_id,),
        )
        cursor.execute(
            "INSERT INTO processing_history (reddit_url, processing_status, message, created_at) VALUES (?, ?, ?, datetime('now', '-120 days'))",
            ("https://reddit.com/r/test/comments/cleanup/post", "completed", "old run"),
        )
        cursor.execute(
            "INSERT INTO processing_history (reddit_url, processing_status, message, created_at) VALUES (?, ?, ?, datetime('now'))",
            (
                "https://reddit.com/r/test/comments/current/post",
                "completed",
                "current run",
            ),
        )
        conn.commit()

    deleted = db_manager.cleanup_old_records(days_to_keep=90)

    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM uploads")
        remaining_uploads = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM processing_history")
        remaining_history = cursor.fetchone()[0]

    assert deleted["uploads_deleted"] == 1
    assert deleted["processing_history_deleted"] == 1
    assert remaining_uploads == 0
    assert remaining_history == 1
