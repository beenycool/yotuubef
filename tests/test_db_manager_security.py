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
