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

    # Attempt SQL injection through the days parameter
    malicious_input = "30') OR 1=1; DROP TABLE uploads; --"

    # Depending on how the method handles invalid strings for `days`,
    # it might raise a ValueError when calling int(days),
    # but it shouldn't execute the DROP TABLE command.
    try:
        db_manager.get_processing_stats(days=malicious_input)
    except ValueError:
        # Expected if it tries to int(malicious_input)
        pass
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")

    # Verify the table still exists
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='uploads'")
        count = cursor.fetchone()[0]
        assert count == 1, "SQL Injection was successful: uploads table was dropped!"

def test_cleanup_old_records_sql_injection(tmp_path):
    """
    Test that cleanup_old_records is immune to SQL injection.
    """
    db_path = tmp_path / "test.db"
    db_manager = DatabaseManager(db_path)

    malicious_input = "30') OR 1=1; DROP TABLE processing_history; --"

    try:
        db_manager.cleanup_old_records(days_to_keep=malicious_input)
    except ValueError:
        pass
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")

    # Verify the table still exists
    with db_manager.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='processing_history'")
        count = cursor.fetchone()[0]
        assert count == 1, "SQL Injection was successful: processing_history table was dropped!"
