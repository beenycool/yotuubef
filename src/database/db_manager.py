"""
Database management for tracking uploaded videos and processing history.
Handles SQLite operations with proper error handling and migrations.
"""

import sqlite3
import logging
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from contextlib import contextmanager

from src.config.settings import get_config


class DatabaseManager:
    """Manages the SQLite database for video tracking"""

    def __init__(self, db_path: Optional[Path] = None):
        self.config = get_config()
        self.db_path = db_path or self.config.paths.db_file
        self.logger = logging.getLogger(__name__)

        self._initialize_database()

    def _initialize_database(self):
        """Initialize database with required tables"""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Create uploads table with comprehensive schema including A/B testing support
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS uploads (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        reddit_url TEXT UNIQUE NOT NULL,
                        reddit_post_id TEXT,
                        youtube_url TEXT,
                        youtube_video_id TEXT,
                        upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        title TEXT,
                        subreddit TEXT,
                        original_score INTEGER,
                        processing_duration_seconds REAL,
                        video_duration_seconds REAL,
                        file_size_mb REAL,
                        status TEXT DEFAULT 'completed',
                        error_message TEXT,
                        thumbnail_uploaded BOOLEAN DEFAULT FALSE,
                        ai_analysis_used BOOLEAN DEFAULT FALSE,
                        thumbnail_ctr_a REAL,
                        thumbnail_ctr_b REAL,
                        active_thumbnail TEXT DEFAULT 'A',
                        thumbnail_test_start_date DATETIME,
                        thumbnail_test_complete BOOLEAN DEFAULT FALSE,
                        winning_thumbnail TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                self._ensure_local_artifacts_table(cursor)

                conn.commit()

                # Check and perform migrations
                self._migrate_database(cursor)

                # Create indexes for performance (after migrations)
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_uploads_reddit_url ON uploads(reddit_url)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_uploads_youtube_video_id ON uploads(youtube_video_id)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_uploads_subreddit ON uploads(subreddit)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_uploads_status ON uploads(status)"
                )
                cursor.execute()
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_local_artifacts_reddit_url ON local_artifacts(reddit_url)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_local_artifacts_youtube_video_id ON local_artifacts(youtube_video_id)"
                )

                conn.commit()

            self.logger.info(f"Database initialized at {self.db_path}")

        except sqlite3.Error as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise

    def _migrate_database(self, cursor: sqlite3.Cursor):
        """Perform database migrations for schema updates"""
        try:
            # Get current table schema
            cursor.execute("PRAGMA table_info(uploads)")
            columns = {row[1] for row in cursor.fetchall()}

            # Add missing columns from older versions
            migrations = [
                (
                    "reddit_post_id",
                    "ALTER TABLE uploads ADD COLUMN reddit_post_id TEXT",
                ),
                (
                    "youtube_video_id",
                    "ALTER TABLE uploads ADD COLUMN youtube_video_id TEXT",
                ),
                (
                    "original_score",
                    "ALTER TABLE uploads ADD COLUMN original_score INTEGER",
                ),
                (
                    "processing_duration_seconds",
                    "ALTER TABLE uploads ADD COLUMN processing_duration_seconds REAL",
                ),
                (
                    "video_duration_seconds",
                    "ALTER TABLE uploads ADD COLUMN video_duration_seconds REAL",
                ),
                ("file_size_mb", "ALTER TABLE uploads ADD COLUMN file_size_mb REAL"),
                (
                    "status",
                    "ALTER TABLE uploads ADD COLUMN status TEXT DEFAULT 'completed'",
                ),
                ("error_message", "ALTER TABLE uploads ADD COLUMN error_message TEXT"),
                (
                    "thumbnail_uploaded",
                    "ALTER TABLE uploads ADD COLUMN thumbnail_uploaded BOOLEAN DEFAULT FALSE",
                ),
                (
                    "ai_analysis_used",
                    "ALTER TABLE uploads ADD COLUMN ai_analysis_used BOOLEAN DEFAULT FALSE",
                ),
                ("created_at", "ALTER TABLE uploads ADD COLUMN created_at DATETIME"),
                ("updated_at", "ALTER TABLE uploads ADD COLUMN updated_at DATETIME"),
                # A/B Testing columns
                (
                    "thumbnail_ctr_a",
                    "ALTER TABLE uploads ADD COLUMN thumbnail_ctr_a REAL",
                ),
                (
                    "thumbnail_ctr_b",
                    "ALTER TABLE uploads ADD COLUMN thumbnail_ctr_b REAL",
                ),
                (
                    "active_thumbnail",
                    "ALTER TABLE uploads ADD COLUMN active_thumbnail TEXT DEFAULT 'A'",
                ),
                (
                    "thumbnail_test_start_date",
                    "ALTER TABLE uploads ADD COLUMN thumbnail_test_start_date DATETIME",
                ),
                (
                    "thumbnail_test_complete",
                    "ALTER TABLE uploads ADD COLUMN thumbnail_test_complete BOOLEAN DEFAULT FALSE",
                ),
                (
                    "winning_thumbnail",
                    "ALTER TABLE uploads ADD COLUMN winning_thumbnail TEXT",
                ),
            ]

            for column_name, migration_sql in migrations:
                if column_name not in columns:
                    try:
                        cursor.execute(migration_sql)
                        self.logger.info(
                            f"Database migration: Added column {column_name}"
                        )
                    except sqlite3.Error as e:
                        self.logger.warning(f"Migration failed for {column_name}: {e}")

            # Local artifacts table migration support
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='local_artifacts'"
            )
            artifacts_table_exists = cursor.fetchone() is not None

            if not artifacts_table_exists:
                self._ensure_local_artifacts_table(cursor)
            else:
                cursor.execute("PRAGMA table_info(local_artifacts)")
                artifact_columns = {row[1] for row in cursor.fetchall()}
                artifact_migrations = [
                    (
                        "local_project_id",
                        "ALTER TABLE local_artifacts ADD COLUMN local_project_id TEXT",
                    ),
                    (
                        "local_video_path",
                        "ALTER TABLE local_artifacts ADD COLUMN local_video_path TEXT",
                    ),
                    (
                        "local_thumbnail_paths_json",
                        "ALTER TABLE local_artifacts ADD COLUMN local_thumbnail_paths_json TEXT",
                    ),
                    (
                        "youtube_video_id",
                        "ALTER TABLE local_artifacts ADD COLUMN youtube_video_id TEXT",
                    ),
                    (
                        "created_at",
                        "ALTER TABLE local_artifacts ADD COLUMN created_at DATETIME",
                    ),
                    (
                        "updated_at",
                        "ALTER TABLE local_artifacts ADD COLUMN updated_at DATETIME",
                    ),
                ]
                for column_name, migration_sql in artifact_migrations:
                    if column_name not in artifact_columns:
                        try:
                            cursor.execute(migration_sql)
                            self.logger.info(
                                f"Database migration: Added local_artifacts column {column_name}"
                            )
                        except sqlite3.Error as e:
                            self.logger.warning(
                                f"Migration failed for local_artifacts.{column_name}: {e}"
                            )

        except sqlite3.Error as e:
            self.logger.warning(f"Error during database migration: {e}")

    def _ensure_local_artifacts_table(self, cursor: sqlite3.Cursor) -> None:
        """Create local_artifacts with the canonical schema when missing."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS local_artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reddit_url TEXT NOT NULL UNIQUE,
                local_project_id TEXT,
                local_video_path TEXT,
                local_thumbnail_paths_json TEXT,
                youtube_video_id TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

    @contextmanager
    def get_connection(self):
        """Get database connection with proper error handling"""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            yield conn
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def is_video_processed(self, reddit_url: str) -> bool:
        """Check if a video has already been processed"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM uploads WHERE reddit_url = ? AND status != 'failed'",
                    (reddit_url,),
                )
                count = cursor.fetchone()[0]
                return count > 0

        except sqlite3.Error as e:
            self.logger.error(f"Error checking if video processed: {e}")
            return False

    def record_upload(
        self,
        reddit_url: str,
        reddit_post_id: str,
        title: str,
        subreddit: str,
        original_score: int,
        youtube_url: Optional[str] = None,
        youtube_video_id: Optional[str] = None,
        processing_duration: Optional[float] = None,
        video_duration: Optional[float] = None,
        file_size_mb: Optional[float] = None,
        thumbnail_uploaded: bool = False,
        ai_analysis_used: bool = False,
        status: str = "completed",
        error_message: Optional[str] = None,
    ) -> Optional[int]:
        """
        Record a video upload to the database

        Returns:
            Upload ID if successful, None if failed
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO uploads (
                        reddit_url, reddit_post_id, youtube_url, youtube_video_id,
                        title, subreddit, original_score,
                        processing_duration_seconds, video_duration_seconds, file_size_mb,
                        status, error_message, thumbnail_uploaded, ai_analysis_used,
                        upload_timestamp, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(reddit_url) DO UPDATE SET
                        reddit_post_id = excluded.reddit_post_id,
                        youtube_url = excluded.youtube_url,
                        youtube_video_id = excluded.youtube_video_id,
                        title = excluded.title,
                        subreddit = excluded.subreddit,
                        original_score = excluded.original_score,
                        processing_duration_seconds = excluded.processing_duration_seconds,
                        video_duration_seconds = excluded.video_duration_seconds,
                        file_size_mb = excluded.file_size_mb,
                        status = excluded.status,
                        error_message = excluded.error_message,
                        thumbnail_uploaded = excluded.thumbnail_uploaded,
                        ai_analysis_used = excluded.ai_analysis_used,
                        updated_at = excluded.updated_at
                """,
                    (
                        reddit_url,
                        reddit_post_id,
                        youtube_url,
                        youtube_video_id,
                        title,
                        subreddit,
                        original_score,
                        processing_duration,
                        video_duration,
                        file_size_mb,
                        status,
                        error_message,
                        thumbnail_uploaded,
                        ai_analysis_used,
                        datetime.now(),
                        datetime.now(),
                    ),
                )

                upload_id = cursor.lastrowid
                if not upload_id:
                    cursor.execute(
                        "SELECT id FROM uploads WHERE reddit_url = ?",
                        (reddit_url,),
                    )
                    row = cursor.fetchone()
                    upload_id = row[0] if row else None
                conn.commit()

                self.logger.info(f"Recorded upload: {title[:50]}... (ID: {upload_id})")
                return upload_id

        except sqlite3.Error as e:
            self.logger.error(f"Error recording upload: {e}")
            return None

    def update_upload_status(
        self,
        upload_id: int,
        status: str,
        error_message: Optional[str] = None,
        youtube_url: Optional[str] = None,
        youtube_video_id: Optional[str] = None,
    ):
        """Update the status of an upload"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    UPDATE uploads
                    SET status = ?,
                        updated_at = ?,
                        error_message = COALESCE(?, error_message),
                        youtube_url = COALESCE(?, youtube_url),
                        youtube_video_id = COALESCE(?, youtube_video_id)
                    WHERE id = ?
                    """,
                    (
                        status,
                        datetime.now(timezone.utc),
                        error_message,
                        youtube_url,
                        youtube_video_id,
                        upload_id,
                    ),
                )

                conn.commit()
                self.logger.info(f"Updated upload {upload_id} status to {status}")

        except sqlite3.Error as e:
            self.logger.error(f"Error updating upload status: {e}")

    def record_local_artifacts(
        self,
        reddit_url: str,
        local_project_id: Optional[str],
        local_video_path: Optional[str],
        local_thumbnail_paths: Optional[List[str]] = None,
        youtube_video_id: Optional[str] = None,
    ) -> Optional[int]:
        """Store/update local artifact mapping for a Reddit source."""
        try:
            local_thumbnail_paths_json = json.dumps(local_thumbnail_paths or [])

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO local_artifacts (
                        reddit_url, local_project_id, local_video_path,
                        local_thumbnail_paths_json, youtube_video_id,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT(reddit_url) DO UPDATE SET
                        local_project_id = excluded.local_project_id,
                        local_video_path = excluded.local_video_path,
                        local_thumbnail_paths_json = excluded.local_thumbnail_paths_json,
                        youtube_video_id = excluded.youtube_video_id,
                        updated_at = CURRENT_TIMESTAMP
                """,
                    (
                        reddit_url,
                        local_project_id,
                        local_video_path,
                        local_thumbnail_paths_json,
                        youtube_video_id,
                    ),
                )

                cursor.execute(
                    "SELECT id FROM local_artifacts WHERE reddit_url = ?", (reddit_url,)
                )
                row = cursor.fetchone()
                conn.commit()
                return row[0] if row else None

        except sqlite3.Error as e:
            self.logger.error(f"Error recording local artifacts: {e}")
            return None

    def get_local_artifacts_by_video_id(
        self, video_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get local artifact mapping by YouTube video ID."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM local_artifacts
                    WHERE youtube_video_id = ?
                    ORDER BY updated_at DESC
                    LIMIT 1
                """,
                    (video_id,),
                )
                row = cursor.fetchone()
                return self._normalize_local_artifacts_row(row)

        except sqlite3.Error as e:
            self.logger.error(f"Error getting local artifacts by video ID: {e}")
            return None

    def get_local_artifacts_by_reddit_url(
        self, reddit_url: str
    ) -> Optional[Dict[str, Any]]:
        """Get local artifact mapping by Reddit URL."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM local_artifacts
                    WHERE reddit_url = ?
                    ORDER BY updated_at DESC
                    LIMIT 1
                """,
                    (reddit_url,),
                )
                row = cursor.fetchone()
                return self._normalize_local_artifacts_row(row)

        except sqlite3.Error as e:
            self.logger.error(f"Error getting local artifacts by Reddit URL: {e}")
            return None

    def _normalize_local_artifacts_row(
        self, row: Optional[sqlite3.Row]
    ) -> Optional[Dict[str, Any]]:
        """Normalize local artifact row and parse JSON fields."""
        if not row:
            return None

        result = dict(row)
        raw_paths = result.get("local_thumbnail_paths_json")
        if raw_paths:
            try:
                parsed_paths = json.loads(raw_paths)
                result["local_thumbnail_paths"] = (
                    parsed_paths if isinstance(parsed_paths, list) else []
                )
            except (json.JSONDecodeError, TypeError):
                result["local_thumbnail_paths"] = []
        else:
            result["local_thumbnail_paths"] = []

        return result

    def get_upload_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent upload history"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM uploads 
                    ORDER BY upload_timestamp DESC 
                    LIMIT ?
                """,
                    (limit,),
                )

                rows = cursor.fetchall()
                return [dict(row) for row in rows]

        except sqlite3.Error as e:
            self.logger.error(f"Error getting upload history: {e}")
            return []

    def get_processing_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get processing statistics for the last N days"""
        try:
            days_int = max(0, int(days))
            days_modifier = f"-{days_int} days"
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Get basic stats - SQL injection vulnerability fixed by parameterizing days_modifier
                cursor.execute(
                    """
                    SELECT 
                        COUNT(*) as total_uploads,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_uploads,
                        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_uploads,
                        AVG(processing_duration_seconds) as avg_processing_time,
                        AVG(video_duration_seconds) as avg_video_duration,
                        SUM(file_size_mb) as total_file_size_mb
                    FROM uploads 
                    WHERE upload_timestamp >= datetime('now', ?)
                """,
                    (days_modifier,),
                )

                stats = dict(cursor.fetchone())

                # Calculate success rate
                if stats["total_uploads"] > 0:
                    stats["success_rate"] = (
                        stats["successful_uploads"] / stats["total_uploads"]
                    ) * 100
                else:
                    stats["success_rate"] = 0

                # Get subreddit breakdown
                cursor.execute(
                    """
                    SELECT subreddit, COUNT(*) as count
                    FROM uploads 
                    WHERE upload_timestamp >= datetime('now', ?)
                    GROUP BY subreddit
                    ORDER BY count DESC
                    LIMIT 10
                """,
                    (days_modifier,),
                )

                stats["top_subreddits"] = [dict(row) for row in cursor.fetchall()]

                return stats

        except sqlite3.Error as e:
            self.logger.error(f"Error getting processing stats: {e}")
            return {}

    def cleanup_old_records(self, days_to_keep: int = 90):
        """Clean up old records to prevent database bloat"""
        try:
            days_int = max(0, int(days_to_keep))
            with self.get_connection() as conn:
                cursor = conn.cursor()

                conn.commit()

        except sqlite3.Error as e:
            self.logger.error(f"Error cleaning up old records: {e}")

    def get_failed_uploads(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get failed uploads for retry analysis"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM uploads 
                    WHERE status = 'failed'
                    ORDER BY upload_timestamp DESC 
                    LIMIT ?
                """,
                    (limit,),
                )

                rows = cursor.fetchall()
                return [dict(row) for row in rows]

        except sqlite3.Error as e:
            self.logger.error(f"Error getting failed uploads: {e}")
            return []

    def export_data(self, output_path: Path, format: str = "json"):
        """Export database data for backup or analysis"""
        try:
            import json

            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Export uploads
                cursor.execute("SELECT * FROM uploads")
                uploads = [dict(row) for row in cursor.fetchall()]

                data = {
                    "uploads": uploads,
                    "export_timestamp": datetime.now().isoformat(),
                }

                output_path.parent.mkdir(parents=True, exist_ok=True)

                if format.lower() == "json":
                    with open(output_path, "w") as f:
                        json.dump(data, f, indent=2, default=str)
                else:
                    raise ValueError(f"Unsupported export format: {format}")

                self.logger.info(f"Database exported to {output_path}")

        except Exception as e:
            self.logger.error(f"Error exporting database: {e}")

    # A/B Thumbnail Testing Methods

    def start_thumbnail_ab_test(self, upload_id: int) -> bool:
        """Start A/B test for thumbnail by recording test start date"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE uploads
                    SET thumbnail_test_start_date = ?,
                        active_thumbnail = 'A',
                        updated_at = ?
                    WHERE id = ?
                """,
                    (datetime.now(), datetime.now(), upload_id),
                )

                conn.commit()
                self.logger.info(f"Started thumbnail A/B test for upload {upload_id}")
                return True

        except sqlite3.Error as e:
            self.logger.error(f"Error starting thumbnail A/B test: {e}")
            return False

    def update_thumbnail_ctr(self, upload_id: int, variant: str, ctr: float) -> bool:
        """
        Update CTR for a specific thumbnail variant

        Args:
            upload_id: ID of the upload
            variant: 'A' or 'B'
            ctr: Click-through rate as decimal (e.g., 0.05 for 5%)
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                if variant.upper() == "A":
                    cursor.execute(
                        """
                        UPDATE uploads
                        SET thumbnail_ctr_a = ?, updated_at = ?
                        WHERE id = ?
                    """,
                        (ctr, datetime.now(), upload_id),
                    )
                elif variant.upper() == "B":
                    cursor.execute(
                        """
                        UPDATE uploads
                        SET thumbnail_ctr_b = ?, updated_at = ?
                        WHERE id = ?
                    """,
                        (ctr, datetime.now(), upload_id),
                    )
                else:
                    raise ValueError(f"Invalid variant: {variant}. Must be 'A' or 'B'")

                conn.commit()
                self.logger.info(
                    f"Updated thumbnail CTR for upload {upload_id}, variant {variant}: {ctr:.4f}"
                )
                return True

        except (sqlite3.Error, ValueError) as e:
            self.logger.error(f"Error updating thumbnail CTR: {e}")
            return False

    def switch_active_thumbnail(self, upload_id: int, variant: str) -> bool:
        """
        Switch the active thumbnail variant

        Args:
            upload_id: ID of the upload
            variant: 'A' or 'B'
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE uploads
                    SET active_thumbnail = ?, updated_at = ?
                    WHERE id = ?
                """,
                    (variant.upper(), datetime.now(), upload_id),
                )

                conn.commit()
                self.logger.info(
                    f"Switched active thumbnail for upload {upload_id} to variant {variant}"
                )
                return True

        except sqlite3.Error as e:
            self.logger.error(f"Error switching active thumbnail: {e}")
            return False

    def complete_thumbnail_ab_test(self, upload_id: int, winning_variant: str) -> bool:
        """
        Complete the A/B test and record the winning variant

        Args:
            upload_id: ID of the upload
            winning_variant: 'A' or 'B'
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE uploads
                    SET thumbnail_test_complete = TRUE,
                        winning_thumbnail = ?,
                        active_thumbnail = ?,
                        updated_at = ?
                    WHERE id = ?
                """,
                    (
                        winning_variant.upper(),
                        winning_variant.upper(),
                        datetime.now(),
                        upload_id,
                    ),
                )

                conn.commit()
                self.logger.info(
                    f"Completed thumbnail A/B test for upload {upload_id}, winner: {winning_variant}"
                )
                return True

        except sqlite3.Error as e:
            self.logger.error(f"Error completing thumbnail A/B test: {e}")
            return False

    def get_pending_ab_tests(self, hours_since_start: int = 24) -> List[Dict[str, Any]]:
        """
        Get uploads that are ready for thumbnail A/B test evaluation

        Args:
            hours_since_start: Minimum hours since test started

        Returns:
            List of uploads ready for evaluation
        """
        try:
            hours_int = max(0, int(hours_since_start))
            hours_modifier = f"+{hours_int} hours"
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM uploads
                    WHERE thumbnail_test_start_date IS NOT NULL
                    AND thumbnail_test_complete = FALSE
                    AND datetime(thumbnail_test_start_date, ?) <= datetime('now')
                    ORDER BY thumbnail_test_start_date ASC
                """,
                    (hours_modifier,),
                )

                rows = cursor.fetchall()
                return [dict(row) for row in rows]

        except sqlite3.Error as e:
            self.logger.error(f"Error getting pending A/B tests: {e}")
            return []

    def get_ab_test_results(self, upload_id: int) -> Optional[Dict[str, Any]]:
        """Get A/B test results for a specific upload"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT thumbnail_ctr_a, thumbnail_ctr_b, active_thumbnail,
                           thumbnail_test_start_date, thumbnail_test_complete,
                           winning_thumbnail
                    FROM uploads WHERE id = ?
                """,
                    (upload_id,),
                )

                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None

        except sqlite3.Error as e:
            self.logger.error(f"Error getting A/B test results: {e}")
            return None

    def get_upload_by_video_id(self, youtube_video_id: str) -> Optional[Dict[str, Any]]:
        """Get latest upload row by YouTube video ID."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM uploads
                    WHERE youtube_video_id = ?
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (youtube_video_id,),
                )
                row = cursor.fetchone()
                return dict(row) if row else None

        except sqlite3.Error as e:
            self.logger.error(f"Error getting upload by video ID: {e}")
            return None

    def complete_thumbnail_ab_test_by_video_id(
        self, youtube_video_id: str, winning_variant: str
    ) -> bool:
        """Complete thumbnail test by video ID and persist winner variant."""
        try:
            normalized_variant = str(winning_variant).strip().upper()
            if normalized_variant not in {"A", "B"}:
                self.logger.warning(
                    "Invalid winning variant '%s' for video %s",
                    winning_variant,
                    youtube_video_id,
                )
                return False

            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE uploads
                    SET thumbnail_test_complete = TRUE,
                        winning_thumbnail = ?,
                        active_thumbnail = ?,
                        updated_at = ?
                    WHERE youtube_video_id = ?
                    """,
                    (
                        normalized_variant,
                        normalized_variant,
                        datetime.now(),
                        youtube_video_id,
                    ),
                )
                conn.commit()
                return cursor.rowcount > 0

        except sqlite3.Error as e:
            self.logger.error(f"Error completing thumbnail test by video ID: {e}")
            return False


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def init_db_manager(db_path: Optional[Path] = None) -> DatabaseManager:
    """Initialize the global database manager"""
    global _db_manager
    _db_manager = DatabaseManager(db_path)
    return _db_manager
