"""
Database management for tracking uploaded videos and processing history.
Handles SQLite operations with proper error handling and migrations.
Uses aiosqlite for async-safe database access.
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import aiosqlite

from src.config.settings import get_config


class DatabaseManager:
    """Manages the SQLite database for video tracking (async)."""

    def __init__(self, db_path: Optional[Path] = None):
        self.config = get_config()
        self.db_path = db_path or self.config.paths.db_file
        self.logger = logging.getLogger(__name__)
        self._conn: Optional[aiosqlite.Connection] = None

    _conn_lock: asyncio.Lock = asyncio.Lock()

    async def _ensure_connection(self) -> aiosqlite.Connection:
        async with self._conn_lock:
            if self._conn is None:
                self._conn = await aiosqlite.connect(str(self.db_path), timeout=30.0)
                self._conn.row_factory = aiosqlite.Row
                await self._conn.execute("PRAGMA journal_mode=WAL")
            return self._conn

    @asynccontextmanager
    async def get_connection(self):
        """Get database connection using persistent connection."""
        conn = await self._ensure_connection()
        try:
            yield conn
        except aiosqlite.Error as e:
            await conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise

    async def initialize_database(self):
        """Initialize database with required tables."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = await self._ensure_connection()

            await conn.execute("""
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

            await self._ensure_local_artifacts_table(conn)
            await self._ensure_processing_history_table(conn)

            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_uploads_reddit_url ON uploads(reddit_url)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_uploads_youtube_video_id ON uploads(youtube_video_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_uploads_subreddit ON uploads(subreddit)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_uploads_status ON uploads(status)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_local_artifacts_reddit_url ON local_artifacts(reddit_url)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_local_artifacts_youtube_video_id ON local_artifacts(youtube_video_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_processing_history_created_at ON processing_history(created_at)"
            )

            await conn.commit()
            self.logger.info(f"Database initialized at {self.db_path}")
        except aiosqlite.Error as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise

    @staticmethod
    async def _ensure_local_artifacts_table(conn: aiosqlite.Connection) -> None:
        await conn.execute("""
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

    @staticmethod
    async def _ensure_processing_history_table(conn: aiosqlite.Connection) -> None:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS processing_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reddit_url TEXT,
                processing_status TEXT,
                message TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

    async def is_video_processed(self, reddit_url: str) -> bool:
        """Check if a video has already been processed."""
        try:
            async with self.get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT COUNT(*) FROM uploads WHERE reddit_url = ? AND status != 'failed'",
                    (reddit_url,),
                )
                row = await cursor.fetchone()
                return row[0] > 0 if row else False
        except aiosqlite.Error as e:
            self.logger.error(f"Error checking if video processed: {e}")
            return False

    async def record_upload(
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
        """Record a video upload to the database."""
        try:
            async with self.get_connection() as conn:
                await conn.execute(
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
                        datetime.now(timezone.utc),
                        datetime.now(timezone.utc),
                    ),
                )
                await conn.commit()

                cursor = await conn.execute(
                    "SELECT id FROM uploads WHERE reddit_url = ?", (reddit_url,)
                )
                row = await cursor.fetchone()
                upload_id = row[0] if row else None

                self.logger.info(f"Recorded upload: {title[:50]}... (ID: {upload_id})")
                return upload_id
        except aiosqlite.Error as e:
            self.logger.error(f"Error recording upload: {e}")
            return None

    async def update_upload_status(
        self,
        upload_id: int,
        status: str,
        error_message: Optional[str] = None,
        youtube_url: Optional[str] = None,
        youtube_video_id: Optional[str] = None,
    ):
        """Update the status of an upload."""
        try:
            async with self.get_connection() as conn:
                await conn.execute(
                    """
                    UPDATE uploads
                    SET status = ?, updated_at = ?,
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
                await conn.commit()
                self.logger.info(f"Updated upload {upload_id} status to {status}")
        except aiosqlite.Error as e:
            self.logger.error(f"Error updating upload status: {e}")

    async def record_local_artifacts(
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
            async with self.get_connection() as conn:
                await conn.execute(
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
                cursor = await conn.execute(
                    "SELECT id FROM local_artifacts WHERE reddit_url = ?", (reddit_url,)
                )
                row = await cursor.fetchone()
                await conn.commit()
                return row[0] if row else None
        except aiosqlite.Error as e:
            self.logger.error(f"Error recording local artifacts: {e}")
            return None

    async def get_local_artifacts_by_video_id(
        self, video_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get local artifact mapping by YouTube video ID."""
        try:
            async with self.get_connection() as conn:
                cursor = await conn.execute(
                    """
                    SELECT * FROM local_artifacts
                    WHERE youtube_video_id = ? ORDER BY updated_at DESC LIMIT 1
                    """,
                    (video_id,),
                )
                row = await cursor.fetchone()
                return self._normalize_local_artifacts_row(row)
        except aiosqlite.Error as e:
            self.logger.error(f"Error getting local artifacts by video ID: {e}")
            return None

    async def get_local_artifacts_by_reddit_url(
        self, reddit_url: str
    ) -> Optional[Dict[str, Any]]:
        """Get local artifact mapping by Reddit URL."""
        try:
            async with self.get_connection() as conn:
                cursor = await conn.execute(
                    """
                    SELECT * FROM local_artifacts
                    WHERE reddit_url = ? ORDER BY updated_at DESC LIMIT 1
                    """,
                    (reddit_url,),
                )
                row = await cursor.fetchone()
                return self._normalize_local_artifacts_row(row)
        except aiosqlite.Error as e:
            self.logger.error(f"Error getting local artifacts by Reddit URL: {e}")
            return None

    @staticmethod
    def _normalize_local_artifacts_row(
        row: Optional[aiosqlite.Row],
    ) -> Optional[Dict[str, Any]]:
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

    async def get_upload_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent upload history."""
        try:
            async with self.get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT * FROM uploads ORDER BY upload_timestamp DESC LIMIT ?",
                    (limit,),
                )
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        except aiosqlite.Error as e:
            self.logger.error(f"Error getting upload history: {e}")
            return []

    async def get_failed_uploads(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get failed uploads for retry analysis."""
        try:
            async with self.get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT * FROM uploads WHERE status = 'failed' ORDER BY upload_timestamp DESC LIMIT ?",
                    (limit,),
                )
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        except aiosqlite.Error as e:
            self.logger.error(f"Error getting failed uploads: {e}")
            return []

    async def export_data(self, output_path: Path, fmt: str = "json"):
        """Export database data for backup or analysis."""
        try:
            async with self.get_connection() as conn:
                cursor = await conn.execute("SELECT * FROM uploads")
                rows = await cursor.fetchall()
                uploads = [dict(row) for row in rows]

                data = {
                    "uploads": uploads,
                    "export_timestamp": datetime.now(timezone.utc).isoformat(),
                }
                output_path.parent.mkdir(parents=True, exist_ok=True)

                if fmt.lower() == "json":
                    output_path.write_text(
                        json.dumps(data, indent=2, default=str), encoding="utf-8"
                    )
                else:
                    raise ValueError(f"Unsupported export format: {fmt}")

                self.logger.info(f"Database exported to {output_path}")
        except Exception as e:
            self.logger.error(f"Error exporting database: {e}")

    # ── A/B Thumbnail Testing ──────────────────────────────────────────

    async def start_thumbnail_ab_test(self, upload_id: int) -> bool:
        """Start A/B test for thumbnail."""
        try:
            async with self.get_connection() as conn:
                await conn.execute(
                    """
                    UPDATE uploads
                    SET thumbnail_test_start_date = ?, active_thumbnail = 'A', updated_at = ?
                    WHERE id = ?
                    """,
                    (datetime.now(timezone.utc), datetime.now(timezone.utc), upload_id),
                )
                await conn.commit()
                self.logger.info(f"Started thumbnail A/B test for upload {upload_id}")
                return True
        except aiosqlite.Error as e:
            self.logger.error(f"Error starting thumbnail A/B test: {e}")
            return False

    async def update_thumbnail_ctr(
        self, upload_id: int, variant: str, ctr: float
    ) -> bool:
        """Update CTR for a specific thumbnail variant."""
        try:
            variant = variant.upper()
            if variant not in ("A", "B"):
                raise ValueError(f"Invalid variant: {variant}. Must be 'A' or 'B'")
            column = "thumbnail_ctr_a" if variant == "A" else "thumbnail_ctr_b"
            async with self.get_connection() as conn:
                await conn.execute(
                    f"UPDATE uploads SET {column} = ?, updated_at = ? WHERE id = ?",
                    (ctr, datetime.now(timezone.utc), upload_id),
                )
                await conn.commit()
                self.logger.info(
                    f"Updated thumbnail CTR for upload {upload_id}, variant {variant}: {ctr:.4f}"
                )
                return True
        except (aiosqlite.Error, ValueError) as e:
            self.logger.error(f"Error updating thumbnail CTR: {e}")
            return False

    async def switch_active_thumbnail(self, upload_id: int, variant: str) -> bool:
        """Switch the active thumbnail variant."""
        try:
            async with self.get_connection() as conn:
                await conn.execute(
                    "UPDATE uploads SET active_thumbnail = ?, updated_at = ? WHERE id = ?",
                    (variant.upper(), datetime.now(timezone.utc), upload_id),
                )
                await conn.commit()
                self.logger.info(
                    f"Switched active thumbnail for upload {upload_id} to variant {variant}"
                )
                return True
        except aiosqlite.Error as e:
            self.logger.error(f"Error switching active thumbnail: {e}")
            return False

    async def complete_thumbnail_ab_test(
        self, upload_id: int, winning_variant: str
    ) -> bool:
        """Complete the A/B test and record the winning variant."""
        try:
            async with self.get_connection() as conn:
                await conn.execute(
                    """
                    UPDATE uploads
                    SET thumbnail_test_complete = TRUE, winning_thumbnail = ?,
                        active_thumbnail = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        winning_variant.upper(),
                        winning_variant.upper(),
                        datetime.now(timezone.utc),
                        upload_id,
                    ),
                )
                await conn.commit()
                self.logger.info(
                    f"Completed thumbnail A/B test for upload {upload_id}, winner: {winning_variant}"
                )
                return True
        except Exception as e:
            self.logger.error(f"Error completing thumbnail A/B test: {e}")
            return False

    async def get_pending_ab_tests(
        self, hours_since_start: int = 24
    ) -> List[Dict[str, Any]]:
        """Get uploads ready for thumbnail A/B test evaluation."""
        try:
            hours_int = max(0, int(hours_since_start))
            async with self.get_connection() as conn:
                cursor = await conn.execute(
                    """
                    SELECT * FROM uploads
                    WHERE thumbnail_test_start_date IS NOT NULL
                    AND thumbnail_test_complete = FALSE
                    AND datetime(thumbnail_test_start_date, ?) <= datetime('now')
                    ORDER BY thumbnail_test_start_date ASC
                    """,
                    (f"+{hours_int} hours",),
                )
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        except aiosqlite.Error as e:
            self.logger.error(f"Error getting pending A/B tests: {e}")
            return []

    async def get_ab_test_results(self, upload_id: int) -> Optional[Dict[str, Any]]:
        """Get A/B test results for a specific upload."""
        try:
            async with self.get_connection() as conn:
                cursor = await conn.execute(
                    """
                    SELECT thumbnail_ctr_a, thumbnail_ctr_b, active_thumbnail,
                           thumbnail_test_start_date, thumbnail_test_complete, winning_thumbnail
                    FROM uploads WHERE id = ?
                    """,
                    (upload_id,),
                )
                row = await cursor.fetchone()
                return dict(row) if row else None
        except aiosqlite.Error as e:
            self.logger.error(f"Error getting A/B test results: {e}")
            return None

    async def get_upload_by_video_id(
        self, youtube_video_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get latest upload row by YouTube video ID."""
        try:
            async with self.get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT * FROM uploads WHERE youtube_video_id = ? ORDER BY updated_at DESC LIMIT 1",
                    (youtube_video_id,),
                )
                row = await cursor.fetchone()
                return dict(row) if row else None
        except aiosqlite.Error as e:
            self.logger.error(f"Error getting upload by video ID: {e}")
            return None

    async def complete_thumbnail_ab_test_by_video_id(
        self, youtube_video_id: str, winning_variant: str
    ) -> bool:
        """Complete thumbnail test by video ID and persist winner variant."""
        try:
            normalized_variant = winning_variant.strip().upper()
            if normalized_variant not in {"A", "B"}:
                self.logger.warning(
                    "Invalid winning variant '%s' for video %s",
                    winning_variant,
                    youtube_video_id,
                )
                return False
            async with self.get_connection() as conn:
                cursor = await conn.execute(
                    """
                    UPDATE uploads
                    SET thumbnail_test_complete = TRUE, winning_thumbnail = ?,
                        active_thumbnail = ?, updated_at = ?
                    WHERE youtube_video_id = ?
                    """,
                    (
                        normalized_variant,
                        normalized_variant,
                        datetime.now(timezone.utc),
                        youtube_video_id,
                    ),
                )
                await conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            self.logger.error(f"Error completing thumbnail test by video ID: {e}")
            return False

    async def close(self) -> None:
        """Close the persistent connection if open."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def get_engagement_metrics(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get video metrics for trend analysis. Used by EnhancementOptimizer."""
        try:
            async with self.get_connection() as conn:
                cursor = await conn.execute(
                    """
                    SELECT enhancements_used, engagement_rate, retention_rate,
                           completion_rate, click_through_rate
                    FROM video_metrics
                    WHERE upload_date >= datetime('now', ?)
                    """,
                    (f"-{days} days",),
                )
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        except aiosqlite.Error:
            return []

    async def get_avg_metrics(self, days: int = 30) -> Dict[str, float]:
        """Get average metrics across recent videos."""
        try:
            async with self.get_connection() as conn:
                cursor = await conn.execute(
                    """
                    SELECT AVG(engagement_rate), AVG(retention_rate),
                           AVG(completion_rate), AVG(click_through_rate)
                    FROM video_metrics
                    WHERE upload_date >= datetime('now', ?)
                    """,
                    (f"-{days} days",),
                )
                row = await cursor.fetchone()
                if row and row[0] is not None:
                    return {
                        "avg_engagement_rate": row[0] or 0.0,
                        "avg_retention_rate": row[1] or 0.0,
                        "avg_completion_rate": row[2] or 0.0,
                        "avg_ctr": row[3] or 0.0,
                    }
        except aiosqlite.Error:
            pass
        return {}


_db_manager: Optional[DatabaseManager] = None
_init_lock: asyncio.Lock = asyncio.Lock()


async def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    async with _init_lock:
        if _db_manager is None:
            _db_manager = DatabaseManager()
            await _db_manager.initialize_database()
    return _db_manager


async def init_db_manager(db_path: Optional[Path] = None) -> DatabaseManager:
    """Initialize the global database manager."""
    global _db_manager
    async with _init_lock:
        _db_manager = DatabaseManager(db_path)
        await _db_manager.initialize_database()
    return _db_manager
