import sys
import types
import importlib
import logging
from src.database.db_manager import DatabaseManager
from pathlib import Path

db_manager = DatabaseManager(db_path=Path("artifacts.db"))

sys.modules.pop("src.management.channel_manager", None)
channel_manager_module = importlib.import_module("src.management.channel_manager")
ChannelManager = channel_manager_module.ChannelManager

manager = ChannelManager.__new__(ChannelManager)
manager.logger = logging.getLogger(__name__)

# Try mocking _get_db_manager instead of _db_manager
manager._get_db_manager = lambda: db_manager

resolved_path = manager._get_video_path("yt_video_123")
print("Resolved path:", resolved_path)
