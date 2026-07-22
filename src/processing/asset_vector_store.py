"""
Local vector database for B-roll, background videos, and sound effects.
Uses ChromaDB with sentence-transformers for semantic asset retrieval.
Falls back to lightweight TF-IDF + cosine similarity when ChromaDB is unavailable.
"""

import logging
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from src.config.settings import get_config

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    EMBEDDER_AVAILABLE = True
except ImportError:
    EMBEDDER_AVAILABLE = False


logger = logging.getLogger(__name__)


class AssetVectorStore:
    """Vector database for local media assets (images, videos, audio)."""

    _METADATA_TYPES = frozenset(
        {"image", "video", "audio", "sound_effect", "background"}
    )

    def __init__(self, persist_dir: Optional[Path] = None):
        self.config = get_config()
        self._persist_dir = persist_dir or (
            self.config.paths.cache_folder / "asset_vectors"
        )
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._embedder: Optional[Any] = None
        self._collection: Optional[Any] = None
        self._id_to_path: Dict[str, str] = {}
        self._path_to_id: Dict[str, str] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection, or fallback."""
        if self._initialized:
            return

        if CHROMA_AVAILABLE:
            try:
                client = chromadb.PersistentClient(
                    path=str(self._persist_dir),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                self._collection = client.get_or_create_collection(
                    name="yotuubef_assets",
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info(
                    "AssetVectorStore: ChromaDB initialized at %s", self._persist_dir
                )
            except Exception as exc:
                logger.warning("ChromaDB init failed, using fallback: %s", exc)
                self._collection = None

        self._id_to_path = {}
        self._path_to_id = {}
        self._load_fallback_index()

        if self._collection is None:
            logger.info("AssetVectorStore: using fallback in-memory index")

        self._initialized = True

    def _load_fallback_index(self) -> None:
        """Load the fallback JSON index."""
        index_path = self._persist_dir / "fallback_index.json"
        if index_path.exists():
            try:
                data = json.loads(index_path.read_text(encoding="utf-8"))
                self._id_to_path = data.get("id_to_path", {})
                self._path_to_id = {v: k for k, v in self._id_to_path.items()}
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load fallback index: %s", exc)

    def _save_fallback_index(self) -> None:
        """Persist the fallback JSON index."""
        try:
            data = {"id_to_path": self._id_to_path}
            index_path = self._persist_dir / "fallback_index.json"
            index_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to save fallback index: %s", exc)

    def _get_embedder(self):
        """Lazy-load the sentence transformer embedder."""
        if self._embedder is None and EMBEDDER_AVAILABLE:
            try:
                self._embedder = SentenceTransformer(
                    "all-MiniLM-L6-v2", cache_folder=str(self._persist_dir / "models")
                )
            except Exception as exc:
                logger.warning("Failed to load embedder: %s", exc)
        return self._embedder

    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding vector for text."""
        embedder = self._get_embedder()
        if embedder:
            return embedder.encode(text, normalize_embeddings=True)
        return self._fallback_embedding(text)

    @staticmethod
    def _fallback_embedding(text: str) -> np.ndarray:
        """Lightweight TF-IDF-like embedding as fallback."""
        words = re.findall(r"[a-z]{2,}", text.lower())
        unique = sorted(set(words))
        vec = np.zeros(256, dtype=np.float32)
        for i, w in enumerate(unique[:256]):
            idx = hash(w) % 256
            count = words.count(w)
            vec[idx] += np.log1p(count)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def _make_asset_id(self, file_path: Path) -> str:
        """Generate a unique asset ID from file path."""
        path_str = str(file_path.resolve())
        return hashlib.md5(path_str.encode()).hexdigest()

    def _extract_tags_from_path(self, file_path: Path) -> List[str]:
        """Extract semantic tags from file path and parent dir names."""
        parts = list(file_path.parts)
        stem = file_path.stem
        tags = set()

        for part in parts:
            cleaned = re.sub(r"[-_]", " ", part.lower())
            cleaned = re.sub(r"[^a-z0-9\s]", "", cleaned)
            for token in cleaned.split():
                if len(token) > 2:
                    tags.add(token)

        stem_cleaned = re.sub(r"[-_]", " ", stem.lower())
        stem_cleaned = re.sub(r"[^a-z0-9\s]", "", stem_cleaned)
        for token in stem_cleaned.split():
            if len(token) > 2:
                tags.add(token)

        return sorted(tags)

    def _classify_asset_type(self, file_path: Path) -> str:
        """Classify a file by its directory or extension."""
        ext = file_path.suffix.lower()
        parent = file_path.parent.name.lower()

        if "background" in parent or "music" in parent:
            return "background"
        if (
            "sound" in parent
            or "effect" in parent
            or ext in {".mp3", ".wav", ".ogg", ".flac"}
        ):
            return "sound_effect"
        if ext in {".mp4", ".webm", ".mov", ".avi"}:
            return "video"
        return "image"

    async def scan_directory(self, directory: Path, recursive: bool = True) -> int:
        """Scan a directory and index all supported media files."""
        supported_exts = {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".gif",
            ".mp4",
            ".webm",
            ".mov",
            ".mp3",
            ".wav",
            ".ogg",
            ".flac",
        }
        count = 0

        pattern = "**/*" if recursive else "*"
        for file_path in Path(directory).glob(pattern):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in supported_exts:
                continue

            asset_id = self._make_asset_id(file_path)
            self._id_to_path[asset_id] = str(file_path.resolve())
            self._path_to_id[str(file_path.resolve())] = asset_id

            tags = self._extract_tags_from_path(file_path)
            asset_type = self._classify_asset_type(file_path)
            metadata = {
                "path": str(file_path.resolve()),
                "type": asset_type,
                "tags": ",".join(tags),
                "size_bytes": file_path.stat().st_size,
                "extension": file_path.suffix.lower(),
            }

            if self._collection is not None:
                embedding = self._compute_embedding(" ".join(tags)).tolist()
                try:
                    self._collection.add(
                        ids=[asset_id],
                        embeddings=[embedding],
                        metadatas=[metadata],
                        documents=[" ".join(tags)],
                    )
                except Exception as exc:
                    logger.debug("ChromaDB add failed for %s: %s", file_path, exc)

            count += 1

        self._save_fallback_index()
        logger.info("Indexed %d assets from %s", count, directory)
        return count

    async def search(
        self,
        query: str,
        asset_type: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for assets semantically similar to query."""
        results: List[Dict[str, Any]] = []

        if top_k <= 0:
            return results

        if self._collection is not None:
            try:
                query_embedding = self._compute_embedding(query).tolist()
                where = (
                    {"type": asset_type} if asset_type in self._METADATA_TYPES else None
                )
                raw = self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k * 2, 50),
                    where=where,
                )
                if raw and raw.get("ids") and raw["ids"][0]:
                    for idx, asset_id in enumerate(raw["ids"][0]):
                        meta = raw["metadatas"][0][idx] if raw.get("metadatas") else {}
                        distance = (
                            raw["distances"][0][idx] if raw.get("distances") else 0.0
                        )
                        results.append(
                            {
                                "id": asset_id,
                                "path": meta.get(
                                    "path", self._id_to_path.get(asset_id, "")
                                ),
                                "type": meta.get("type", "unknown"),
                                "tags": meta.get("tags", "").split(",")
                                if meta.get("tags")
                                else [],
                                "score": float(1.0 - distance),
                            }
                        )
            except Exception as exc:
                logger.warning("ChromaDB query failed, using fallback: %s", exc)

        if not results:
            results = self._search_fallback(query, asset_type=asset_type, top_k=top_k)

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def _search_fallback(
        self,
        query: str,
        asset_type: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Fallback search using cosine similarity on lightweight embeddings."""
        query_vec = self._fallback_embedding(query)
        results: List[Dict[str, Any]] = []
        scored: List[Tuple[float, str]] = []

        for asset_id, asset_path in self._id_to_path.items():
            path = Path(asset_path)
            tags = self._extract_tags_from_path(path)
            tag_text = " ".join(tags)
            asset_type_detected = self._classify_asset_type(path)

            if asset_type and asset_type_detected != asset_type:
                continue

            asset_vec = self._fallback_embedding(tag_text)
            sim = float(np.dot(query_vec, asset_vec))
            scored.append((sim, asset_id))

        scored.sort(key=lambda x: x[0], reverse=True)
        for sim, asset_id in scored[:top_k]:
            path = self._id_to_path.get(asset_id, "")
            p = Path(path) if path else None
            if p:
                results.append(
                    {
                        "id": asset_id,
                        "path": path,
                        "type": self._classify_asset_type(p),
                        "tags": self._extract_tags_from_path(p),
                        "score": sim,
                    }
                )

        return results

    async def scan_all_default_directories(self) -> Dict[str, int]:
        """Scan all default asset directories."""
        counts: Dict[str, int] = {}
        dirs = [
            ("images", self.config.paths.cache_folder / "broll_images"),
            ("backgrounds", self.config.paths.cache_folder / "backgrounds"),
            ("sound_effects", Path("sound_effects")),
            ("music", Path("music")),
        ]
        for name, directory in dirs:
            if directory.exists():
                count = await self.scan_directory(directory)
                counts[name] = count
        return counts

    async def get_similar_assets(
        self,
        query_or_path: str,
        asset_type: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find assets semantically similar to a given asset path or text query.

        Args:
            query_or_path: File path string to match similar assets against, or search query string
            asset_type: Optional filter for asset type (image, video, audio, sound_effect, background)
            top_k: Max number of results to return

        Returns:
            List of matching asset result dicts
        """
        resolved = Path(query_or_path)
        if resolved.exists() and resolved.is_file():
            tags = self._extract_tags_from_path(resolved)
            query = " ".join(tags)
            if asset_type is None:
                asset_type = self._classify_asset_type(resolved)
        else:
            query = query_or_path

        return await self.search(query, asset_type=asset_type, top_k=top_k)

    @property
    def asset_count(self) -> int:
        return len(self._id_to_path)
