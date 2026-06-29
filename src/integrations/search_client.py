from __future__ import annotations
import asyncio
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Optional

import aiohttp

if TYPE_CHECKING:
    from src.utils.search_audit_logger import SearchAuditLogger


class DeepResearchClient:
    """Exa-based deep research client via Hack Club AI proxy."""

    def __init__(self, audit_logger: Optional[SearchAuditLogger] = None):
        self.logger = logging.getLogger(__name__)
        self.api_key = os.getenv("HACKCLUB_SEARCH_API_KEY")
        self.base_url = "https://ai.hackclub.com/proxy/v1/exa/search"
        self.audit_logger = audit_logger
        self._session = None

    async def _ensure_session(self):
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def conduct_deep_research(self, query: str) -> str:
        if not self.api_key:
            return "No external research available."

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {"query": query, "numResults": 3}

        if self.audit_logger:
            self.audit_logger.log_request("POST", self.base_url, body, headers)

        max_retries = 3
        base_delay = 1.0
        last_error = None
        for attempt in range(max_retries):
            start = time.perf_counter()
            try:
                session = await self._ensure_session()
                async with session.post(
                    self.base_url,
                    headers=headers,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as response:
                    resp_body = await response.text()
                    duration_ms = (time.perf_counter() - start) * 1000
                    if self.audit_logger:
                        self.audit_logger.log_response(
                            response.status,
                            resp_body,
                            duration_ms,
                            url=self.base_url,
                        )
                    if response.status == 200:
                        try:
                            data = json.loads(resp_body)
                        except json.JSONDecodeError:
                            data = {}
                        return self._compile_report(data)
                    if 500 <= response.status < 600 or response.status == 429:
                        last_error = f"HTTP {response.status}"
                        delay = base_delay * (attempt + 1)
                        self.logger.warning(
                            "Exa search attempt %d/%d failed with status %s, retrying in %.1fs",
                            attempt + 1,
                            max_retries,
                            response.status,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    self.logger.warning(
                        "Exa search failed with status %s", response.status
                    )
                    return "Research failed."
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    delay = base_delay * (attempt + 1)
                    self.logger.warning(
                        "Exa search attempt %d/%d failed: %s, retrying in %.1fs",
                        attempt + 1,
                        max_retries,
                        e,
                        delay,
                    )
                    duration_ms = (time.perf_counter() - start) * 1000
                    if self.audit_logger:
                        self.audit_logger.log_response(
                            500, str(e), duration_ms, url=self.base_url
                        )
                    await asyncio.sleep(delay)
                    continue
                self.logger.error("Deep research error: %s", e)
                return "Research error."
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    delay = base_delay * (attempt + 1)
                    self.logger.warning(
                        "Exa search attempt %d/%d failed: %s, retrying in %.1fs",
                        attempt + 1,
                        max_retries,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                duration_ms = (time.perf_counter() - start) * 1000
                if self.audit_logger:
                    self.audit_logger.log_response(
                        500, str(e), duration_ms, url=self.base_url
                    )
                self.logger.error("Deep research error: %s", e)
                return "Research error."
        self.logger.warning(
            "Exa search failed after %d attempts: %s", max_retries, last_error
        )
        return "Research failed."

    def _compile_report(self, data: dict) -> str:
        compiled_lore = "EXTERNAL FACTS:\n"
        for result in data.get("results", []):
            if not isinstance(result, dict):
                continue
            title = str(result.get("title", "")).strip()
            url = str(result.get("url", "")).strip()
            text = result.get("text", "").strip()
            if text:
                source = f" Source: {url}" if url else ""
                heading = f"{title}: " if title else ""
                compiled_lore += f"- {heading}{text}{source}\n"
        return compiled_lore


class AgenticResearcher:
    """
    Multi-turn Agentic Research System.
    Uses deterministic query expansion and executes searches in parallel.
    """

    def __init__(self, audit_logger: Optional[SearchAuditLogger] = None):
        self.logger = logging.getLogger(__name__)
        self.api_key = os.getenv("BRAVE_SEARCH_API_KEY")
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.audit_logger = audit_logger
        self._session = None

        self.logger.info("AgenticResearcher initialized with deterministic planner")

    async def _ensure_session(self):
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _execute_search(self, query: str, session: aiohttp.ClientSession) -> str:
        """Execute a single Brave search and return results."""
        if not self.api_key:
            self.logger.warning("BRAVE_SEARCH_API_KEY not set")
            return ""

        headers = {"Accept": "application/json", "X-Subscription-Token": self.api_key}
        params = {"q": query, "count": 3}

        if self.audit_logger:
            self.audit_logger.log_request("GET", self.base_url, params, headers)

        start = time.perf_counter()
        try:
            async with session.get(
                self.base_url,
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as response:
                body = await response.text()
                duration_ms = (time.perf_counter() - start) * 1000
                if self.audit_logger:
                    self.audit_logger.log_response(
                        response.status,
                        body,
                        duration_ms,
                        url=self.base_url,
                    )
                if response.status == 200:
                    try:
                        data = json.loads(body)
                    except json.JSONDecodeError:
                        return ""
                    results = data.get("web", {}).get("results", [])
                    return " ".join([r.get("description", "") for r in results])
                self.logger.warning(
                    "Brave search failed with status %s", response.status
                )
                return ""
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            if self.audit_logger:
                self.audit_logger.log_response(
                    500, str(e), duration_ms, url=self.base_url
                )
            self.logger.error("Search error for '%s': %s", query, e)
            return ""

    async def deep_dive(
        self, topic: str, initial_context: str, max_turns: int = 2
    ) -> str:
        """
        Multi-turn investigative research loop.

        Args:
            topic: The main topic to research (e.g., "Action Park Cannonball Loop")
            initial_context: Initial context from Reddit post
            max_turns: Number of research iterations (default 2)

        Returns:
            Accumulated research knowledge from all turns
        """
        self.logger.info(f"Starting deep dive on: {topic}")

        accumulated_knowledge = f"REDDIT POST: {initial_context}\n\n"
        previous_queries = []

        session = await self._ensure_session()
        try:
            for turn in range(max_turns):
                self.logger.info(f"Research turn {turn + 1}/{max_turns}")

                queries = self._build_queries_for_turn(topic, turn, previous_queries)
                try:
                    if not queries:
                        self.logger.warning(
                            "No queries generated for turn %s", turn + 1
                        )
                        break

                    self.logger.info(f"Planned search queries: {queries}")
                    previous_queries.extend(queries)

                    # Execute all searches concurrently
                    search_tasks = [self._execute_search(q, session) for q in queries]
                    results = await asyncio.gather(*search_tasks)

                    # Accumulate knowledge for next turn or final output
                    for q, res in zip(queries, results):
                        accumulated_knowledge += f"\nSearch '{q}': {res}"

                except Exception as e:
                    self.logger.error(f"Research turn {turn} failed: {e}")
                    break
        finally:
            await self.close()

        self.logger.info(
            f"Deep dive complete. Final knowledge length: {len(accumulated_knowledge)} chars"
        )
        return accumulated_knowledge

    def _build_queries_for_turn(
        self,
        topic: str,
        turn: int,
        previous_queries: list,
    ) -> list:
        """Build deterministic search queries for each research turn."""
        normalized_topic = topic.strip()
        if turn == 0:
            return [
                normalized_topic,
                f"{normalized_topic} forum archive screenshot",  # Added to find UI/forum proof
                f"{normalized_topic} controversy timeline",
            ]

        variants = [
            f"{normalized_topic} primary sources evidence",
            f"{normalized_topic} original post screenshot",  # Forces image results of the actual post
            f"{normalized_topic} fact check",
        ]

        return [query for query in variants if query not in previous_queries]
