from __future__ import annotations
import asyncio
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import aiohttp

try:
    from openai import AsyncOpenAI

    OPENAI_ASYNC_AVAILABLE = True
except ImportError:
    OPENAI_ASYNC_AVAILABLE = False

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
                self.logger.warning("Exa search failed with status %s", response.status)
                return "Research failed."
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            if self.audit_logger:
                self.audit_logger.log_response(
                    500, str(e), duration_ms, url=self.base_url
                )
            self.logger.error("Deep research error: %s", e)
            return "Research error."

    def _compile_report(self, data: dict) -> str:
        compiled_lore = "EXTERNAL FACTS:\n"
        for result in data.get("results", []):
            text = result.get("text", "").strip()
            if text:
                compiled_lore += f"- {text}\n"
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


class AgenticExaResearcher:
    """
    Autonomous deep research agent using Exa search + LLM evaluation loop.

    Runs an investigative loop:
      search -> LLM evaluates -> LLM decides next queries -> repeat
    until all claims are verified or max_iterations reached.
    Discards junk results, keeps verified facts with exact quotes + URLs.
    """

    def __init__(
        self,
        search_client: Any,
        max_iterations: int = 4,
        audit_logger: Optional[SearchAuditLogger] = None,
    ):
        self.search_client = search_client
        self.max_iterations = max_iterations
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(__name__)

        self.api_key = os.getenv("NVIDIA_NIM_API_KEY") or os.getenv(
            "NVIDIA_API_KEY", ""
        )
        self.base_url = os.getenv(
            "NVIDIA_BASE_URL",
            "https://integrate.api.nvidia.com/v1",
        )
        self.summary_model = os.getenv(
            "NVIDIA_SUMMARY_MODEL",
            "qwen/qwen2.5-7b-instruct",
        )

        self.client = None
        if OPENAI_ASYNC_AVAILABLE and self.api_key:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )

        # Accumulated verified facts across all iterations
        self.accumulated_knowledge: List[Dict[str, str]] = []
        self.previous_queries: set = set()

    async def conduct_deep_research(
        self,
        topic: str,
        initial_claims: List[str],
    ) -> Dict[str, Any]:
        """
        Run the full agentic research loop.

        Args:
            topic: The documentary angle / chosen_angle (e.g. "Action Park Cannonball Loop hoax")
            initial_claims: List of evidence_questions from synthesis (e.g. ["Was the loop real?", ...])

        Returns:
            dict with:
              - verified_evidence: list of {"claim", "quote", "url"}
              - total_results_evaluated: int
        """
        if not self.client:
            self.logger.warning(
                "AsyncOpenAI client not available; cannot run agentic research"
            )
            return {"verified_evidence": [], "total_results_evaluated": 0}

        current_queries = await self._generate_initial_queries(topic, initial_claims)
        all_raw_results: List[Dict[str, str]] = []

        for iteration in range(self.max_iterations):
            if not current_queries:
                self.logger.info("No more queries to run. Research complete.")
                break

            self.logger.info(
                "Research iteration %d/%d. Queries: %s",
                iteration + 1,
                self.max_iterations,
                current_queries,
            )

            # 1. Execute searches in parallel
            search_tasks = [
                self.search_client.search_web(q, count=3) for q in current_queries
            ]
            search_results = await asyncio.gather(*search_tasks)

            # Mark queries as done
            self.previous_queries.update(current_queries)

            # 2. Flatten raw results
            raw_results: List[Dict[str, str]] = []
            for result_list in search_results:
                for r in result_list:
                    raw_results.append(
                        {
                            "url": r.url,
                            "title": r.title,
                            "text": getattr(r, "description", "") or "",
                        }
                    )
            all_raw_results.extend(raw_results)

            # 3. Evaluate with LLM
            evaluation = await self._evaluate_results(
                topic,
                initial_claims,
                raw_results,
            )

            # 4. Save verified facts
            verified = evaluation.get("verified_facts", [])
            if verified:
                self.accumulated_knowledge.extend(verified)
                self.logger.info("Verified %d new facts", len(verified))

            # 5. Check if research is complete
            if evaluation.get("research_complete", False):
                self.logger.info("LLM indicates research is complete. Stopping search.")
                break

            # 6. Get next queries, filter out already-run ones
            next_queries = evaluation.get("next_queries", [])
            current_queries = [
                q for q in next_queries if q not in self.previous_queries
            ]

        return {
            "verified_evidence": self.accumulated_knowledge,
            "total_results_evaluated": len(all_raw_results),
        }

    async def _generate_initial_queries(
        self,
        topic: str,
        claims: List[str],
    ) -> List[str]:
        """Generate the first batch of search queries based on topic and claims."""
        if not claims:
            return [topic]

        prompt = (
            "You are an internet investigator. We are making a documentary video about:\n"
            f"{topic}\n\n"
            "We need to verify these claims:\n"
            f"{json.dumps(claims)}\n\n"
            "Generate 3 highly specific search queries to find primary sources "
            "or evidence for these claims. Focus on archived pages, forum posts, "
            "official statements, and primary sources.\n"
            'Return JSON with a "queries" key containing a list of strings.'
        )

        try:
            resp = await self.client.chat.completions.create(
                model=self.summary_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            result = json.loads(resp.choices[0].message.content)
            queries = result.get("queries", [])
            if queries:
                return queries[:5]
        except Exception as e:
            self.logger.error("Failed to generate initial queries: %s", e)

        # Fallback: use the claims themselves as queries
        return claims[:3] if claims else [topic]

    async def _evaluate_results(
        self,
        topic: str,
        claims: List[str],
        raw_results: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        LLM evaluates search results: keeps good facts, discards junk, plans next queries.
        """
        system_prompt = (
            "You are a strict Documentary Fact-Checker and Research Director.\n"
            "You will be given raw search results. You must:\n"
            "1. Evaluate if the search results actually contain proof for our claims.\n"
            "2. If a result proves a claim, extract the EXACT QUOTE and URL "
            "and add it to 'verified_facts'.\n"
            "3. If a result is useless, spam, or doesn't help, ignore it.\n"
            "4. Based on what you found (or didn't find), decide what to search "
            "for NEXT to find the missing pieces.\n"
            "5. If all claims are verified, set 'research_complete' to true.\n\n"
            "Return JSON with this schema:\n"
            "{\n"
            '    "verified_facts": [{"claim": "...", "quote": "...", "url": "..."}],\n'
            '    "next_queries": ["query 1", "query 2"],\n'
            '    "research_complete": false\n'
            "}"
        )

        user_prompt = (
            f"TOPIC: {topic}\n"
            f"CLAIMS TO VERIFY: {json.dumps(claims)}\n\n"
            "RAW SEARCH RESULTS:\n"
            f"{json.dumps(raw_results, indent=2)}"
        )

        try:
            resp = await self.client.chat.completions.create(
                model=self.summary_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            self.logger.error("Failed to evaluate search results: %s", e)
            return {
                "verified_facts": [],
                "next_queries": [],
                "research_complete": False,
            }
