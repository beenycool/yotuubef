#!/usr/bin/env python3
import os
import sys
import time
import json
import textwrap
from datetime import datetime
from typing import List, Dict, Any, Tuple, cast

import requests
from openai import APITimeoutError, NotFoundError, OpenAI

# -----------------------------
# Config via env vars
# -----------------------------
NVIDIA_API_KEY = os.getenv(
    "NVIDIA_API_KEY",
    "",
)
HACKCLUB_SEARCH_KEY = os.getenv(
    "HACKCLUB_SEARCH_KEY",
    "",
)

NVIDIA_BASE_URL = os.getenv(
    "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
).rstrip("/")
NVIDIA_CHAT_URL = f"{NVIDIA_BASE_URL}/chat/completions"

# Model IDs: default to NVIDIA NIM-style IDs.
MODEL_QWEN = os.getenv("MODEL_QWEN", "qwen/qwen3.5-397b-a17b")
MODEL_KIMI = os.getenv("MODEL_KIMI", "moonshotai/kimi-k2.5")

# Iteration / search tuning
ROUNDS = int(os.getenv("ROUNDS", "10"))  # minimum 10 requested
if ROUNDS < 10:
    ROUNDS = 10

SEARCH_COUNT = int(os.getenv("SEARCH_COUNT", "5"))  # results per query
MAX_QUERIES_PER_ROUND = int(os.getenv("MAX_QUERIES_PER_ROUND", "3"))

TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))
NVIDIA_CHAT_RETRIES = int(os.getenv("NVIDIA_CHAT_RETRIES", "3"))
SLEEP_BETWEEN_CALLS = float(os.getenv("SLEEP_BETWEEN_CALLS", "0.5"))
LOG_FILE = os.getenv("LOG_FILE", "run_log.txt")


def log_event(title: str, payload: Any | None = None) -> None:
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {title}\n")
        if payload is not None:
            if isinstance(payload, str):
                f.write(payload.rstrip() + "\n")
            else:
                f.write(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
                f.write("\n")
        f.write("\n")


# -----------------------------
# System prompt (the one you gave earlier)
# -----------------------------
SYSTEM_PROMPT = r"""
You are an investigative YouTube Shorts scriptwriter who specializes in “mini internet mysteries” and surprising trivia explainers.

GOAL
Create a 45–60 second faceless narration script that feels like a fast, research-backed story:
- Hook with a question or bold claim.
- Immediately misdirect with “you might think…” then flip it.
- Walk through 3–5 beats of evidence, each punchy and visual.
- End with a final reveal AND a “perfect loop” line that naturally connects back to the first line, making rewatching feel seamless.

TONE + STYLE
- Fast, confident, curious. Slightly dramatic, never cringe.
- Short sentences. Minimal filler. No rambling.
- Write like a YouTuber who actually did the homework.
- Use concrete details (names, dates, IDs, durations, counts) when available.
- Do not invent sources, quotes, or specific facts. If a detail can’t be verified from the user-provided context, either:
  (a) omit it, or
  (b) label it clearly as unconfirmed/likely/speculated and keep it brief.

CONTENT RULES (HARD)
- No porn/sexual content, self-harm, hate, explicit gore, or instructions for wrongdoing.
- Avoid medical/legal advice.
- Avoid defamatory claims about private individuals.
- If the topic is inherently sensitive (crime, tragedy, politics), keep it factual, non-sensational, and brand-safe.

OUTPUT FORMAT (STRICT JSON ONLY)
Return a single JSON object with these keys exactly:
{
  "title": string,
  "hook": string,
  "script_full": string,
  "loop_line": string,
  "segments": [
    {
      "time_seconds": number,
      "intended_duration_seconds": number,
      "narration": string,
      "on_screen_text": string,
      "b_roll_search_query": string,
      "sfx_suggestion": string,
      "pace": "fast" | "normal" | "slow",
      "emotion": "excited" | "calm" | "dramatic" | "neutral"
    }
  ],
  "sources_to_check": [string],
  "hashtags": [string]
}

SEGMENT RULES
- Total duration across segments must be 45–60 seconds.
- 6–10 segments max.
- Each narration field should be 1–2 spoken sentences.
- on_screen_text must be 2–6 words, big and punchy (caption-style).
- b_roll_search_query must be something a stock/image search could find.
- sfx_suggestion should be simple (whoosh, hit, pop, boom, glitch, click).
- The final segment must include the loop_line (verbatim) at the end of narration.

STRUCTURE BLUEPRINT (FOLLOW THIS)
1) 0–3s: Hook question / bold claim.
2) 3–8s: “You might think…” + common assumption.
3) 8–35s: Evidence beats (3–5 beats). Each beat includes one concrete detail + one visualizable moment.
4) 35–52s: Twist/reveal: the answer, the weird detail, or the “most people missed this” moment.
5) 52–60s: Loop ending: a line that re-frames the hook and tees up rewatch without saying “rewatch.”

QUALITY CHECK BEFORE YOU OUTPUT
- Is the hook attention-grabbing in the first sentence?
- Does every beat advance the mystery?
- Is the final line a clean loop back to the start?
- Are you accidentally guessing facts? If yes, remove or mark as unconfirmed.
- Is the output valid JSON with no extra text?
""".strip()


# -----------------------------
# Helpers
# -----------------------------
def die(msg: str, code: int = 2) -> None:
    log_event("FATAL", {"message": msg, "exit_code": code})
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def http_json(
    method: str,
    url: str,
    headers: Dict[str, str],
    params: Dict[str, Any] | None = None,
    body: Dict[str, Any] | None = None,
    timeout: int = TIMEOUT,
    retries: int = 4,
) -> Dict[str, Any]:
    last_err = None
    for attempt in range(retries):
        try:
            log_event(
                "HTTP_REQUEST",
                {
                    "attempt": attempt + 1,
                    "method": method.upper(),
                    "url": url,
                    "params": params,
                    "body": body,
                    "timeout_seconds": timeout,
                },
            )
            if method.upper() == "GET":
                r = requests.get(url, headers=headers, params=params, timeout=timeout)
            else:
                r = requests.post(url, headers=headers, json=body, timeout=timeout)
            log_event(
                "HTTP_RESPONSE",
                {
                    "attempt": attempt + 1,
                    "status_code": r.status_code,
                    "url": str(r.url),
                    "body_preview": r.text[:5000],
                },
            )
            if r.status_code >= 400:
                # retry on rate limit / transient
                if r.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"{r.status_code} {r.text[:500]}")
            return r.json()
        except Exception as e:
            log_event(
                "HTTP_ERROR",
                {
                    "attempt": attempt + 1,
                    "method": method.upper(),
                    "url": url,
                    "error": str(e),
                },
            )
            last_err = e
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            break
    raise RuntimeError(f"Request failed after retries: {last_err}")


def hackclub_web_search(query: str, count: int = SEARCH_COUNT) -> List[Dict[str, str]]:
    """
    Hack Club Search is a Brave Search API proxy.
    Docs show: /res/v1/web/search?q=...
    """
    url = "https://search.hackclub.com/res/v1/web/search"
    headers = {
        "Authorization": f"Bearer {HACKCLUB_SEARCH_KEY}",
        "Accept": "application/json",
        "User-Agent": "speedrun-scriptmaker/1.0",
    }
    params = {"q": query, "count": str(count)}
    data = http_json("GET", url, headers=headers, params=params)

    results = []
    web = data.get("web", {}) or {}
    for item in (web.get("results", []) or [])[:count]:
        results.append(
            {
                "title": (item.get("title") or "").strip(),
                "url": (item.get("url") or "").strip(),
                "description": (item.get("description") or "").strip(),
            }
        )
    return results


def nvidia_chat(
    model: str, messages: List[Dict[str, str]], temperature: float = 0.6
) -> str:
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY,
    )
    retries = max(1, NVIDIA_CHAT_RETRIES)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            log_event(
                "NVIDIA_CHAT_REQUEST",
                {
                    "attempt": attempt + 1,
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": 1800,
                    "timeout_seconds": 120,
                    "messages": messages,
                },
            )
            response = client.chat.completions.create(
                model=model,
                messages=cast(Any, messages),
                temperature=temperature,
                max_tokens=1800,
                timeout=120,
            )
            content = response.choices[0].message.content or ""
            log_event(
                "NVIDIA_CHAT_RESPONSE",
                {"attempt": attempt + 1, "model": model, "response": content},
            )
            return content
        except APITimeoutError as e:
            log_event(
                "NVIDIA_CHAT_TIMEOUT",
                {"attempt": attempt + 1, "model": model, "error": str(e)},
            )
            last_err = e
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            break
        except NotFoundError as e:
            log_event(
                "NVIDIA_CHAT_NOT_FOUND",
                {"attempt": attempt + 1, "model": model, "error": str(e)},
            )
            raise RuntimeError(
                f"OpenAI API call failed for model '{model}': {e}"
            ) from e
        except Exception as e:
            log_event(
                "NVIDIA_CHAT_ERROR",
                {"attempt": attempt + 1, "model": model, "error": str(e)},
            )
            raise RuntimeError(
                f"OpenAI API call failed for model '{model}': {e}"
            ) from e
    raise RuntimeError(
        f"OpenAI API call failed for model '{model}': {last_err}"
    ) from last_err


def chat_with_model_fallback(
    model: str,
    fallback_model: str,
    messages: List[Dict[str, str]],
    temperature: float,
) -> str:
    try:
        return nvidia_chat(model, messages, temperature=temperature)
    except RuntimeError as first_err:
        first_msg = str(first_err)
        should_fallback = ("404" in first_msg) or ("timed out" in first_msg.lower())
        if not should_fallback or not fallback_model or fallback_model == model:
            raise
        print(
            f"WARN: model '{model}' failed ({first_msg}). Falling back to '{fallback_model}'.",
            file=sys.stderr,
        )
        log_event(
            "MODEL_FALLBACK",
            {
                "from_model": model,
                "to_model": fallback_model,
                "reason": first_msg,
            },
        )
        return nvidia_chat(fallback_model, messages, temperature=temperature)


def safe_json_extract(text: str) -> Any | None:
    """
    If the model "accidentally" wraps JSON in text, try to recover.
    """
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            return None
    # crude extraction: find first { ... last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = text[start : end + 1]
        try:
            return json.loads(chunk)
        except Exception:
            return None
    return None


def clamp_queries(qs: List[str], max_n: int) -> List[str]:
    clean = []
    for q in qs:
        q = " ".join(q.split()).strip()
        if q and q.lower() not in [x.lower() for x in clean]:
            clean.append(q)
    return clean[:max_n]


# -----------------------------
# Main iterative loop
# -----------------------------
def main() -> None:
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(
            f"Run started at {datetime.utcnow().isoformat(timespec='seconds')}Z\n\n"
        )

    log_event(
        "RUN_CONFIG",
        {
            "rounds": ROUNDS,
            "search_count": SEARCH_COUNT,
            "max_queries_per_round": MAX_QUERIES_PER_ROUND,
            "http_timeout_seconds": TIMEOUT,
            "chat_retries": NVIDIA_CHAT_RETRIES,
            "model_qwen": MODEL_QWEN,
            "model_kimi": MODEL_KIMI,
            "log_file": LOG_FILE,
        },
    )

    if not NVIDIA_API_KEY:
        die(
            "Set NVIDIA_API_KEY (NVIDIA NIM / Build key, typically starts with nvapi-)."
        )
    if not HACKCLUB_SEARCH_KEY:
        die("Set HACKCLUB_SEARCH_KEY (Hack Club Search API key).")

    # Base instruction to keep it speedrunning-focused.
    user_goal = (
        "We are making a 45–60s investigative YouTube Shorts script in the SPEEDRUNNING niche. "
        "Find a real mini-mystery or surprising, verifiable speedrunning story. "
        "Use web search results I provide as your evidence pool. "
        "Each round: propose up to 3 specific search queries that would reduce uncertainty, "
        "and briefly explain what each query is trying to confirm. "
        "Do NOT write the final script until the last round."
    )

    # Conversation state
    notes: List[str] = []
    source_urls: List[str] = []
    chosen_angle: str | None = None

    # Prime messages (system prompt is your strict JSON script generator)
    base_messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_goal},
    ]

    for round_i in range(1, ROUNDS + 1):
        model = MODEL_QWEN if round_i % 2 == 1 else MODEL_KIMI

        round_header = f"ROUND {round_i}/{ROUNDS}"
        print(f"\n=== {round_header} | model={model} ===", file=sys.stderr)
        log_event("ROUND_START", {"round": round_i, "model": model})

        # Ask the model what to search next (or finalize on last round)
        if round_i < ROUNDS:
            prompt = {
                "role": "user",
                "content": textwrap.dedent(f"""
                Current notes (may be messy, that's life):
                {json.dumps(notes[-10:], indent=2)}

                Known source URLs so far:
                {json.dumps(source_urls[-20:], indent=2)}

                If we don't have a chosen angle yet, propose 2–3 candidate angles in speedrunning.
                Then output JSON with keys:
                {{
                  "chosen_angle": string,
                  "queries": [string, ...],
                  "why": [string, ...]
                }}
                Rules:
                - queries must be web-searchable
                - prefer primary sources (leaderboards, wikis, speedrun.com pages, event vods, official posts)
                - max {MAX_QUERIES_PER_ROUND} queries
                """).strip(),
            }
            messages = (
                base_messages
                + (
                    [{"role": "assistant", "content": "\n".join(notes[-6:])}]
                    if notes
                    else []
                )
                + [prompt]
            )
            backup_model = MODEL_KIMI if model == MODEL_QWEN else MODEL_QWEN
            raw = chat_with_model_fallback(
                model,
                backup_model,
                messages,
                temperature=0.5,
            )
            plan = safe_json_extract(raw)
            log_event("ROUND_PLAN_RAW", {"round": round_i, "model": model, "raw": raw})
            if not isinstance(plan, dict):
                # If model refuses to behave, force it back on track.
                plan = {
                    "chosen_angle": chosen_angle or "Unclear",
                    "queries": [
                        "speedrunning controversy removed run evidence",
                        "speedrun.com moderator removed run explanation",
                        "fastest time disqualified proof video",
                    ],
                    "why": ["fallback", "fallback", "fallback"],
                }

            chosen_angle = (plan.get("chosen_angle") or chosen_angle or "").strip()
            queries = clamp_queries(plan.get("queries") or [], MAX_QUERIES_PER_ROUND)
            log_event(
                "ROUND_PLAN_PARSED",
                {
                    "round": round_i,
                    "chosen_angle": chosen_angle,
                    "queries": queries,
                    "why": plan.get("why") or [],
                },
            )

            if not queries:
                queries = [
                    "speedrun.com world record disqualified evidence",
                    "speedrunning hidden technique discovered first",
                    "fastest speedrun controversy timeline",
                ][:MAX_QUERIES_PER_ROUND]

            # Perform searches
            round_results: List[Tuple[str, List[Dict[str, str]]]] = []
            for q in queries:
                try:
                    results = hackclub_web_search(q, count=SEARCH_COUNT)
                except Exception as e:
                    results = [
                        {"title": "SEARCH_ERROR", "url": "", "description": str(e)}
                    ]
                log_event(
                    "SEARCH_RESULTS",
                    {"round": round_i, "query": q, "results": results},
                )
                round_results.append((q, results))
                time.sleep(SLEEP_BETWEEN_CALLS)

            # Summarize results back into notes
            snippet_block = []
            for q, results in round_results:
                snippet_block.append(f"QUERY: {q}")
                for r in results:
                    if r.get("url"):
                        source_urls.append(r["url"])
                    snippet_block.append(
                        f"- {r.get('title', '').strip()} | {r.get('url', '').strip()} | {r.get('description', '').strip()}"
                    )
                snippet_block.append("")

            notes.append(
                f"[{round_header}] angle={chosen_angle}\n" + "\n".join(snippet_block)
            )
            log_event(
                "ROUND_NOTES_APPENDED",
                {
                    "round": round_i,
                    "chosen_angle": chosen_angle,
                    "note_preview": notes[-1][:4000],
                },
            )

        else:
            # Final round: generate the actual script JSON (STRICT)
            final_prompt = {
                "role": "user",
                "content": textwrap.dedent(f"""
                FINAL ROUND. Produce the finished script JSON now.
                Topic/angle: {chosen_angle or "Pick the strongest verified speedrunning mini-mystery from notes."}

                Evidence notes:
                {json.dumps(notes, indent=2)}

                Source URLs (dedupe mentally, but include the best in sources_to_check):
                {json.dumps(source_urls[-60:], indent=2)}

                Requirements:
                - MUST output STRICT JSON only, no extra commentary.
                - Must be speedrunning niche.
                - Do not invent facts. If uncertain, label as unconfirmed and keep it minimal.
                - 45–60 seconds total across segments; 6–10 segments.
                """).strip(),
            }
            messages = base_messages + [final_prompt]
            backup_model = MODEL_KIMI if model == MODEL_QWEN else MODEL_QWEN
            raw = chat_with_model_fallback(
                model,
                backup_model,
                messages,
                temperature=0.6,
            )
            log_event("FINAL_RAW", {"round": round_i, "model": model, "raw": raw})

            # Print only the final JSON to stdout
            extracted = safe_json_extract(raw)
            if extracted is None:
                # If model output isn't valid JSON, still output raw as last resort
                log_event("FINAL_OUTPUT", {"valid_json": False, "output": raw.strip()})
                print(raw.strip())
            else:
                log_event("FINAL_OUTPUT", {"valid_json": True, "output": extracted})
                print(json.dumps(extracted, indent=2, ensure_ascii=False))

    # done


if __name__ == "__main__":
    main()
