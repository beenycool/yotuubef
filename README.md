# yotuubef — Hybrid Documentary YouTube Shorts Generator

Automated pipeline that converts Reddit posts into documentary-style YouTube Shorts with AI-powered research, script generation, TTS narration, and video rendering.

## Architecture

Reddit → Research (Exa/Brave) → AI Script (NVIDIA NIM) → TTS → Render (MoviePy) → Upload (YouTube API)

## Pipeline Phases

1. **IDEA_GENERATION** — Analyze Reddit post, generate story angles
2. **WAIT_FOR_GEMINI_REPORT** — Pause for manual deep research
3. **SYNTHESIS** — Choose angle, plan media queries
4. **EVIDENCE_GATHERING** — Search for primary sources
5. **SCRIPTING** — Generate segment-by-segment narration script
6. **VIDEO_RENDER** — TTS + background video + captions + effects

## Quick Start

1. Copy `secrets.example` to `.env` and fill in API keys
2. Add background videos to `data/backgrounds/`
3. Run: `python -m src.enhanced_orchestrator --project "my_documentary" --reddit-url "https://reddit.com/r/..."`

## Configuration

See `config.yaml` for all tunable parameters. Environment variables override YAML.

## Testing

```
pytest tests/ -v --tb=short
```
