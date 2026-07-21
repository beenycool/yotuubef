"""
Gradio GUI for the yotuubef documentary pipeline.

Designed to run inside Google Colab with `launch(share=True)`.
The notebook mounts Google Drive for persistence so projects
and results survive across sessions.

Tabs:
  1. Dashboard   - live pipeline status, project cards, log stream
  2. New Run     - trigger a new documentary run
  3. Script Editor - review / edit the final script before render
  4. History     - browse past runs and results
  5. Settings    - edit config.yaml values
  6. Results     - preview rendered videos and thumbnails
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO)

from gui.runner import (
    PHASE_LABELS,
    PHASE_ORDER,
    RUNNER,
    get_monitor_status,
    get_project_files,
    get_project_state,
    get_project_video,
    get_result_files,
    get_upload_history,
    list_projects,
    load_config_yaml,
    load_script,
    save_config_yaml,
    save_script,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _phase_summary(state: Dict[str, Any]) -> str:
    phase = state.get("current_phase", "—")
    label = PHASE_LABELS.get(phase, phase)
    status = state.get("status", "—")
    updated = state.get("updated_at", "")[:19].replace("T", " ")
    return f"**{label}** · status: `{status}` · updated {updated}"


def _project_choices() -> List[Tuple[str, str]]:
    return [(p["name"], p["name"]) for p in list_projects()]


def _phase_choices() -> List[Tuple[str, str]]:
    return [(PHASE_LABELS.get(p, p), p) for p in PHASE_ORDER]


# ── Dashboard callbacks ────────────────────────────────────────────────


def dashboard_refresh() -> Tuple[str, str, str]:
    """Return (status_md, projects_md, logs)."""
    status = get_monitor_status()
    running = status["running"]
    done = status["done"]
    active = status["active"]

    badge = "\U0001F534 Running" if running else ("\u2705 Done" if done else "\u23F8 Idle")
    status_md = f"### Pipeline Status\n\n{badge}  ·  last refresh {status['timestamp']}\n"
    if active:
        status_md += "\n**Active project:** " + active["name"] + "\n"
        status_md += _phase_summary(
            get_project_state(active["name"]) or {"current_phase": active["phase"]}
        )

    projects = status["projects"]
    if not projects:
        projects_md = "_No projects yet. Use the **New Run** tab to start one._"
    else:
        rows = ["| Project | Phase | Status | Reddit URL |", "|---|---|---|---|"]
        for p in projects:
            url = p.get("reddit_url", "")
            url_short = url[:40] + "…" if len(url) > 40 else url
            rows.append(
                f"| **{p['name']}** | {p['phase']} | {p['status']} | {url_short} |"
            )
        projects_md = "\n".join(rows)

    return status_md, projects_md, status["logs"]


def new_run_start(
    project_name: str,
    reddit_url: str,
    resume: bool,
    phase_override: str,
    no_upload: bool,
    no_auto_research: bool,
    gemini_report_path: str,
):
    if not project_name.strip():
        yield "Please enter a project name.", ""
        return
    if RUNNER.is_running:
        yield "A pipeline is already running. Wait for it to finish or stop it first.", RUNNER.get_logs()
        return
    RUNNER.start(
        project_name=project_name.strip(),
        reddit_url=reddit_url,
        resume=resume,
        phase_override=phase_override if phase_override else "",
        no_upload=no_upload,
        no_auto_research=no_auto_research,
        gemini_report_path=gemini_report_path,
    )
    yield f"Started pipeline for **{project_name}** — streaming logs below…", RUNNER.get_logs()
    while not RUNNER.done:
        yield f"Running… (project: {project_name})", RUNNER.get_logs()
    final = "✅ Pipeline finished." if not RUNNER.get_logs().endswith("1]\n") else "❌ Pipeline failed — see logs."
    yield final, RUNNER.get_logs()


def new_run_stop():
    RUNNER.stop()
    return "Stopped.", RUNNER.get_logs()


# ── Script Editor callbacks ───────────────────────────────────────────


def script_load(project: str):
    if not project:
        return None, None, None, None, "", "Pick a project first."
    data = load_script(project)
    if data is None:
        return None, None, None, None, "", f"No `final_script.json` found for **{project}**. Run the pipeline through SCRIPTING first."
    segments = data.get("segments", [])
    seg_texts = [seg.get("narration", "") for seg in segments]
    seg_cues = [seg.get("expression_cue", "") for seg in segments]
    seg_durations = [seg.get("intended_duration_seconds", 0) for seg in segments]
    return (
        data.get("title", ""),
        data.get("hook", ""),
        data.get("loop_bridge", ""),
        json.dumps(segments, indent=2, ensure_ascii=True),
        "\n---\n".join(seg_texts),
        f"Loaded script for **{project}** ({len(segments)} segments).",
    )


def script_save(
    project: str,
    title: str,
    hook: str,
    loop_bridge: str,
    segments_json: str,
):
    if not project:
        return "Pick a project first."
    data = load_script(project) or {}
    data["title"] = title
    data["hook"] = hook
    data["loop_bridge"] = loop_bridge
    try:
        segs = json.loads(segments_json) if segments_json.strip() else data.get("segments", [])
    except json.JSONDecodeError as exc:
        return f"Invalid segment JSON: {exc}"
    data["segments"] = segs
    return save_script(project, data)


# ── Results callbacks ──────────────────────────────────────────


def results_load_video(project: str):
    if not project:
        return None, "Pick a project."
    video = get_project_video(project)
    if video:
        return video, f"Video found at `{video}`"
    return None, f"No rendered video found for **{project}**."


def results_load_files(project: str):
    if not project:
        return "Pick a project.", ""
    files = get_project_files(project)
    lines = [f"### Files for {project}\n"]
    for folder, items in files.items():
        if not items:
            continue
        lines.append(f"**{folder}** ({len(items)})")
        for it in items:
            lines.append(f"- `{it['name']}` — {it['size_kb']} KB — {it['modified']}")
        lines.append("")
    return "\n".join(lines), ""


# ── History callbacks ──────────────────────────────────────────


def history_refresh():
    results = get_result_files()
    uploads = get_upload_history(limit=50)

    if not results:
        results_md = "_No result files found in `data/results/`._"
    else:
        lines = [
            "| Timestamp | Project | Phase | Success | Paused | Final Video | Error |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in results:
            video = r["video"] or "—"
            if len(video) > 50:
                video = "…" + video[-49:]
            lines.append(
                f"| {r['timestamp']} | {r['project']} | {r['phase']} | "
                f"{'yes' if r['success'] else 'no'} | "
                f"{'yes' if r['paused'] else ''} | `{video}` | {r['error'] or ''} |"
            )
        results_md = "\n".join(lines)

    if not uploads:
        db_md = "_Database empty or missing._"
    else:
        lines = [
            "| Date | Title | Subreddit | YouTube URL | Status | Duration |",
            "|---|---|---|---|---|---|",
        ]
        for u in uploads:
            ts = str(u.get("upload_timestamp", ""))[:19]
            title = (u.get("title") or "")[:50]
            url = u.get("youtube_url") or "—"
            dur = u.get("video_duration_seconds")
            dur_s = f"{dur:.0f}s" if dur else "—"
            lines.append(
                f"| {ts} | {title} | {u.get('subreddit', '')} | {url} | "
                f"{u.get('status', '')} | {dur_s} |"
            )
        db_md = "\n".join(lines)

    return results_md, db_md


# ── Settings callbacks ────────────────────────────────────────────────


def settings_load():
    cfg = load_config_yaml()

    api = cfg.get("api", {}) or {}
    tts = cfg.get("tts", {}) or {}
    content = cfg.get("content", {}) or {}
    video_processing = cfg.get("video_processing", {}) or {}
    audio = cfg.get("audio", {}) or {}

    cur_subreddits = content.get("curated_subreddits", []) or []
    cur_hard = content.get("hard_disallowed", []) or []

    return (
        api.get("nvidia_nim_model", ""),
        api.get("nvidia_nim_alt_model", ""),
        api.get("ai_provider", "nvidia_nim"),
        tts.get("primary_service", "elevenlabs"),
        "\n".join(cur_subreddits),
        "\n".join(cur_hard),
        video_processing.get("video_codec", "libx264"),
        video_processing.get("target_fps", 30),
        str(video_processing.get("default_output_resolution", [1080, 1920])),
        (audio.get("background_music", {}) or {}).get("volume", 0.4),
        json.dumps(cfg, indent=2, ensure_ascii=True),
    )


def settings_save(
    nvidia_model: str,
    nvidia_alt_model: str,
    ai_provider: str,
    tts_service: str,
    subreddits_text: str,
    hard_disallowed_text: str,
    video_codec: str,
    target_fps: int,
    resolution: str,
    bg_music_volume: float,
    raw_json: str,
):
    try:
        data = json.loads(raw_json) if raw_json.strip() else {}
    except json.JSONDecodeError as exc:
        return f"Invalid raw JSON: {exc}"
    api = data.setdefault("api", {})
    api["nvidia_nim_model"] = nvidia_model
    api["nvidia_nim_alt_model"] = nvidia_alt_model
    api["ai_provider"] = ai_provider
    tts = data.setdefault("tts", {})
    tts["primary_service"] = tts_service
    content = data.setdefault("content", {})
    content["curated_subreddits"] = [s.strip() for s in subreddits_text.splitlines() if s.strip()]
    content["hard_disallowed"] = [s.strip() for s in hard_disallowed_text.splitlines() if s.strip()]
    vp = data.setdefault("video_processing", {})
    vp["video_codec"] = video_codec
    vp["target_fps"] = target_fps
    try:
        res = json.loads(resolution) if isinstance(resolution, str) else resolution
    except Exception:
        res = [1080, 1920]
    vp["default_output_resolution"] = res
    audio_cfg = data.setdefault("audio", {})
    bg = audio_cfg.setdefault("background_music", {})
    bg["volume"] = bg_music_volume
    return save_config_yaml(data)


# ── Build the UI ─────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Yotuubef Studio") as app:
        gr.Markdown("# Yotuubef Studio\n\nDocumentary pipeline control panel")

        timer = gr.Timer(3)

        with gr.Tabs():
            # ── Dashboard tab ─────────────────────────────
            with gr.Tab("\U0001F4CA Dashboard"):
                status_md = gr.Markdown("Loading…")
                projects_md = gr.Markdown("Loading projects…")
                with gr.Accordion("Live Logs", open=False):
                    logs_box = gr.Code(
                        label="Pipeline Output",
                        language="shell",
                        lines=20,
                    )
                timer.tick(dashboard_refresh, outputs=[status_md, projects_md, logs_box])

            # ── New Run tab ────────────────────────────────
            with gr.Tab("\U0001F680 New Run"):
                gr.Markdown("### Start a new documentary project")
                with gr.Row():
                    with gr.Column(scale=2):
                        nr_project = gr.Textbox(
                            label="Project name",
                            placeholder="my_documentary_v1",
                        )
                        nr_reddit_url = gr.Textbox(
                            label="Reddit URL (optional)",
                            placeholder="https://reddit.com/r/...",
                        )
                        nr_gemini_report = gr.Textbox(
                            label="Gemini report path (optional)",
                            placeholder="/content/drive/MyDrive/report.txt",
                        )
                    with gr.Column(scale=1):
                        nr_resume = gr.Checkbox(label="Resume existing run")
                        nr_no_upload = gr.Checkbox(label="Skip YouTube upload", value=True)
                        nr_no_auto_research = gr.Checkbox(
                            label="Skip auto research (manual Gemini report)"
                        )
                        nr_phase = gr.Dropdown(
                            label="Phase override",
                            choices=[("", "")] + _phase_choices(),
                            value="",
                        )
                with gr.Row():
                    nr_start_btn = gr.Button("\U0001F680 Start Pipeline", variant="primary")
                    nr_stop_btn = gr.Button("\u26D4 Stop", variant="stop")
                nr_status = gr.Markdown("")
                nr_logs = gr.Code(
                    label="Live pipeline output",
                    language="shell",
                    lines=25,
                )
                nr_start_btn.click(
                    new_run_start,
                    inputs=[
                        nr_project,
                        nr_reddit_url,
                        nr_resume,
                        nr_phase,
                        nr_no_upload,
                        nr_no_auto_research,
                        nr_gemini_report,
                    ],
                    outputs=[nr_status, nr_logs],
                )
                nr_stop_btn.click(new_run_stop, outputs=[nr_status, nr_logs])

            # ── Script Editor tab ──────────────────────────
            with gr.Tab("\U0001F4DD Script Editor"):
                gr.Markdown(
                    "### Review and edit the generated script before rendering.\n\n"
                    "The script is at `findings/<project>/research/scripts/final_script.json`. "
                    "After editing, click Save, then switch to **New Run** and run with `--resume --phase VIDEO_RENDER`."
                )
                with gr.Row():
                    se_project = gr.Dropdown(
                        label="Project",
                        choices=_project_choices(),
                        interactive=True,
                    )
                    se_refresh_btn = gr.Button("\U0001F501 Refresh list")
                    se_load_btn = gr.Button("\U0001F4C2 Load script", variant="primary")
                se_title = gr.Textbox(label="Title")
                se_hook = gr.Textbox(label="Hook", lines=2)
                se_loop_bridge = gr.Textbox(label="Loop bridge", lines=2)
                se_segments = gr.Code(
                    label="Segments (JSON)",
                    language="json",
                    lines=20,
                )
                se_save_btn = gr.Button("\U0001F4BE Save script", variant="primary")
                se_msg = gr.Markdown("")
                se_refresh_btn.click(
                    lambda: gr.Dropdown(choices=_project_choices()),
                    outputs=se_project,
                )
                se_load_btn.click(
                    script_load,
                    inputs=se_project,
                    outputs=[se_title, se_hook, se_loop_bridge, se_segments, se_msg, se_msg],
                )
                se_save_btn.click(
                    script_save,
                    inputs=[se_project, se_title, se_hook, se_loop_bridge, se_segments],
                    outputs=se_msg,
                )

            # ── History tab ────────────────────────────────
            with gr.Tab("\U0001F4DC History"):
                gr.Markdown("### Run history")
                hist_refresh_btn = gr.Button("\U0001F501 Refresh")
                gr.Markdown("#### Result files (`data/results/`)")
                hist_results = gr.Markdown("_Click Refresh to load._")
                gr.Markdown("#### Upload history (SQLite DB)")
                hist_db = gr.Markdown("_No database rows yet._")
                hist_refresh_btn.click(
                    history_refresh,
                    outputs=[hist_results, hist_db],
                )
                timer.tick(history_refresh, outputs=[hist_results, hist_db])

            # ── Settings tab ────────────────────────────────
            with gr.Tab("\u2699 Settings"):
                gr.Markdown("### Edit `config.yaml`\n\nField changes are written back to `config.yaml`. Secrets come from environment/Colab secrets — don't put them here.")
                with gr.Row():
                    with gr.Column(scale=1):
                        st_model = gr.Textbox(label="NVIDIA NIM model")
                        st_alt_model = gr.Textbox(label="NVIDIA NIM alt model")
                        st_provider = gr.Textbox(label="AI provider")
                        st_tts = gr.Textbox(label="TTS primary service")
                        st_codec = gr.Textbox(label="Video codec")
                        st_fps = gr.Number(label="Target FPS", value=30)
                        st_res = gr.Textbox(label="Resolution [w,h]", value="[1080, 1920]")
                        st_bg_volume = gr.Slider(0, 1, value=0.4, step=0.05, label="Background music volume")
                    with gr.Column(scale=1):
                        st_subreddits = gr.Code(
                            label="Curated subreddits (one per line)",
                            language="shell",
                        )
                        st_hard = gr.Code(
                            label="Hard disallowed words (one per line)",
                            language="shell",
                        )
                gr.Markdown("#### Raw config (advanced)")
                st_raw = gr.Code(label="config.yaml (JSON preview)", language="json", lines=15)
                with gr.Row():
                    st_load_btn = gr.Button("\U0001F501 Load current config")
                    st_save_btn = gr.Button("\U0001F4BE Save config", variant="primary")
                st_msg = gr.Markdown("")
                st_load_btn.click(
                    settings_load,
                    outputs=[
                        st_model,
                        st_alt_model,
                        st_provider,
                        st_tts,
                        st_subreddits,
                        st_hard,
                        st_codec,
                        st_fps,
                        st_res,
                        st_bg_volume,
                        st_raw,
                    ],
                )
                st_save_btn.click(
                    settings_save,
                    inputs=[
                        st_model,
                        st_alt_model,
                        st_provider,
                        st_tts,
                        st_subreddits,
                        st_hard,
                        st_codec,
                        st_fps,
                        st_res,
                        st_bg_volume,
                        st_raw,
                    ],
                    outputs=st_msg,
                )
                app.load(settings_load, outputs=[
                    st_model, st_alt_model, st_provider, st_tts,
                    st_subreddits, st_hard, st_codec, st_fps, st_res,
                    st_bg_volume, st_raw,
                ])

            # ── Results tab ────────────────────────────────
            with gr.Tab("\U0001F3AC Results"):
                gr.Markdown("### Preview rendered videos and project artefacts")
                with gr.Row():
                    re_project = gr.Dropdown(
                        label="Project",
                        choices=_project_choices(),
                        interactive=True,
                    )
                    re_refresh_btn = gr.Button("\U0001F501 Refresh list")
                    re_load_btn = gr.Button("\U0001F4C2 Load", variant="primary")
                with gr.Row():
                    with gr.Column(scale=2):
                        re_video = gr.Video(label="Rendered video")
                    with gr.Column(scale=1):
                        re_files = gr.Markdown("Files will appear here.")
                re_msg = gr.Markdown("")
                re_refresh_btn.click(
                    lambda: gr.Dropdown(choices=_project_choices()),
                    outputs=re_project,
                )
                re_load_btn.click(
                    lambda p: results_load_video(p) + (results_load_files(p)[0],),
                    inputs=re_project,
                    outputs=[re_video, re_msg, re_files],
                )

        # Refresh dropdown choices on load
        app.load(lambda: gr.Dropdown(choices=_project_choices()), outputs=se_project)
        app.load(lambda: gr.Dropdown(choices=_project_choices()), outputs=re_project)

    return app


def launch(
    share: bool = True,
    server_port: int = 7860,
    debug: bool = False,
    **kwargs,
):
    """Build and launch the app. Convenience entry point for cells."""
    app = build_app()
    return app.queue().launch(
        share=share,
        server_port=server_port,
        debug=debug,
        show_error=True,
        theme=kwargs.pop("theme", gr.themes.Soft()),
        css=kwargs.pop(
            "css",
            ".phase-pill { padding: 0.25em 0.5em; border-radius: 0.35em; font-size: 0.85em; }",
        ),
        **kwargs,
    )


if __name__ == "__main__":
    launch()
