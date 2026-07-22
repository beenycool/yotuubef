#!/usr/bin/env python3
"""
Yotuubef Studio — Launcher & Google Colab Integration
Launches FastAPI backend + Vite frontend for local dev or Google Colab sessions.
"""

import os
import sys
import subprocess
import time
import signal


def run_colab_studio(port: int = 8420):
    """Run Yotuubef Studio inside a Google Colab notebook session."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    gui_dir = os.path.join(root_dir, "gui")

    # Build frontend if dist does not exist
    dist_dir = os.path.join(gui_dir, "dist")
    if not os.path.exists(dist_dir) or not os.path.exists(os.path.join(dist_dir, "index.html")):
        print("🔨 Building frontend assets for Colab...")
        subprocess.run(["npm", "run", "build"], cwd=gui_dir, check=True)

    print(f"🚀 Launching FastAPI backend + Web Studio server on port {port}...")
    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.server:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]
    proc = subprocess.Popen(backend_cmd, cwd=root_dir)

    time.sleep(2)

    try:
        from google.colab import output
        from IPython.display import display, HTML

        output.serve_kernel_port(port)
        url = output.eval_js(f'google.colab.kernel.proxyPort({port})')

        html_code = f"""
        <div style="
            background: linear-gradient(135deg, #12121a 0%, #1a1a26 100%);
            border: 2px solid #6c5ce7;
            border-radius: 12px;
            padding: 24px;
            margin: 16px 0;
            color: #f0f0f5;
            font-family: system-ui, sans-serif;
            box-shadow: 0 8px 24px rgba(108, 92, 231, 0.3);
        ">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                <span style="font-size: 28px;">🎬</span>
                <div>
                    <h2 style="margin: 0; font-size: 1.3rem; color: #fff;">Yotuubef Studio is Live!</h2>
                    <span style="font-size: 0.85rem; color: #a29bfe;">Google Colab Tunnel Active</span>
                </div>
            </div>

            <p style="margin: 8px 0 16px 0; color: #8c8ca1; font-size: 0.9rem;">
                Click the button below to open Yotuubef Studio in a new tab:
            </p>

            <a href="{url}" target="_blank" style="
                display: inline-block;
                background: linear-gradient(135deg, #6c5ce7 0%, #4ecdc4 100%);
                color: #ffffff;
                text-decoration: none;
                font-weight: 700;
                font-size: 1rem;
                padding: 14px 28px;
                border-radius: 8px;
                box-shadow: 0 4px 14px rgba(108, 92, 231, 0.5);
                transition: transform 0.2s;
            ">
                👉 OPEN YOTUUBEF STUDIO WORKSPACE 🚀
            </a>

            <div style="margin-top: 16px; font-size: 0.8rem; color: #5a5a70;">
                Direct URL: <a href="{url}" target="_blank" style="color: #4ecdc4;">{url}</a>
            </div>
        </div>
        """
        display(HTML(html_code))
        print(f"✨ Studio URL: {url}")
    except ImportError:
        print(f"✨ FastAPI running on http://localhost:{port} (serving GUI static app at /)")

    return proc


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    gui_dir = os.path.join(root_dir, "gui")

    print("=" * 60)
    print("🎬 Starting Yotuubef Studio...")
    print("=" * 60)

    # 1. Start FastAPI backend
    print("\n🚀 [1/2] Starting FastAPI backend on http://localhost:8420 ...")
    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.server:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8420",
    ]
    backend_proc = subprocess.Popen(backend_cmd, cwd=root_dir)

    time.sleep(1.5)

    # 2. Start Vite frontend dev server
    print("⚡ [2/2] Starting Vite frontend server on http://localhost:5173 ...")
    frontend_cmd = ["npm", "run", "dev"]
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=gui_dir)

    print("\n" + "=" * 60)
    print("✨ Yotuubef Studio is live!")
    print("👉 Open your browser at: http://localhost:5173")
    print("Press Ctrl+C to stop both servers.")
    print("=" * 60 + "\n")

    def signal_handler(sig, frame):
        print("\nStopping Yotuubef Studio...")
        backend_proc.terminate()
        frontend_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        backend_proc.wait()
    except KeyboardInterrupt:
        backend_proc.terminate()
        frontend_proc.terminate()


if __name__ == "__main__":
    main()
