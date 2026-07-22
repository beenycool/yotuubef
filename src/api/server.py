"""FastAPI application entrypoint for Yotuubef Studio."""

from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routes import projects, pipeline, scripts, artifacts, config, videos
from src.api.ws import ws_manager

app = FastAPI(
    title="Yotuubef Studio API",
    description="Backend API for Yotuubef Hybrid Documentary Video Generator",
    version="1.0.0",
)

# CORS middleware for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["pipeline"])
app.include_router(scripts.router, prefix="/api/scripts", tags=["scripts"])
app.include_router(artifacts.router, prefix="/api/artifacts", tags=["artifacts"])
app.include_router(config.router, prefix="/api/config", tags=["config"])
app.include_router(videos.router, prefix="/api/videos", tags=["videos"])

# Static file mounts
findings_dir = Path("findings")
findings_dir.mkdir(exist_ok=True)
app.mount("/media", StaticFiles(directory=str(findings_dir)), name="media")

processed_dir = Path("processed")
processed_dir.mkdir(exist_ok=True)
app.mount("/processed", StaticFiles(directory=str(processed_dir)), name="processed")


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "app": "Yotuubef Studio API", "version": "1.0.0"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Global WebSocket log/event endpoint."""
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


@app.websocket("/ws/{project}")
async def project_websocket_endpoint(websocket: WebSocket, project: str):
    """Project-specific WebSocket log endpoint."""
    await ws_manager.connect(websocket, project_name=project)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, project_name=project)


# Mount frontend dist static site (for Google Colab, production, single-port serve)
gui_dist = Path("gui/dist")
if gui_dist.exists() and (gui_dist / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(gui_dist), html=True), name="gui")

