import os, mimetypes, uuid, threading, time
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from model_service import improve_music_or_audio, train_model
from utils import makedirs

# --------------------------------------------------------------------
# Directory Setup
# --------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
UPLOAD_DIR = BASE / "uploads"
OUTPUT_DIR = UPLOAD_DIR / "outputs"
STATIC_DIR = BASE / "static"
makedirs(UPLOAD_DIR)
makedirs(OUTPUT_DIR)

# --------------------------------------------------------------------
# FastAPI App Setup
# --------------------------------------------------------------------
app = FastAPI(title="AI Music Improver API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ✅ Serve static frontend assets
assets_dir = STATIC_DIR / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

# --------------------------------------------------------------------
# In-memory job store
# --------------------------------------------------------------------
jobs = {}

# --------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home():
    """Serve the frontend index.html"""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"), status_code=200)
    return HTMLResponse("<h1>Frontend not built — please run docker build again.</h1>", status_code=500)


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker"""
    return JSONResponse({"status": "ok", "timestamp": time.time()})


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Upload a music/audio file"""
    ext = Path(file.filename).suffix.lower()
    if ext not in {".mid", ".midi", ".wav", ".mp3"}:
        raise HTTPException(status_code=400, detail="Invalid file type")
    save_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    save_path = UPLOAD_DIR / save_name
    with open(save_path, "wb") as f:
        f.write(await file.read())
    return {"filename": save_name}


def _run_job(job_id: str, in_path: Path, out_path: Path, mode: str):
    """Background thread to improve audio"""
    jobs[job_id]["status"] = "running"
    try:
        improve_music_or_audio(str(in_path), str(out_path), mode)
        jobs[job_id]["status"] = "done"
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


@app.post("/api/submit")
def submit_job(payload: dict):
    """Queue a music improvement job"""
    fname = payload.get("filename")
    mode = payload.get("mode", "auto")
    if not fname:
        raise HTTPException(status_code=400, detail="filename required")

    in_path = UPLOAD_DIR / fname
    if not in_path.exists():
        raise HTTPException(status_code=404, detail="file not found")

    out_name = f"improved_{fname}"
    out_path = OUTPUT_DIR / out_name
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "queued", "input": str(in_path), "output": str(out_path), "error": None}

    thread = threading.Thread(target=_run_job, args=(job_id, in_path, out_path, mode), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.post("/api/train")
def train_job(background_tasks: BackgroundTasks, payload: dict):
    """Start a background model training job"""
    mode = payload.get("mode", "music")
    data_dir = payload.get("data_dir", str(UPLOAD_DIR))
    job_id = uuid.uuid4().hex[:12]
    out_path = OUTPUT_DIR / f"training_output_{job_id}.txt"

    jobs[job_id] = {"status": "queued", "input": data_dir, "output": str(out_path), "error": None}

    def _train():
        jobs[job_id]["status"] = "running"
        try:
            train_model(data_dir=data_dir, mode=mode)
            with open(out_path, "w") as f:
                f.write(f"Training complete at {time.ctime()}\n")
            jobs[job_id]["status"] = "done"
        except Exception as e:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)

    background_tasks.add_task(_train)
    return {"job_id": job_id, "message": "Training started"}


@app.get("/api/status/{job_id}")
def job_status(job_id: str):
    """Return job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="job not found")
    return {"job_id": job_id, "status": jobs[job_id]["status"], "output": jobs[job_id]["output"], "error": jobs[job_id].get("error")}


@app.get("/api/download/{name}")
def download(name: str):
    """Download output files"""
    path = OUTPUT_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="output not found")
    return FileResponse(path, media_type="application/octet-stream", filename=name)


@app.get("/api/list")
def list_outputs():
    """List generated output files"""
    files = [p.name for p in OUTPUT_DIR.iterdir() if p.suffix.lower() in {".mid", ".midi", ".wav", ".mp3"}]
    return {"outputs": files}

