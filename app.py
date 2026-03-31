import os
import json
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi import HTTPException
import traceback

from job_recommendation import JobRecommenderEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
STATE_PATH = os.path.join(BASE_DIR, "data", "user_state.json")

app = FastAPI(title="JobNexus API")

# If you later host frontend separately, keep CORS on.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = JobRecommenderEngine()


def load_state():
    if not os.path.exists(STATE_PATH):
        return {"likes": [], "history": []}
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# -------- Serve Frontend --------
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def home():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/admin")
def admin():
    return FileResponse(os.path.join(FRONTEND_DIR, "admin.html"))


# -------- API Schemas --------
class SearchReq(BaseModel):
    query: str = ""
    city: str | None = None
    top_k: int = 10


class AddJobReq(BaseModel):
    title: str
    company: str = ""
    city: str = ""
    desc: str


class DeleteReq(BaseModel):
    id: int


class LikeReq(BaseModel):
    id: int
    title: str


# -------- API Endpoints (match your HTML) --------
@app.post("/api/search")
def api_search(req: SearchReq):
    jobs = engine.recommend(req.query, top_k=req.top_k, city=req.city)
    return JSONResponse(jobs)


@app.post("/api/add_job")
def api_add_job(req: AddJobReq):
    try:
        new_id = engine.add_job(req.title, req.company, req.city, req.desc)
        return {"id": new_id}
    except ValueError as e:
        # client mistake (missing title/desc etc.)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # server bug
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/delete_job")
def api_delete_job(req: DeleteReq):
    ok = engine.delete_job(req.id)
    if not ok:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return {"status": "success"}


@app.post("/api/like")
def api_like(req: LikeReq):
    state = load_state()
    state["likes"].append({"id": req.id, "title": req.title})
    state["history"].append(req.title)
    # keep history small
    state["history"] = state["history"][-50:]
    save_state(state)
    return {"status": "ok"}


@app.get("/api/history")
def api_history():
    state = load_state()
    return state.get("history", [])


@app.post("/api/reset")
def api_reset():
    save_state({"likes": [], "history": []})
    return {"status": "ok"}
