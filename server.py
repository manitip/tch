import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, confloat, conint, constr

app = FastAPI(title="Telegram WebApp Demo")

storage: Dict[int, List[dict]] = {}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EntryIn(BaseModel):
    user_id: conint(gt=0) = Field(..., description="Telegram user id")
    title: constr(min_length=1, max_length=50) = Field(..., description="1..50 chars")
    amount: confloat(gt=0) = Field(..., description="> 0")


class EntryOut(BaseModel):
    id: str
    title: str
    amount: float
    createdAt: str


class EntriesResponse(BaseModel):
    items: List[EntryOut]


class CreateEntryResponse(BaseModel):
    item: EntryOut


@app.get("/webapp", response_class=HTMLResponse)
def webapp_page():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "webapp.html")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="webapp.html not found рядом с server.py")


@app.get("/api/entries", response_model=EntriesResponse)
def get_entries(user_id: int = Query(..., gt=0)):
    return {"items": storage.get(user_id, [])}


@app.post("/api/entries", response_model=CreateEntryResponse, status_code=201)
def create_entry(payload: EntryIn):
    # В проде нужно проверять initData подпись и НЕ доверять user_id от клиента.
    item = {
        "id": str(uuid.uuid4()),
        "title": payload.title.strip(),
        "amount": float(payload.amount),
        "createdAt": now_utc_iso(),
    }
    storage.setdefault(int(payload.user_id), []).insert(0, item)
    return {"item": item}


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    import uvicorn
    uvicorn.run("server:app", host=host, port=port, reload=False)
