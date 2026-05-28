#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging
from fastapi import FastAPI, Request, Depends, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd
import io
from pathlib import Path

from .database import engine, Base, get_db
from .models import Session, Event
from .schema import RecommendOut
from .recommender import process_session

logger = logging.getLogger("uvicorn.error")

# Папка web/app
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()

# Шаблоны из web/app/templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
# Сервим статику из web/app/static на /static
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)

@app.on_event("startup")
async def on_startup():
    # Создаём таблицы, если их нет
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ingest/")
async def ingest(
    log_file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    # Парсим CSV
    try:
        content = await log_file.read()
        df = pd.read_csv(
            io.BytesIO(content),
            sep=";",
            encoding="cp1251",
            parse_dates=["watch_tms"]
        )
    except Exception as e:
        logger.error(f"Ingest failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Cannot parse CSV: {e}")

    # Проверяем колонки
    required = {"client_id", "watch_tms", "goal_nm_lvl1", "device_id"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join(missing)}")

    # Простая сессизация по 3 событиям
    df["session_index"] = df.groupby("client_id").cumcount() // 3
    df["session_id"] = df["client_id"].astype(str) + "_" + df["session_index"].astype(str)

    # Сохраняем сессии и события
    for _, row in df.iterrows():
        if pd.isna(row.client_id) or pd.isna(row.watch_tms):
            continue
        # Создаём сессию при необходимости
        result = await db.execute(
            Session.__table__.select().where(Session.id == row.session_id)
        )
        sess = result.scalar_one_or_none()
        if not sess:
            sess = Session(
                id=row.session_id,
                user_id=None,
                started_at=row.watch_tms
            )
            db.add(sess)
        # Добавляем событие
        ev = Event(
            session_id=row.session_id,
            ts=row.watch_tms,
            raw_action=row.goal_nm_lvl1,
            action_id=None,
            p_flow=None
        )
        db.add(ev)

    await db.commit()
    return {"status": "ok", "sessions_created": int(df["session_id"].nunique())}

@app.post("/recommend/", response_model=RecommendOut)
async def recommend(
    session_id: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    # Получаем события сессии
    result = await db.execute(
        Event.__table__
        .select()
        .where(Event.session_id == session_id)
        .order_by(Event.ts)
    )
    events = result.scalars().all()
    if len(events) < 3:
        raise HTTPException(status_code=400, detail="Not enough events for recommendation")

    # Считаем churn_prob и recommendation
    churn_prob, recommended = process_session(events)

    # Сохраняем в таблицу Session
    await db.execute(
        Session.__table__
        .update()
        .where(Session.id == session_id)
        .values(
            churn_prob=churn_prob,
            recommended=recommended,
            last_event=events[-1].ts
        )
    )
    await db.commit()

    return RecommendOut(
        session_id=session_id,
        churn_prob=churn_prob,
        recommended_actions=[recommended]
    )

@app.get("/download/actions.csv")
async def download_actions():
    path = Path("data/logs/latest_recommendations.csv")
    if not path.exists():
        raise HTTPException(status_code=404, detail="No recommendations CSV available")
    return FileResponse(path, media_type="text/csv", filename="actions.csv")