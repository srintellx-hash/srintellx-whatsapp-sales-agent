"""FastAPI application entrypoint."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import Contact, DemoBooking, Objection
from app.schemas import BookingOut, ContactOut
from app.utils import configure_logging, get_logger
from app.webhook import router as webhook_router

configure_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting %s (env=%s)", settings.app_name, settings.environment)
    yield
    log.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="WhatsApp AI Sales & Support agent for SrintellX.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(webhook_router)


@app.get("/", tags=["health"])
async def root():
    return {"service": settings.app_name, "status": "ok", "env": settings.environment}


@app.get("/health", tags=["health"])
async def health(db: AsyncSession = Depends(get_db)):
    await db.execute(select(1))
    return {"status": "healthy", "database": "connected"}


# --------------------------------------------------------------------------
# Minimal read-only admin endpoints for inspecting captured data.
# Protect these behind auth/network rules before exposing publicly.
# --------------------------------------------------------------------------
@app.get("/admin/leads", response_model=list[ContactOut], tags=["admin"])
async def list_leads(limit: int = 50, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Contact).order_by(Contact.updated_at.desc()).limit(limit)
    )
    return list(result.scalars().all())


@app.get("/admin/bookings", response_model=list[BookingOut], tags=["admin"])
async def list_bookings(limit: int = 50, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(DemoBooking).order_by(DemoBooking.start_time.desc()).limit(limit)
    )
    return list(result.scalars().all())


@app.get("/admin/stats", tags=["admin"])
async def stats(db: AsyncSession = Depends(get_db)):
    leads = await db.scalar(select(func.count()).select_from(Contact))
    bookings = await db.scalar(select(func.count()).select_from(DemoBooking))
    objections = await db.scalar(select(func.count()).select_from(Objection))
    return {"leads": leads, "bookings": bookings, "objections": objections}
