"""Lead management: update contact fields, score interest, log objections."""
from __future__ import annotations

import re
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Contact, InterestLevel, Objection, ObjectionType
from app.utils import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------
# Lead field updates (called from AI tool execution).
# --------------------------------------------------------------------------
async def update_lead(db: AsyncSession, contact: Contact, **fields) -> Contact:
    """Patch known lead fields. Never overwrites a set value with None."""
    allowed = {
        "doctor_name",
        "clinic_name",
        "specialty",
        "city",
        "calls_per_day",
        "has_receptionist",
        "demo_requested",
        "notes",
    }
    for key, value in fields.items():
        if key not in allowed or value is None:
            continue
        if key == "notes" and contact.notes:
            contact.notes = f"{contact.notes}\n{value}"
        else:
            setattr(contact, key, value)
    await db.flush()
    log.info("Updated lead %s: %s", contact.id, ", ".join(fields.keys()))
    return contact


async def set_interest(db: AsyncSession, contact: Contact, level: InterestLevel) -> None:
    # Interest only escalates automatically; downgrade only if explicit.
    order = {InterestLevel.cold: 0, InterestLevel.warm: 1, InterestLevel.hot: 2}
    if order[level] >= order[contact.interest_level]:
        contact.interest_level = level
        await db.flush()


# --------------------------------------------------------------------------
# Objection classification.
# --------------------------------------------------------------------------
_OBJECTION_PATTERNS = {
    ObjectionType.too_expensive: [
        r"\bexpensive\b", r"\bcostly\b", r"too much", r"\bbudget\b",
        r"can'?t afford", r"\bprice(y)?\b high",
    ],
    ObjectionType.already_have_receptionist: [
        r"already have (a )?receptionist", r"have (a )?front desk",
        r"have staff", r"my receptionist",
    ],
    ObjectionType.not_enough_calls: [
        r"not many calls", r"few calls", r"don'?t (get|receive) many",
        r"low call volume", r"hardly any calls",
    ],
    ObjectionType.whatsapp_only: [
        r"\bonly\b.{0,15}whatsapp", r"\bjust\b.{0,15}whatsapp",
        r"whatsapp\b.{0,10}\bonly\b", r"whatsapp\b.{0,15}enough",
    ],
    ObjectionType.trust_concerns: [
        r"don'?t trust", r"not sure i trust", r"is it safe", r"data (safe|secure|privacy)",
    ],
    ObjectionType.ai_concerns: [
        r"\bai\b.*(mistake|wrong|error)", r"robot", r"sounds? robotic",
        r"patients? prefer humans?", r"too impersonal",
    ],
}
_COMPILED = {
    t: [re.compile(p, re.IGNORECASE) for p in pats]
    for t, pats in _OBJECTION_PATTERNS.items()
}


def classify_objection(text: str) -> Optional[ObjectionType]:
    """Heuristic classifier used as a backstop to the AI tool call."""
    for otype, patterns in _COMPILED.items():
        if any(p.search(text) for p in patterns):
            return otype
    return None


async def log_objection(
    db: AsyncSession,
    contact: Contact,
    objection_type: ObjectionType,
    excerpt: str | None = None,
) -> Objection:
    # De-duplicate: don't stack the same unresolved objection repeatedly.
    existing = await db.execute(
        select(Objection).where(
            Objection.contact_id == contact.id,
            Objection.objection_type == objection_type,
            Objection.resolved.is_(False),
        )
    )
    obj = existing.scalar_one_or_none()
    if obj:
        return obj
    obj = Objection(contact_id=contact.id, objection_type=objection_type, excerpt=excerpt)
    db.add(obj)
    await db.flush()
    log.info("Logged objection %s for contact %s", objection_type.value, contact.id)
    return obj


# --------------------------------------------------------------------------
# Hot-lead signal detection (backstop to AI scoring).
# --------------------------------------------------------------------------
_HOT_SIGNALS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bpric(e|ing)\b", r"\bdemo\b", r"missed calls?", r"book",
        r"overload", r"too busy", r"delayed responses?",
    ]
]
_WARM_SIGNALS = [
    re.compile(p, re.IGNORECASE)
    for p in [r"how does it work", r"\bfeatures?\b", r"tell me more", r"interested"]
]


def detect_interest(text: str) -> InterestLevel:
    if any(p.search(text) for p in _HOT_SIGNALS):
        return InterestLevel.hot
    if any(p.search(text) for p in _WARM_SIGNALS):
        return InterestLevel.warm
    return InterestLevel.cold
