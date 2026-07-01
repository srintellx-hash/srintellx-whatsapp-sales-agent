"""WhatsApp webhook endpoints and the inbound-message processing pipeline."""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import APIRouter, Header, Request, Response
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai_service import generate_reply
from app.config import settings
from app.conversation_service import (
    get_or_create_contact,
    get_recent_history,
    message_already_processed,
    save_message,
)
from app.database import AsyncSessionLocal
from app.lead_service import classify_objection, detect_interest, log_objection, set_interest
from app.models import MessageRole
from app.schemas import InboundMessage, WAWebhook
from app.utils import get_logger
from app.whatsapp import parse_inbound_messages, verify_signature, whatsapp_client

log = get_logger(__name__)
router = APIRouter(tags=["webhook"])


# --------------------------------------------------------------------------
# Lightweight in-memory rate limiter (per sender).
# For multi-instance production use a shared store (e.g. Redis).
# --------------------------------------------------------------------------
_WINDOW_SECONDS = 60
_hits: Dict[str, Deque[float]] = defaultdict(deque)


def _rate_limited(sender: str) -> bool:
    now = time.monotonic()
    q = _hits[sender]
    while q and now - q[0] > _WINDOW_SECONDS:
        q.popleft()
    if len(q) >= settings.rate_limit_per_minute:
        return True
    q.append(now)
    return False


# --------------------------------------------------------------------------
# GET /webhook — Meta verification handshake.
# --------------------------------------------------------------------------
@router.get("/webhook")
async def verify_webhook(
    request: Request,
) -> Response:
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")
    if mode == "subscribe" and token == settings.whatsapp_verify_token:
        log.info("Webhook verified by Meta.")
        return PlainTextResponse(content=challenge or "", status_code=200)
    log.warning("Webhook verification failed (mode=%s).", mode)
    return PlainTextResponse(content="Verification failed", status_code=403)


# --------------------------------------------------------------------------
# POST /webhook — inbound messages.
# --------------------------------------------------------------------------
@router.post("/webhook")
async def receive_webhook(
    request: Request,
    x_hub_signature_256: str | None = Header(default=None),
) -> Response:
    raw = await request.body()

    if not verify_signature(raw, x_hub_signature_256):
        log.warning("Invalid webhook signature.")
        return PlainTextResponse("invalid signature", status_code=403)

    try:
        payload = WAWebhook.model_validate_json(raw)
    except Exception as exc:
        log.warning("Malformed webhook payload: %s", exc)
        # 200 so Meta does not retry a permanently broken payload.
        return PlainTextResponse("ok", status_code=200)

    messages = parse_inbound_messages(payload)

    # Process in the background so we ACK Meta within its timeout window.
    for msg in messages:
        asyncio.create_task(_safe_process(msg))

    return PlainTextResponse("ok", status_code=200)


async def _safe_process(msg: InboundMessage) -> None:
    try:
        await process_message(msg)
    except Exception:
        log.exception("Failed processing message %s", msg.wa_message_id)


async def process_message(msg: InboundMessage) -> None:
    """Full pipeline for a single inbound text message."""
    if _rate_limited(msg.sender):
        log.warning("Rate limit hit for %s; dropping message.", msg.sender)
        return

    async with AsyncSessionLocal() as db:  # type: AsyncSession
        try:
            # Idempotency: skip duplicates Meta may redeliver.
            if await message_already_processed(db, msg.wa_message_id):
                log.info("Duplicate message %s ignored.", msg.wa_message_id)
                return

            contact = await get_or_create_contact(db, msg.sender, msg.profile_name)

            # Check if this is a new conversation BEFORE saving the message.
            history_before = await get_recent_history(db, contact)
            is_new_conversation = len(history_before) == 0

            await save_message(
                db, contact, MessageRole.user, msg.text, wa_message_id=msg.wa_message_id
            )

            # --- DIRECT SLOT BOOKING (bypasses AI entirely) ---
            direct_reply = await _try_direct_booking(db, contact, msg.text)
            if direct_reply:
                reply = direct_reply
            else:
                # Heuristic backstops (the AI also does this via tools).
                otype = classify_objection(msg.text)
                if otype:
                    await log_objection(db, contact, otype, excerpt=msg.text[:500])
                await set_interest(db, contact, detect_interest(msg.text))

                reply = await generate_reply(
                    db, contact, history_before, msg.text, is_new_conversation
                )

            await save_message(db, contact, MessageRole.assistant, reply)
            await db.commit()
        except Exception:
            await db.rollback()
            raise

    # Send outside the DB transaction.
    await whatsapp_client.mark_read(msg.wa_message_id)
    await whatsapp_client.send_text(msg.sender, reply)


async def _try_direct_booking(db, contact, text: str) -> str | None:
    """If user sent a slot number and we have cached slots, book directly.
    
    Returns the reply text if booking happened, None otherwise.
    """
    from app.ai_service import _slot_cache
    from app.models import BookingStatus, DemoBooking, InterestLevel
    from app.calendar_service import calendar_service
    from app.config import settings
    from app.lead_service import set_interest, update_lead
    from datetime import timedelta

    # Check if the message is a slot number (1-9).
    cleaned = text.strip().lower().replace("slot ", "").replace("option ", "").replace("#", "")
    try:
        slot_num = int(cleaned)
    except ValueError:
        return None

    if slot_num < 1 or slot_num > 9:
        return None

    # Check if we have cached slots for this contact.
    cached = _slot_cache.get(contact.id)
    if not cached or slot_num > len(cached):
        return None

    start = cached[slot_num - 1]
    from app.utils import tz
    if start.tzinfo is None:
        start = start.replace(tzinfo=tz())

    log.info("Direct booking: contact %s picked slot %s → %s", contact.id, slot_num, start)

    # Check if slot is still free.
    if not await calendar_service.is_slot_free(start):
        return "That slot was just taken. Let me check what's available.\n\n" + await _format_fresh_slots(contact)

    # Create calendar event.
    summary = f"SrintellX Demo — {contact.clinic_name or contact.doctor_name or contact.wa_phone}"
    description = (
        f"Lead: {contact.doctor_name or '-'} | Clinic: {contact.clinic_name or '-'} "
        f"| Specialty: {contact.specialty or '-'} | City: {contact.city or '-'} "
        f"| WhatsApp: {contact.wa_phone}"
    )
    event = await calendar_service.create_event(start, summary, description)

    # Save booking to database.
    end = start + timedelta(minutes=settings.demo_duration_minutes)
    booking = DemoBooking(
        contact_id=contact.id,
        start_time=start,
        end_time=end,
        status=BookingStatus.confirmed,
        google_event_id=event.get("event_id"),
        meeting_link=event.get("meeting_link"),
    )
    db.add(booking)
    contact.demo_requested = True
    await set_interest(db, contact, InterestLevel.hot)
    await db.flush()

    friendly_time = start.strftime("%A %d %b, %I:%M %p")
    log.info("Demo booked for contact %s at %s (event: %s)", contact.id, friendly_time, event.get("event_id"))

    return f"Your demo is booked for {friendly_time}. You'll hear from us shortly before the session. See you then! 👋"


async def _format_fresh_slots(contact) -> str:
    """Fetch fresh slots and return formatted text."""
    from app.ai_service import _slot_cache
    from app.calendar_service import calendar_service

    slots = await calendar_service.get_available_slots(limit=4)
    _slot_cache[contact.id] = slots
    lines = [f"{i+1}. {s.strftime('%A %d %b, %I:%M %p')}" for i, s in enumerate(slots)]
    return "Here are the available slots:\n" + "\n".join(lines) + "\n\nReply with the slot number (1, 2, or 3) that works for you."
