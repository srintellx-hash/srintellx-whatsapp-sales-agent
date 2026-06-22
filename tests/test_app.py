"""Unit tests covering parsing, security, classification, and persistence."""
from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import pytest

from app.config import settings
from app.conversation_service import (
    get_or_create_contact,
    get_recent_history,
    message_already_processed,
    save_message,
)
from app.lead_service import (
    classify_objection,
    detect_interest,
    log_objection,
    update_lead,
)
from app.models import InterestLevel, MessageRole, ObjectionType
from app.schemas import WAWebhook
from app.whatsapp import parse_inbound_messages, verify_signature

PAYLOADS = Path(__file__).parent / "payloads"


# ---------------- WhatsApp parsing ----------------
def test_parse_inbound_text():
    raw = (PAYLOADS / "inbound_text.json").read_text()
    webhook = WAWebhook.model_validate_json(raw)
    msgs = parse_inbound_messages(webhook)
    assert len(msgs) == 1
    m = msgs[0]
    assert m.sender == "919876543210"
    assert m.profile_name == "Dr. Meera"
    assert "pricing" in m.text.lower()
    assert m.wa_message_id.startswith("wamid.")


def test_status_callback_yields_no_messages():
    raw = (PAYLOADS / "status_callback.json").read_text()
    webhook = WAWebhook.model_validate_json(raw)
    assert parse_inbound_messages(webhook) == []


# ---------------- Signature verification ----------------
def test_signature_valid(monkeypatch):
    monkeypatch.setattr(settings, "whatsapp_app_secret", "topsecret")
    body = b'{"hello":"world"}'
    digest = hmac.new(b"topsecret", body, hashlib.sha256).hexdigest()
    assert verify_signature(body, f"sha256={digest}") is True


def test_signature_invalid(monkeypatch):
    monkeypatch.setattr(settings, "whatsapp_app_secret", "topsecret")
    assert verify_signature(b"{}", "sha256=deadbeef") is False
    assert verify_signature(b"{}", None) is False


# ---------------- Classification ----------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("This is too expensive for us", ObjectionType.too_expensive),
        ("We already have a receptionist", ObjectionType.already_have_receptionist),
        ("We don't get many calls", ObjectionType.not_enough_calls),
        ("We only need WhatsApp", ObjectionType.whatsapp_only),
        ("Is my data secure?", ObjectionType.trust_concerns),
        ("Patients prefer humans, sounds robotic", ObjectionType.ai_concerns),
        ("What are your opening hours?", None),
    ],
)
def test_classify_objection(text, expected):
    assert classify_objection(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Can I book a demo?", InterestLevel.hot),
        ("What's the pricing?", InterestLevel.hot),
        ("How does it work?", InterestLevel.warm),
        ("Just browsing", InterestLevel.cold),
    ],
)
def test_detect_interest(text, expected):
    assert detect_interest(text) == expected


# ---------------- Persistence (async) ----------------
@pytest.mark.asyncio
async def test_contact_lifecycle(db_session):
    contact = await get_or_create_contact(db_session, "919876543210", "Dr. Meera")
    assert contact.id is not None
    # Idempotent get.
    same = await get_or_create_contact(db_session, "919876543210")
    assert same.id == contact.id

    await update_lead(db_session, contact, clinic_name="Smile Dental", city="Bangalore", calls_per_day=40)
    assert contact.clinic_name == "Smile Dental"
    assert contact.calls_per_day == 40


@pytest.mark.asyncio
async def test_message_idempotency_and_history(db_session):
    contact = await get_or_create_contact(db_session, "918888888888")
    await save_message(db_session, contact, MessageRole.user, "Hello", wa_message_id="wamid.1")
    await save_message(db_session, contact, MessageRole.assistant, "Hi! How can I help your clinic?")

    assert await message_already_processed(db_session, "wamid.1") is True
    assert await message_already_processed(db_session, "wamid.does-not-exist") is False

    history = await get_recent_history(db_session, contact)
    assert history[0] == {"role": "user", "content": "Hello"}
    assert history[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_objection_dedup(db_session):
    contact = await get_or_create_contact(db_session, "917777777777")
    o1 = await log_objection(db_session, contact, ObjectionType.too_expensive, "too pricey")
    o2 = await log_objection(db_session, contact, ObjectionType.too_expensive, "again")
    assert o1.id == o2.id  # same unresolved objection reused
