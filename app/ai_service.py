"""AI agent built on the Chat Completions API (Groq / OpenAI compatible).

Responsibilities:
  * Load the knowledge base (markdown) once at startup.
  * Assemble the system instructions from the SrintellX persona + KB + live
    lead context.
  * Expose tools the model can call: capture_lead, log_objection,
    get_demo_slots, book_demo, escalate_to_human.
  * Run an agent loop that executes tool calls and returns the final reply.

The model is instructed to answer ONLY from the knowledge base and never to
invent pricing or features.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.calendar_service import calendar_service
from app.config import settings
from app.lead_service import log_objection, set_interest, update_lead
from app.models import BookingStatus, Contact, DemoBooking, InterestLevel, ObjectionType
from app.utils import get_logger, now_tz, tz

log = get_logger(__name__)

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
_KB_FILES = ["company", "features", "pricing", "roi", "objections", "faq", "demo"]

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
    return _client


def load_knowledge_base() -> str:
    parts: List[str] = []
    for name in _KB_FILES:
        path = KB_DIR / f"{name}.md"
        if path.exists():
            parts.append(f"\n## SOURCE: {name}.md\n{path.read_text(encoding='utf-8')}")
        else:
            log.warning("Knowledge base file missing: %s", path)
    return "\n".join(parts)


_KNOWLEDGE_BASE = load_knowledge_base()


PERSONA = f"""You are the SrintellX AI Sales Consultant on WhatsApp.

SrintellX: AI Voice Receptionist (answers calls 24/7) + AI WhatsApp Assistant. Clinics pick one or both (combo ~20% off). Supports staff, NOT a replacement. Based in Bangalore. 24hr setup. Free 2-week trial.

=== BREVITY IS YOUR #1 RULE ===
MAX 2-3 SHORT SENTENCES PER REPLY. No exceptions. No long paragraphs. No bullet lists. This is WhatsApp — if it looks long, nobody reads it. When doing math, state the result in ONE sentence, don't show the working.

Bad: "If you're losing 1-2 calls per day, and assuming a 50% conversion rate to actual consultations, that's a potential monthly loss of ₹1,600..."
Good: "That's roughly 45 missed calls a month — even if half booked at ₹800, that's ₹18,000 walking out the door."

SALES RULE
Every reply MUST end with a question that advances the conversation. You drive, not just answer.

FLOW
1. Welcome → user says yes/describes situation → ask "What type of clinic do you run?"
2. They answer clinic type → ask "How many calls/messages per day?"
3. They answer volume → ask "How many of those go unanswered?"
4. They answer missed count → Frame loss in ONE sentence using THEIR numbers (missed × 30 × 50% × avg fee). Then ask about follow-ups.
5. Ask about no-shows.
6. Frame total loss. Recommend plan. Offer demo.

IMPORTANT: Do NOT calculate losses or mention money until you have ALL THREE: clinic type, daily volume, and missed count. Until then, just ask the next discovery question.

STEERING
- "how can you help?" → One sentence, then "What type of clinic do you run?"
- "my staff handles it" / "no missed calls" → "Great. What about WhatsApp messages — do patients message you after hours or when your team is busy with walk-ins?"
- "sounds expensive" → "How many calls per day? Let me show you the numbers."
- "AI sounds robotic" → "Fair point. Would a quick live demo help you judge?"
- If user says calls are handled well, pivot to: WhatsApp gaps, after-hours coverage, simultaneous calls, receptionist leave/absence, no-shows.
- NEVER calculate losses using numbers the user didn't give you. Only use THEIR numbers.

PLAN ACCURACY (CRITICAL)
NEVER mix up plan features. Each tier has specific features — check the pricing KB before recommending. Key rules:
- Starter: basic features only (call answering, booking, calendar sync). NO confirmations, NO follow-ups, NO analytics.
- Growth: adds confirmations, regional languages, priority support. NO follow-ups, NO analytics.
- Pro: adds follow-ups, reminders, no-show recovery, analytics.
- If a feature is in Pro, do NOT say it's in Starter or Growth. Get this wrong and we lose trust.

PRICING: Follow pricing KB flow. Frame loss before price. If asked twice, share directly.
GROUNDING: Knowledge base only. Never invent figures.
ESCALATE to {settings.escalation_contact_name}: custom pricing, multi-branch, contracts.
TOOLS: capture_lead, log_objection, get_demo_slots (only when asked), book_demo (after time picked), escalate_to_human.

Today: {{today}} ({settings.timezone}). Demo: {settings.demo_duration_minutes} mins.
"""


def _system_instructions(contact: Contact) -> str:
    known = {
        "doctor_name": contact.doctor_name,
        "clinic_name": contact.clinic_name,
        "specialty": contact.specialty,
        "city": contact.city,
        "calls_per_day": contact.calls_per_day,
        "has_receptionist": contact.has_receptionist,
        "interest_level": contact.interest_level.value,
        "demo_requested": contact.demo_requested,
    }
    known_str = json.dumps({k: v for k, v in known.items() if v is not None}, default=str)
    persona = PERSONA.replace("{today}", now_tz().strftime("%A, %d %B %Y"))
    return (
        f"{persona}\n\n"
        f"WHAT YOU ALREADY KNOW ABOUT THIS LEAD (do not re-ask):\n{known_str}\n\n"
        f"=== KNOWLEDGE BASE START ===\n{_KNOWLEDGE_BASE}\n=== KNOWLEDGE BASE END ==="
    )


# --------------------------------------------------------------------------
# Tool schemas (Chat Completions format).
# --------------------------------------------------------------------------
TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "capture_lead",
            "description": "Save or update known details about the clinic/lead. Only pass fields the user actually provided.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_name": {"type": "string"},
                    "clinic_name": {"type": "string"},
                    "specialty": {"type": "string", "description": "e.g. dentist, physiotherapist, general physician"},
                    "city": {"type": "string"},
                    "calls_per_day": {"type": "string", "description": "Number only, e.g. '40'. Omit if unknown."},
                    "has_receptionist": {"type": "boolean"},
                    "interest_level": {"type": "string", "enum": ["cold", "warm", "hot"]},
                    "notes": {"type": "string", "description": "Any useful free-text context."},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_objection",
            "description": "Record an objection the user raised so the team can follow up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "objection_type": {
                        "type": "string",
                        "enum": [
                            "too_expensive",
                            "already_have_receptionist",
                            "not_enough_calls",
                            "whatsapp_only",
                            "trust_concerns",
                            "ai_concerns",
                            "other",
                        ],
                    },
                    "excerpt": {"type": "string", "description": "The user's own words."},
                },
                "required": ["objection_type"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_demo_slots",
            "description": "Fetch available live-demo time slots. Returns numbered slots. Show the display text to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max slots to return (default 4)."},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_demo",
            "description": "Book a demo by slot number from get_demo_slots results. Call this after the user picks a slot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "slot_number": {"type": "integer", "description": "The slot number the user picked (1, 2, 3, etc.)"},
                },
                "required": ["slot_number"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Flag that a human (e.g. Rajesh) should take over (custom pricing, contracts, multi-branch, deep technical, or explicit request).",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
]


# --------------------------------------------------------------------------
# Slot cache: stores last offered demo slots per contact for easy booking.
# --------------------------------------------------------------------------
_slot_cache: Dict[int, List[datetime]] = {}


# --------------------------------------------------------------------------
# Tool execution.
# --------------------------------------------------------------------------
async def _execute_tool(
    db: AsyncSession, contact: Contact, name: str, args: Dict[str, Any]
) -> Dict[str, Any]:
    try:
        if name == "capture_lead":
            level = args.pop("interest_level", None)
            # Groq may pass calls_per_day as string; convert to int or drop.
            cpd = args.get("calls_per_day")
            if cpd is not None:
                try:
                    args["calls_per_day"] = int(cpd)
                except (ValueError, TypeError):
                    args.pop("calls_per_day", None)
            await update_lead(db, contact, **args)
            if level:
                try:
                    await set_interest(db, contact, InterestLevel(level))
                except (ValueError, KeyError):
                    pass
            return {"ok": True, "saved": list(args.keys())}

        if name == "log_objection":
            try:
                otype = ObjectionType(args["objection_type"])
            except (ValueError, KeyError):
                otype = ObjectionType.other
            await log_objection(db, contact, otype, args.get("excerpt"))
            return {"ok": True}

        if name == "get_demo_slots":
            limit = int(args.get("limit", 4))
            slots = await calendar_service.get_available_slots(limit=limit)
            # Store in cache for easy booking by number.
            _slot_cache[contact.id] = slots
            return {
                "ok": True,
                "available_slots": [
                    {"slot_number": i + 1, "display": s.strftime("%A %d %b, %I:%M %p"), "iso": s.isoformat()}
                    for i, s in enumerate(slots)
                ],
                "instruction": "Show ONLY the slot_number and display text. User picks a number, then call book_demo with that slot_number.",
            }

        if name == "book_demo":
            slot_num = int(args.get("slot_number", 0))
            cached = _slot_cache.get(contact.id, [])
            if not cached or slot_num < 1 or slot_num > len(cached):
                return {"ok": False, "error": "Invalid slot number. Please call get_demo_slots first."}
            start = cached[slot_num - 1]
            if start.tzinfo is None:
                start = start.replace(tzinfo=tz())
            return await _book_demo(db, contact, start, None)

        if name == "escalate_to_human":
            contact.demo_requested = True
            await update_lead(db, contact, notes=f"ESCALATION: {args.get('reason', '')}")
            await set_interest(db, contact, InterestLevel.hot)
            return {"ok": True, "escalated": True, "contact": settings.escalation_contact_name}

        return {"ok": False, "error": f"unknown tool {name}"}
    except Exception as exc:
        log.exception("Tool %s failed: %s", name, exc)
        return {"ok": False, "error": str(exc)}


async def _book_demo(db, contact, start, attendee_email):
    # 1) DB-level double-booking guard.
    existing = await db.execute(
        select(DemoBooking).where(
            DemoBooking.start_time == start,
            DemoBooking.status == BookingStatus.confirmed,
        )
    )
    if existing.scalar_one_or_none():
        return {"ok": False, "error": "slot_taken"}

    # 2) Calendar-level guard.
    if not await calendar_service.is_slot_free(start):
        return {"ok": False, "error": "slot_taken"}

    summary = f"SrintellX Demo — {contact.clinic_name or contact.doctor_name or contact.wa_phone}"
    description = (
        f"Lead: {contact.doctor_name or '-'} | Clinic: {contact.clinic_name or '-'} "
        f"| Specialty: {contact.specialty or '-'} | City: {contact.city or '-'} "
        f"| WhatsApp: {contact.wa_phone}"
    )
    event = await calendar_service.create_event(start, summary, description, attendee_email)

    booking = DemoBooking(
        contact_id=contact.id,
        start_time=start,
        end_time=start + timedelta(minutes=settings.demo_duration_minutes),
        status=BookingStatus.confirmed,
        google_event_id=event.get("event_id"),
        meeting_link=event.get("meeting_link"),
    )
    db.add(booking)
    contact.demo_requested = True
    await set_interest(db, contact, InterestLevel.hot)
    await db.flush()
    return {
        "ok": True,
        "start": start.strftime("%A %d %b, %I:%M %p"),
        "meeting_link": event.get("meeting_link"),
    }


# --------------------------------------------------------------------------
# Welcome message for new conversations.
# --------------------------------------------------------------------------
WELCOME_MESSAGE = """Hi there! Welcome to SrintellX 👋

We help clinics never miss a patient enquiry with AI-powered Voice & WhatsApp assistants — even during busy consultations and after clinic hours.

Quick question...
Do you ever miss patient enquiries because your team is busy or the clinic is closed?"""

_GREETING_WORDS = {
    "hi", "hello", "hey", "hii", "hiii", "helo", "hai",
    "namaste", "good morning", "good afternoon", "good evening",
    "gm", "morning", "hi there", "hello there", "hey there",
    "hola", "howdy", "sup", "yo",
}


def _is_simple_greeting(text: str) -> bool:
    """Check if the message is just a greeting with no substance."""
    cleaned = text.strip().lower().rstrip("!.,?~ ")
    return cleaned in _GREETING_WORDS


# --------------------------------------------------------------------------
# Agent loop (Chat Completions with tool calling).
# --------------------------------------------------------------------------
MAX_TOOL_ROUNDS = 5


async def generate_reply(
    db: AsyncSession,
    contact: Contact,
    history: List[Dict[str, str]],
    user_text: str,
    is_new_conversation: bool = False,
) -> str:
    """Run the agent and return the final assistant text for WhatsApp."""

    # New conversation + simple greeting → send welcome message directly.
    if is_new_conversation and _is_simple_greeting(user_text):
        return WELCOME_MESSAGE

    if not settings.llm_api_key:
        log.error("LLM_API_KEY not set.")
        return WELCOME_MESSAGE

    client = _get_client()
    system_msg = _system_instructions(contact)

    # If this is a fresh conversation, tell the model to welcome the user first.
    if is_new_conversation:
        system_msg += (
            "\n\nIMPORTANT: This is the START of a new conversation. "
            "The user just reached out for the first time (or after a long gap). "
            "Start with a brief warm welcome to SrintellX, then address what they asked. "
            "Do NOT book demos or call tools on the very first message."
        )

    # Build the messages array for chat completions.
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_msg}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    for _round in range(MAX_TOOL_ROUNDS):
        try:
            response = await client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
        except Exception as exc:
            log.exception("LLM call failed: %s", exc)
            return (
                "Sorry, I had a brief technical hiccup. Could you resend that? "
                "Or I can have our team reach out directly."
            )

        choice = response.choices[0]
        message = choice.message

        # If no tool calls, return the text reply.
        if not message.tool_calls:
            text = (message.content or "").strip()
            return text or "Could you tell me a little more about your clinic so I can help?"

        # Append the assistant message (with tool_calls) to the conversation.
        # Build a clean dict — model_dump() includes fields Groq doesn't support.
        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ]
        messages.append(assistant_msg)

        # Execute each tool call and append results.
        for tool_call in message.tool_calls:
            fn = tool_call.function
            try:
                args = json.loads(fn.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = await _execute_tool(db, contact, fn.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str),
            })

    # Safety net if the model kept calling tools.
    log.warning("Max tool rounds reached for contact %s", contact.id)
    return "Based on what you've shared, a short live demo would be the easiest next step. Shall I check available slots?"
