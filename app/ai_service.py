"""AI agent built on the Chat Completions API (OpenAI / Groq compatible).

Architecture:
  * Slim system prompt (~100 words): identity, safety, tools, format.
  * Behavior files (markdown): how to sell — loaded from knowledge_base/behavior/
  * Product files (markdown): what to sell — loaded from knowledge_base/
  * Tools: capture_lead, log_objection, get_demo_slots, book_demo, escalate_to_human.
  * Leaked function call parser: catches and executes Llama-style text leaks.
"""
from __future__ import annotations

import json
import re
import time
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

# --------------------------------------------------------------------------
# Knowledge base loading.
# --------------------------------------------------------------------------
KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
_BEHAVIOR_DIR = KB_DIR / "behavior"
_BEHAVIOR_FILES = [
    "01_operating_principles",
    "02_personality",
    "03_sales_methodology",
    "04_conversation_stages",
    "05_conversation_rules",
    "06_psychology",
    "07_discovery_questions",
    # "08_case_library",  # Enable when real cases are added
    "09_sales_playbook",
    "response_patterns",
    "conversation_examples",
]
_PRODUCT_FILES = ["company", "features", "pricing", "roi", "objections", "faq", "demo"]

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
    return _client


def _load_files(directory: Path, names: list[str]) -> str:
    parts: list[str] = []
    for name in names:
        path = directory / f"{name}.md"
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
        else:
            log.warning("KB file missing: %s", path)
    return "\n\n---\n\n".join(parts)


# Load once at startup.
_BEHAVIOR = _load_files(_BEHAVIOR_DIR, _BEHAVIOR_FILES)
_PRODUCT = _load_files(KB_DIR, _PRODUCT_FILES)


# --------------------------------------------------------------------------
# System prompt: identity, safety, tools, grounding, format. Nothing else.
# All behavior and product knowledge comes from the markdown files.
# --------------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are the SrintellX AI Sales Consultant. You talk to clinic owners on WhatsApp.

SAFETY
- Answer ONLY from the knowledge base below. Never invent pricing, features, or figures.
- Never include function calls, XML tags, or code in your response.
- If asked something not covered, say {settings.escalation_contact_name} will follow up.

TOOLS
- capture_lead: when user shares clinic details (name, specialty, city, calls/day).
- log_objection: when user raises a concern.
- escalate_to_human: custom pricing, multi-branch, contracts, or request to talk to a person.

DEMO BOOKING (critical — follow exactly)
- When the user wants to see a demo, include the exact marker [DEMO_SLOTS] at the end of your response. The system will replace it with real available time slots.
- NEVER invent or list demo times yourself. NEVER say "Monday 8:30 PM" or any specific time.
- NEVER say "I've booked" — the system handles booking when the user picks a slot number.
- Example response: "A 20-minute demo is the best way to see it in action. Let me check what's available. [DEMO_SLOTS]"

FORMAT
- WhatsApp messages. 20-40 words max. 1-2 sentences. One question per reply.

Today: {{today}} ({settings.timezone}).
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
    prompt = SYSTEM_PROMPT.replace("{today}", now_tz().strftime("%A, %d %B %Y"))
    return (
        f"{prompt}\n\n"
        f"LEAD CONTEXT (do not re-ask what you already know):\n{known_str}\n\n"
        f"=== BEHAVIOR ===\n{_BEHAVIOR}\n\n"
        f"=== PRODUCT KNOWLEDGE ===\n{_PRODUCT}"
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
# Now includes a timestamp to detect stale caches.
# --------------------------------------------------------------------------
_slot_cache: Dict[int, Dict[str, Any]] = {}
_CACHE_MAX_AGE = 600  # 10 minutes


def _cache_slots(contact_id: int, slots: list) -> None:
    _slot_cache[contact_id] = {"slots": slots, "timestamp": time.monotonic()}


def _get_cached_slots(contact_id: int) -> list | None:
    entry = _slot_cache.get(contact_id)
    if not entry:
        return None
    if time.monotonic() - entry["timestamp"] > _CACHE_MAX_AGE:
        log.info("Slot cache stale for contact %s", contact_id)
        return None
    return entry["slots"]


# --------------------------------------------------------------------------
# Tool execution.
# --------------------------------------------------------------------------
async def _execute_tool(
    db: AsyncSession, contact: Contact, name: str, args: Dict[str, Any]
) -> Dict[str, Any]:
    try:
        args = args or {}  # Llama models sometimes pass None

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
            _cache_slots(contact.id, slots)
            return {
                "ok": True,
                "available_slots": [
                    {"slot_number": i + 1, "display": s.strftime("%A %d %b, %I:%M %p"), "iso": s.isoformat()}
                    for i, s in enumerate(slots)
                ],
                "instruction": "Show the numbered slots. End with: 'Reply with the slot number (1, 2, or 3) that works for you.'",
            }

        if name == "book_demo":
            slot_num = int(args.get("slot_number", 0))
            cached = _get_cached_slots(contact.id)
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
    event = await calendar_service.create_event(start, summary, description)

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
# Response cleanup: strip leaked function call syntax.
# --------------------------------------------------------------------------
_CLEANUP_PATTERNS = [
    re.compile(r'</?function[^>]*>(\{.*?\})?', re.DOTALL),
    re.compile(r'You can (?:also )?use the following function[^.]*\.?', re.IGNORECASE),
    re.compile(r'[Pp]lease wait[^.]*\.\.\.?'),
    re.compile(r'Let me (?:check|fetch|get)[^.]*\.\.\.?'),
    re.compile(r'\{[^}]*"function"[^}]*\}'),
    re.compile(r'</?tool[^>]*>'),
]


def _parse_leaked_calls(text: str) -> List[tuple]:
    """Extract function calls leaked as text by Llama models."""
    results = []
    for match in re.finditer(r'<function=(\w+)>\s*(\{[^}]*\})', text):
        name = match.group(1)
        try:
            args = json.loads(match.group(2))
        except json.JSONDecodeError:
            args = {}
        results.append((name, args))
    return results


def _clean_response(text: str) -> str:
    """Strip leaked function calls and artifacts from model output."""
    if not text:
        return text
    for pattern in _CLEANUP_PATTERNS:
        text = pattern.sub('', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'  +', ' ', text)
    return text.strip()


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

        # If no tool calls, check for leaked function calls in the text.
        if not message.tool_calls:
            raw_text = (message.content or "").strip()

            # Parse and execute any leaked function calls.
            leaked = _parse_leaked_calls(raw_text)
            extra_info = ""
            for func_name, func_args in leaked:
                log.info("Executing leaked tool call: %s(%s)", func_name, func_args)
                result = await _execute_tool(db, contact, func_name, func_args)
                if result.get("ok"):
                    if func_name == "book_demo" and result.get("start"):
                        extra_info = f"\n\nYour demo is booked for {result['start']}. See you then!"
                    elif func_name == "get_demo_slots" and result.get("available_slots"):
                        slots_text = "\n".join(
                            f"{s['slot_number']}. {s['display']}" for s in result["available_slots"]
                        )
                        extra_info = f"\n\nHere are the available slots:\n{slots_text}\n\nReply with the slot number (1, 2, or 3) that works for you."

            text = _clean_response(raw_text) + extra_info
            return text.strip() or "Could you tell me a little more about your clinic so I can help?"

        # Append the assistant message (with tool_calls) to the conversation.
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
