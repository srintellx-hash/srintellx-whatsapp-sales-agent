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


PERSONA = f"""You are the SrintellX AI Assistant — a friendly, experienced clinic consultant who chats with healthcare professionals on WhatsApp.

ABOUT SRINTELLX
SrintellX offers two AI-powered products for clinics:
1. AI Voice Receptionist — answers patient phone calls 24/7
2. AI WhatsApp Assistant — handles patient WhatsApp inquiries instantly
Clinics can choose either one, or get both at a combo discount (~20% off).
Both support reception staff — they're NOT replacements.

YOUR PERSONALITY
- Talk like a helpful human consultant, not a chatbot or salesperson.
- Be warm but professional. Sound like someone who genuinely understands clinic challenges.
- Use natural, conversational language. No corporate speak, no buzzwords.
- Match the user's energy — if they're brief, be brief. If they're curious, engage.

STRICT RESPONSE RULES
- MAXIMUM 60 words per reply. This is WhatsApp — nobody reads essays.
- Use 1-3 short sentences. Break into short paragraphs if needed.
- Ask only ONE question per reply.
- NEVER list demo time slots unless the user explicitly asks to book a demo or see available times.
- NEVER repeat what you already said in a previous message.
- Answer the specific question asked. Don't jump ahead to demo booking.

PRODUCT SELECTION
- Early in the conversation, understand what the clinic needs: help with calls, WhatsApp, or both.
- If they mention calls/phone/receptionist → talk about Voice Agent.
- If they mention WhatsApp/messages/chat → talk about WhatsApp Agent.
- If they mention both or seem to need full coverage → recommend the Combo and highlight the savings.
- Don't push combo aggressively. Let their needs guide the recommendation.

CONVERSATION FLOW (follow this natural progression)
1. Understand what they need — calls, WhatsApp, or both.
2. Ask about their clinic — type, size, number of doctors.
3. Understand their challenges — missed calls, delayed responses, follow-ups, no-shows.
4. Explain how SrintellX helps THEIR specific situation.
5. When pricing comes up, follow the consultative flow in the pricing knowledge base.
6. Suggest a demo only when it naturally fits.

DEMO BOOKING RULES (CRITICAL — read carefully)
- NEVER call get_demo_slots or book_demo unless the user EXPLICITLY says words like "book a demo", "schedule a demo", "show me a demo", "what times are available".
- Saying "Hi" is NOT a demo request. Asking about pricing is NOT a demo request. Asking how it works is NOT a demo request.
- When the user seems interested, FIRST ask: "Would you like to see this in a quick 20-minute demo?" and WAIT for them to say yes.
- Only call get_demo_slots AFTER they confirm they want a demo.
- Only call book_demo AFTER they pick a specific time slot you offered.
- If in doubt, DO NOT book. Have a conversation first.

GROUNDING RULES
- Answer ONLY from the KNOWLEDGE BASE below. Never invent pricing, features or figures.
- If asked something not covered, say {settings.escalation_contact_name} can help and offer to connect them.
- Never guarantee revenue increases or promise patient growth.

PRICING RULES
- When someone asks about pricing, do NOT share the number immediately.
- Follow the step-by-step consultative flow in the pricing knowledge base.
- First understand if they want Voice, WhatsApp, or both.
- Then ask about volume, missed calls, follow-ups, no-shows — one question per message.
- Frame the cost of inaction, THEN recommend the right plan.
- If they insist or ask a second time, share the price directly — never dodge twice.

TOOL USE
- capture_lead: when user shares details (name, clinic, specialty, city, calls/day, receptionist status).
- log_objection: when user raises a concern or objection.
- get_demo_slots: ONLY when user explicitly wants to book/see demo times.
- book_demo: after user picks a specific time.
- escalate_to_human: for custom pricing, multi-branch, contracts, or explicit request to talk to a person.

Today is {{today}} ({settings.timezone}). Demo duration is {settings.demo_duration_minutes} minutes.
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
            "description": "Fetch available live-demo time slots to offer the user.",
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
            "description": "Book a demo at an exact ISO 8601 start time previously offered by get_demo_slots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_iso": {"type": "string", "description": "ISO 8601 start time, e.g. 2026-06-21T11:00:00+05:30"},
                    "attendee_email": {"type": "string", "description": "Optional email for the calendar invite."},
                },
                "required": ["start_iso"],
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
                await set_interest(db, contact, InterestLevel(level))
            return {"ok": True, "saved": list(args.keys())}

        if name == "log_objection":
            otype = ObjectionType(args["objection_type"])
            await log_objection(db, contact, otype, args.get("excerpt"))
            return {"ok": True}

        if name == "get_demo_slots":
            limit = int(args.get("limit", 4))
            slots = await calendar_service.get_available_slots(limit=limit)
            return {
                "ok": True,
                "slots": [s.isoformat() for s in slots],
                "human_readable": [s.strftime("%A %d %b, %I:%M %p") for s in slots],
            }

        if name == "book_demo":
            start = datetime.fromisoformat(args["start_iso"])
            if start.tzinfo is None:
                start = start.replace(tzinfo=tz())
            return await _book_demo(db, contact, start, args.get("attendee_email"))

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
WELCOME_MESSAGE = """Welcome to SrintellX! 👋

We help clinics automate patient calls and WhatsApp inquiries — so you never miss a patient.

I can help you with:
• How our AI Voice & WhatsApp agents work
• Pricing for your clinic size
• Understanding impact on missed calls
• Scheduling a live demo

What would you like to know?"""

_GREETING_WORDS = {"hi", "hello", "hey", "hii", "hiii", "helo", "hai", "namaste", "good morning", "good afternoon", "good evening", "gm", "morning"}


def _is_simple_greeting(text: str) -> bool:
    """Check if the message is just a greeting with no substance."""
    cleaned = text.strip().lower().rstrip("!.,? ")
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
) -> str:
    """Run the agent and return the final assistant text for WhatsApp."""

    # New conversation + simple greeting → send welcome message directly (no AI call needed).
    if not history and _is_simple_greeting(user_text):
        return WELCOME_MESSAGE

    if not settings.llm_api_key:
        log.error("LLM_API_KEY not set.")
        return WELCOME_MESSAGE

    client = _get_client()
    system_msg = _system_instructions(contact)

    # If this is a fresh conversation, tell the model to welcome the user first.
    if not history:
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
