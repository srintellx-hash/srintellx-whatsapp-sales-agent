"""Conversation memory: persist messages and build prompt history."""
from __future__ import annotations

from datetime import timedelta
from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Contact, Conversation, MessageRole
from app.utils import get_logger, now_tz

log = get_logger(__name__)

# How many past turns to feed back into the model.
HISTORY_WINDOW = 20
# If the last message is older than this, treat it as a new conversation.
CONVERSATION_TIMEOUT = timedelta(hours=2)


async def get_or_create_contact(
    db: AsyncSession, wa_phone: str, profile_name: str | None = None
) -> Contact:
    result = await db.execute(select(Contact).where(Contact.wa_phone == wa_phone))
    contact = result.scalar_one_or_none()
    if contact is None:
        contact = Contact(wa_phone=wa_phone, wa_profile_name=profile_name)
        db.add(contact)
        await db.flush()  # assign PK without committing the outer txn
        log.info("Created contact %s (%s)", contact.id, wa_phone)
    elif profile_name and not contact.wa_profile_name:
        contact.wa_profile_name = profile_name
    return contact


async def message_already_processed(db: AsyncSession, wa_message_id: str) -> bool:
    """Idempotency guard: Meta can deliver the same webhook more than once."""
    result = await db.execute(
        select(Conversation.id).where(Conversation.wa_message_id == wa_message_id)
    )
    return result.scalar_one_or_none() is not None


async def save_message(
    db: AsyncSession,
    contact: Contact,
    role: MessageRole,
    content: str,
    wa_message_id: str | None = None,
) -> Conversation:
    msg = Conversation(
        contact_id=contact.id,
        role=role,
        content=content,
        wa_message_id=wa_message_id,
    )
    db.add(msg)
    await db.flush()
    return msg


async def get_recent_history(db: AsyncSession, contact: Contact) -> List[dict]:
    """Return recent turns for the LLM, oldest first.

    If the most recent message is older than CONVERSATION_TIMEOUT, return an
    empty list so the model treats this as a fresh conversation.
    """
    result = await db.execute(
        select(Conversation)
        .where(Conversation.contact_id == contact.id)
        .order_by(Conversation.id.desc())
        .limit(HISTORY_WINDOW)
    )
    rows = list(result.scalars().all())
    if not rows:
        return []

    # Check if the conversation is stale.
    most_recent = rows[0]
    if most_recent.created_at and (now_tz() - most_recent.created_at.replace(
        tzinfo=now_tz().tzinfo
    ) if most_recent.created_at.tzinfo is None else now_tz() - most_recent.created_at) > CONVERSATION_TIMEOUT:
        log.info("Conversation timeout for contact %s — starting fresh.", contact.id)
        return []

    rows.reverse()
    items: List[dict] = []
    for r in rows:
        if r.role == MessageRole.system:
            continue
        items.append({"role": r.role.value, "content": r.content})
    return items
