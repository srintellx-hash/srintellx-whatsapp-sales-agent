"""Conversation memory: persist messages and build prompt history."""
from __future__ import annotations

from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Contact, Conversation, MessageRole
from app.utils import get_logger

log = get_logger(__name__)

# How many past turns to feed back into the model.
HISTORY_WINDOW = 20


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
    """Return recent turns as OpenAI Responses-API input items, oldest first."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.contact_id == contact.id)
        .order_by(Conversation.id.desc())
        .limit(HISTORY_WINDOW)
    )
    rows = list(result.scalars().all())
    rows.reverse()
    items: List[dict] = []
    for r in rows:
        # System rows are internal notes; don't replay them as user/assistant.
        if r.role == MessageRole.system:
            continue
        items.append({"role": r.role.value, "content": r.content})
    return items
