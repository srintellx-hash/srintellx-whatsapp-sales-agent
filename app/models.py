"""ORM models.

Tables:
  contacts        - one row per WhatsApp sender (the lead)
  conversations   - every inbound/outbound message
  objections      - classified objections raised during chats
  demo_bookings   - confirmed demo slots synced to Google Calendar
"""
from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class MessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class InterestLevel(str, enum.Enum):
    cold = "cold"
    warm = "warm"
    hot = "hot"


class ObjectionType(str, enum.Enum):
    too_expensive = "too_expensive"
    already_have_receptionist = "already_have_receptionist"
    not_enough_calls = "not_enough_calls"
    whatsapp_only = "whatsapp_only"
    trust_concerns = "trust_concerns"
    ai_concerns = "ai_concerns"
    other = "other"


class BookingStatus(str, enum.Enum):
    confirmed = "confirmed"
    cancelled = "cancelled"


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Contact(TimestampMixin, Base):
    __tablename__ = "contacts"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Digits-only MSISDN (e.g. 919876543210). Unique per WhatsApp sender.
    wa_phone: Mapped[str] = mapped_column(String(32), unique=True, index=True, nullable=False)
    wa_profile_name: Mapped[str | None] = mapped_column(String(255))

    # Lead fields, captured progressively by the agent.
    doctor_name: Mapped[str | None] = mapped_column(String(255))
    clinic_name: Mapped[str | None] = mapped_column(String(255))
    specialty: Mapped[str | None] = mapped_column(String(255))
    city: Mapped[str | None] = mapped_column(String(255))
    calls_per_day: Mapped[int | None] = mapped_column(Integer)
    has_receptionist: Mapped[bool | None] = mapped_column(Boolean)
    interest_level: Mapped[InterestLevel] = mapped_column(
        Enum(InterestLevel, name="interest_level"),
        default=InterestLevel.cold,
        nullable=False,
    )
    demo_requested: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)

    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="contact", cascade="all, delete-orphan", lazy="selectin"
    )
    objections: Mapped[list["Objection"]] = relationship(
        back_populates="contact", cascade="all, delete-orphan", lazy="selectin"
    )
    bookings: Mapped[list["DemoBooking"]] = relationship(
        back_populates="contact", cascade="all, delete-orphan", lazy="selectin"
    )


class Conversation(TimestampMixin, Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    contact_id: Mapped[int] = mapped_column(
        ForeignKey("contacts.id", ondelete="CASCADE"), index=True, nullable=False
    )
    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole, name="message_role"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Meta message id, used for idempotency / de-duplication of webhooks.
    wa_message_id: Mapped[str | None] = mapped_column(String(128), index=True)

    contact: Mapped["Contact"] = relationship(back_populates="conversations")

    __table_args__ = (
        UniqueConstraint("wa_message_id", name="uq_conversation_wa_message_id"),
    )


class Objection(TimestampMixin, Base):
    __tablename__ = "objections"

    id: Mapped[int] = mapped_column(primary_key=True)
    contact_id: Mapped[int] = mapped_column(
        ForeignKey("contacts.id", ondelete="CASCADE"), index=True, nullable=False
    )
    objection_type: Mapped[ObjectionType] = mapped_column(
        Enum(ObjectionType, name="objection_type"), nullable=False
    )
    excerpt: Mapped[str | None] = mapped_column(Text)  # the user's words
    resolved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    contact: Mapped["Contact"] = relationship(back_populates="objections")


class DemoBooking(TimestampMixin, Base):
    __tablename__ = "demo_bookings"

    id: Mapped[int] = mapped_column(primary_key=True)
    contact_id: Mapped[int] = mapped_column(
        ForeignKey("contacts.id", ondelete="CASCADE"), index=True, nullable=False
    )
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[BookingStatus] = mapped_column(
        Enum(BookingStatus, name="booking_status"),
        default=BookingStatus.confirmed,
        nullable=False,
    )
    google_event_id: Mapped[str | None] = mapped_column(String(255), index=True)
    meeting_link: Mapped[str | None] = mapped_column(String(512))

    contact: Mapped["Contact"] = relationship(back_populates="bookings")

    __table_args__ = (
        UniqueConstraint("start_time", "status", name="uq_booking_slot_active"),
    )
