"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-18 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


message_role = sa.Enum("user", "assistant", "system", name="message_role")
interest_level = sa.Enum("cold", "warm", "hot", name="interest_level")
objection_type = sa.Enum(
    "too_expensive",
    "already_have_receptionist",
    "not_enough_calls",
    "whatsapp_only",
    "trust_concerns",
    "ai_concerns",
    "other",
    name="objection_type",
)
booking_status = sa.Enum("confirmed", "cancelled", name="booking_status")


def upgrade() -> None:
    bind = op.get_bind()
    message_role.create(bind, checkfirst=True)
    interest_level.create(bind, checkfirst=True)
    objection_type.create(bind, checkfirst=True)
    booking_status.create(bind, checkfirst=True)

    op.create_table(
        "contacts",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("wa_phone", sa.String(32), nullable=False),
        sa.Column("wa_profile_name", sa.String(255)),
        sa.Column("doctor_name", sa.String(255)),
        sa.Column("clinic_name", sa.String(255)),
        sa.Column("specialty", sa.String(255)),
        sa.Column("city", sa.String(255)),
        sa.Column("calls_per_day", sa.Integer),
        sa.Column("has_receptionist", sa.Boolean),
        sa.Column("interest_level", interest_level, nullable=False, server_default="cold"),
        sa.Column("demo_requested", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("notes", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_contacts_wa_phone", "contacts", ["wa_phone"], unique=True)

    op.create_table(
        "conversations",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("contact_id", sa.Integer, sa.ForeignKey("contacts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", message_role, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("wa_message_id", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("wa_message_id", name="uq_conversation_wa_message_id"),
    )
    op.create_index("ix_conversations_contact_id", "conversations", ["contact_id"])
    op.create_index("ix_conversations_wa_message_id", "conversations", ["wa_message_id"])

    op.create_table(
        "objections",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("contact_id", sa.Integer, sa.ForeignKey("contacts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("objection_type", objection_type, nullable=False),
        sa.Column("excerpt", sa.Text),
        sa.Column("resolved", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_objections_contact_id", "objections", ["contact_id"])

    op.create_table(
        "demo_bookings",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("contact_id", sa.Integer, sa.ForeignKey("contacts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", booking_status, nullable=False, server_default="confirmed"),
        sa.Column("google_event_id", sa.String(255)),
        sa.Column("meeting_link", sa.String(512)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("start_time", "status", name="uq_booking_slot_active"),
    )
    op.create_index("ix_demo_bookings_contact_id", "demo_bookings", ["contact_id"])
    op.create_index("ix_demo_bookings_google_event_id", "demo_bookings", ["google_event_id"])


def downgrade() -> None:
    op.drop_table("demo_bookings")
    op.drop_table("objections")
    op.drop_table("conversations")
    op.drop_index("ix_contacts_wa_phone", table_name="contacts")
    op.drop_table("contacts")
    bind = op.get_bind()
    booking_status.drop(bind, checkfirst=True)
    objection_type.drop(bind, checkfirst=True)
    interest_level.drop(bind, checkfirst=True)
    message_role.drop(bind, checkfirst=True)
