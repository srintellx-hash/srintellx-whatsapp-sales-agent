"""Pydantic schemas: WhatsApp webhook payloads + internal DTOs."""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------
# WhatsApp Cloud API inbound webhook (subset we actually consume).
# Full schema: https://developers.facebook.com/docs/whatsapp/cloud-api/webhooks
# --------------------------------------------------------------------------
class WAText(BaseModel):
    body: str = ""


class WAMessage(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    id: str
    from_: str = Field(alias="from")  # 'from' is reserved in Python
    type: str
    timestamp: Optional[str] = None
    text: Optional[WAText] = None


class WAProfile(BaseModel):
    name: Optional[str] = None


class WAContact(BaseModel):
    model_config = ConfigDict(extra="ignore")
    wa_id: str
    profile: Optional[WAProfile] = None


class WAValue(BaseModel):
    model_config = ConfigDict(extra="ignore")
    messaging_product: Optional[str] = None
    contacts: Optional[List[WAContact]] = None
    messages: Optional[List[WAMessage]] = None


class WAChange(BaseModel):
    model_config = ConfigDict(extra="ignore")
    field: Optional[str] = None
    value: WAValue


class WAEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: Optional[str] = None
    changes: List[WAChange] = []


class WAWebhook(BaseModel):
    model_config = ConfigDict(extra="ignore")
    object: Optional[str] = None
    entry: List[WAEntry] = []


# We override WAMessage to map the reserved word `from`.
class InboundMessage(BaseModel):
    """Flattened, validated representation of one inbound text message."""
    wa_message_id: str
    sender: str          # MSISDN digits
    profile_name: Optional[str]
    text: str
    timestamp: Optional[str]


# --------------------------------------------------------------------------
# Internal DTOs (admin/inspection responses).
# --------------------------------------------------------------------------
class ContactOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    wa_phone: str
    wa_profile_name: Optional[str]
    doctor_name: Optional[str]
    clinic_name: Optional[str]
    specialty: Optional[str]
    city: Optional[str]
    calls_per_day: Optional[int]
    has_receptionist: Optional[bool]
    interest_level: str
    demo_requested: bool
    created_at: datetime


class BookingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    contact_id: int
    start_time: datetime
    end_time: datetime
    status: str
    meeting_link: Optional[str]
