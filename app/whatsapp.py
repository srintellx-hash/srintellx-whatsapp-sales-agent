"""Meta WhatsApp Business Cloud API integration."""
from __future__ import annotations

import hashlib
import hmac
from typing import List

import httpx

from app.config import settings
from app.schemas import InboundMessage, WAWebhook
from app.utils import get_logger, normalise_phone, truncate

log = get_logger(__name__)


class WhatsAppClient:
    def __init__(self) -> None:
        self._base = (
            f"{settings.whatsapp_api_base}/{settings.whatsapp_api_version}/"
            f"{settings.whatsapp_phone_number_id}/messages"
        )
        self._headers = {
            "Authorization": f"Bearer {settings.whatsapp_token}",
            "Content-Type": "application/json",
        }

    async def send_text(self, to: str, body: str) -> bool:
        """Send a plain text message. Returns True on success."""
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": normalise_phone(to),
            "type": "text",
            "text": {"preview_url": False, "body": truncate(body)},
        }
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(self._base, headers=self._headers, json=payload)
            if resp.status_code >= 400:
                log.error("WhatsApp send failed (%s): %s", resp.status_code, resp.text)
                return False
            log.info("Sent WhatsApp message to %s", to)
            return True
        except httpx.HTTPError as exc:  # network errors
            log.exception("WhatsApp send error: %s", exc)
            return False

    async def mark_read(self, message_id: str) -> None:
        payload = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(self._base, headers=self._headers, json=payload)
        except httpx.HTTPError:
            pass  # non-critical


def verify_signature(payload: bytes, signature_header: str | None) -> bool:
    """Validate Meta's X-Hub-Signature-256 header.

    Header format: 'sha256=<hexdigest>'. In development, if no app secret is
    configured we skip verification (and log a warning).
    """
    if not settings.whatsapp_app_secret:
        if settings.is_production:
            log.error("WHATSAPP_APP_SECRET not set in production - rejecting.")
            return False
        log.warning("Signature check skipped: no app secret configured (dev only).")
        return True
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(
        settings.whatsapp_app_secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()
    provided = signature_header.split("=", 1)[1]
    return hmac.compare_digest(expected, provided)


def parse_inbound_messages(webhook: WAWebhook) -> List[InboundMessage]:
    """Flatten a webhook envelope into a list of text messages we handle.

    Non-text message types (image, audio, status callbacks, etc.) are skipped.
    """
    out: List[InboundMessage] = []
    for entry in webhook.entry:
        for change in entry.changes:
            value = change.value
            # Build a name lookup from the contacts array.
            name_by_id = {}
            for c in value.contacts or []:
                name_by_id[c.wa_id] = c.profile.name if c.profile else None
            for msg in value.messages or []:
                if msg.type != "text" or not msg.text:
                    log.info("Skipping non-text message type=%s", msg.type)
                    continue
                out.append(
                    InboundMessage(
                        wa_message_id=msg.id,
                        sender=normalise_phone(msg.from_),
                        profile_name=name_by_id.get(msg.from_),
                        text=msg.text.body,
                        timestamp=msg.timestamp,
                    )
                )
    return out


whatsapp_client = WhatsAppClient()
