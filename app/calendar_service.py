"""Google Calendar integration for demo scheduling.

Uses a service account. Availability is constrained both by the master
prompt's demo windows AND by real free/busy data from the calendar, so we
never double-book.

Demo windows (Asia/Kolkata):
  Mon-Fri : 20:30 - 22:00
  Sat-Sun : 10:00 - 22:00
Duration: settings.demo_duration_minutes (default 30).
"""
from __future__ import annotations

import json
from datetime import datetime, time, timedelta
from typing import List, Optional

from app.config import settings
from app.utils import get_logger, now_tz, tz

log = get_logger(__name__)

# Per-weekday demo windows: weekday() -> list of (start, end) time tuples.
_DEMO_WINDOWS = {
    0: [(time(20, 30), time(22, 0))],   # Monday
    1: [(time(20, 30), time(22, 0))],
    2: [(time(20, 30), time(22, 0))],
    3: [(time(20, 30), time(22, 0))],
    4: [(time(20, 30), time(22, 0))],   # Friday
    5: [(time(10, 0), time(22, 0))],    # Saturday
    6: [(time(10, 0), time(22, 0))],    # Sunday
}


class CalendarService:
    """Wraps the Google Calendar API. Degrades gracefully if not configured."""

    def __init__(self) -> None:
        self._service = None
        self._enabled = False
        self._init_service()

    def _load_credentials(self):
        from google.oauth2 import service_account  # lazy import

        scopes = ["https://www.googleapis.com/auth/calendar"]
        if settings.google_service_account_json.strip():
            info = json.loads(settings.google_service_account_json)
            return service_account.Credentials.from_service_account_info(info, scopes=scopes)
        if settings.google_service_account_file.strip():
            return service_account.Credentials.from_service_account_file(
                settings.google_service_account_file, scopes=scopes
            )
        return None

    def _init_service(self) -> None:
        try:
            creds = self._load_credentials()
            if creds is None:
                log.warning("Google Calendar not configured; booking will be simulated.")
                return
            from googleapiclient.discovery import build  # lazy import

            self._service = build("calendar", "v3", credentials=creds, cache_discovery=False)
            self._enabled = True
            log.info("Google Calendar service initialised.")
        except Exception as exc:  # pragma: no cover - config/runtime dependent
            log.exception("Calendar init failed; falling back to simulation: %s", exc)
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Slot generation
    # ------------------------------------------------------------------
    def _candidate_slots(self, days_ahead: int = 7) -> List[datetime]:
        """All demo-window start times in the next `days_ahead` days (future only)."""
        duration = timedelta(minutes=settings.demo_duration_minutes)
        start_ref = now_tz()
        slots: List[datetime] = []
        for d in range(days_ahead + 1):
            day = (start_ref + timedelta(days=d)).date()
            for win_start, win_end in _DEMO_WINDOWS.get(datetime(day.year, day.month, day.day).weekday(), []):
                cursor = datetime.combine(day, win_start, tzinfo=tz())
                window_end = datetime.combine(day, win_end, tzinfo=tz())
                while cursor + duration <= window_end:
                    if cursor > start_ref + timedelta(minutes=30):  # need lead time
                        slots.append(cursor)
                    cursor += duration
        return slots

    async def _busy_intervals(self, start: datetime, end: datetime):
        if not self._enabled:
            return []
        body = {
            "timeMin": start.isoformat(),
            "timeMax": end.isoformat(),
            "timeZone": settings.timezone,
            "items": [{"id": settings.google_calendar_id}],
        }
        try:
            resp = self._service.freebusy().query(body=body).execute()
            cal = resp["calendars"][settings.google_calendar_id]
            return [
                (
                    datetime.fromisoformat(b["start"].replace("Z", "+00:00")),
                    datetime.fromisoformat(b["end"].replace("Z", "+00:00")),
                )
                for b in cal.get("busy", [])
            ]
        except Exception as exc:  # pragma: no cover
            log.exception("freebusy query failed: %s", exc)
            return []

    async def get_available_slots(self, limit: int = 4) -> List[datetime]:
        """Return up to `limit` bookable start times, filtered by real busy data."""
        candidates = self._candidate_slots()
        if not candidates:
            return []
        busy = await self._busy_intervals(candidates[0], candidates[-1] + timedelta(hours=1))
        duration = timedelta(minutes=settings.demo_duration_minutes)
        free: List[datetime] = []
        for slot in candidates:
            slot_end = slot + duration
            overlap = any(b_start < slot_end and slot < b_end for b_start, b_end in busy)
            if not overlap:
                free.append(slot)
            if len(free) >= limit:
                break
        return free

    async def is_slot_free(self, start: datetime) -> bool:
        duration = timedelta(minutes=settings.demo_duration_minutes)
        busy = await self._busy_intervals(start - timedelta(minutes=1), start + duration + timedelta(minutes=1))
        slot_end = start + duration
        return not any(b_start < slot_end and start < b_end for b_start, b_end in busy)

    async def create_event(
        self, start: datetime, summary: str, description: str, attendee_email: Optional[str] = None
    ) -> dict:
        """Create the calendar event. Returns {event_id, meeting_link} or simulated values."""
        duration = timedelta(minutes=settings.demo_duration_minutes)
        end = start + duration
        if not self._enabled:
            log.warning("Calendar disabled - returning simulated event.")
            return {"event_id": f"sim-{int(start.timestamp())}", "meeting_link": None}

        event_body = {
            "summary": summary,
            "description": description,
            "start": {"dateTime": start.isoformat(), "timeZone": settings.timezone},
            "end": {"dateTime": end.isoformat(), "timeZone": settings.timezone},
        }
        attendees = []
        if settings.demo_organizer_email:
            attendees.append({"email": settings.demo_organizer_email})
        if attendee_email:
            attendees.append({"email": attendee_email})
        if attendees:
            event_body["attendees"] = attendees

        created = (
            self._service.events()
            .insert(calendarId=settings.google_calendar_id, body=event_body, sendUpdates="all")
            .execute()
        )
        return {
            "event_id": created.get("id"),
            "meeting_link": created.get("htmlLink"),
        }


calendar_service = CalendarService()
