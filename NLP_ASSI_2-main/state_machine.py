from __future__ import annotations

import re
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional


class Intent(Enum):
    BOOK_APPOINTMENT = "BOOK_APPOINTMENT"
    CANCEL_APPOINTMENT = "CANCEL_APPOINTMENT"
    RESCHEDULE_APPOINTMENT = "RESCHEDULE_APPOINTMENT"
    CLINIC_INFO = "CLINIC_INFO"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class BookingState(Enum):
    COLLECT_NAME = "COLLECT_NAME"
    COLLECT_TYPE = "COLLECT_TYPE"
    COLLECT_DATETIME = "COLLECT_DATETIME"
    COLLECT_CONTACT = "COLLECT_CONTACT"
    CONFIRM = "CONFIRM"
    COMPLETE = "COMPLETE"


class CancelState(Enum):
    COLLECT_NAME = "COLLECT_NAME"
    CONFIRM_APPOINTMENT = "CONFIRM_APPOINTMENT"
    APPLY_POLICY = "APPLY_POLICY"
    CONFIRM_CANCEL = "CONFIRM_CANCEL"
    COMPLETE = "COMPLETE"


class RescheduleState(Enum):
    COLLECT_NAME = "COLLECT_NAME"
    CONFIRM_APPOINTMENT = "CONFIRM_APPOINTMENT"
    COLLECT_NEW_DATETIME = "COLLECT_NEW_DATETIME"
    CONFIRM = "CONFIRM"
    COMPLETE = "COMPLETE"


TASK_INTENTS = {
    Intent.BOOK_APPOINTMENT,
    Intent.CANCEL_APPOINTMENT,
    Intent.RESCHEDULE_APPOINTMENT,
}

INTENT_KEYWORDS = [
    (Intent.RESCHEDULE_APPOINTMENT, ["reschedule", "change my appointment", "move my appointment"]),
    (Intent.CANCEL_APPOINTMENT, ["cancel", "cancellation", "cancel my appointment"]),
    (Intent.BOOK_APPOINTMENT, ["book", "schedule", "make an appointment", "new appointment"]),
    (
        Intent.CLINIC_INFO,
        [
            "hours",
            "open",
            "close",
            "services",
            "service",
            "insurance",
            "pricing",
            "price",
            "cost",
            "emergency",
            "where",
            "location",
        ],
    ),
]

VALID_APPOINTMENT_TYPES = [
    "routine check-up",
    "cleaning",
    "filling",
    "teeth whitening",
    "orthodontic consultation",
    "root canal",
    "emergency care",
]

APPOINTMENT_TYPE_ALIASES = {
    "checkup": "routine check-up",
    "check-up": "routine check-up",
    "cleaning": "cleaning",
    "fillings": "filling",
    "whitening": "teeth whitening",
    "braces": "orthodontic consultation",
    "orthodontic": "orthodontic consultation",
    "root canal treatment": "root canal",
    "root canal": "root canal",
    "emergency": "emergency care",
}

CLINIC_HOURS = {
    0: (9, 0, 16, 30),
    1: (9, 0, 16, 30),
    2: (9, 0, 16, 30),
    3: (9, 0, 16, 30),
    4: (9, 0, 16, 30),
    5: (9, 0, 12, 30),
    6: None,
}

WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

POSITIVE_CONFIRMATIONS = {"yes", "yeah", "yep", "sure", "confirm", "correct", "go ahead", "please do"}
NEGATIVE_CONFIRMATIONS = {"no", "nope", "nah", "not now", "do not", "don't"}

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_RE = re.compile(r"^[+\d][\d\s().-]{6,}$")
TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.IGNORECASE)
NAME_PREFIX_RE = re.compile(r"^(my name is|this is|i am|it's|it is)\s+", re.IGNORECASE)


def detect_intent(user_message: str) -> Intent:
    message_lower = user_message.lower()
    for intent, keywords in INTENT_KEYWORDS:
        if any(keyword in message_lower for keyword in keywords):
            return intent
    return Intent.OUT_OF_SCOPE


def get_initial_state(intent: Intent):
    if intent == Intent.BOOK_APPOINTMENT:
        return BookingState.COLLECT_NAME
    if intent == Intent.CANCEL_APPOINTMENT:
        return CancelState.COLLECT_NAME
    if intent == Intent.RESCHEDULE_APPOINTMENT:
        return RescheduleState.COLLECT_NAME
    return None


def normalize_name(raw_name: str) -> str:
    cleaned = NAME_PREFIX_RE.sub("", raw_name.strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.title()


def normalize_appointment_type(user_input: str) -> Optional[str]:
    value = user_input.lower().strip()
    for appointment_type in VALID_APPOINTMENT_TYPES:
        if appointment_type in value:
            return appointment_type
    for alias, appointment_type in APPOINTMENT_TYPE_ALIASES.items():
        if alias in value:
            return appointment_type
    return None


def is_valid_contact_info(user_input: str) -> bool:
    value = user_input.strip()
    return bool(EMAIL_RE.match(value) or PHONE_RE.match(value))


def interpret_confirmation(user_input: str) -> Optional[bool]:
    lowered = user_input.lower().strip()
    if any(token in lowered for token in POSITIVE_CONFIRMATIONS):
        return True
    if any(token in lowered for token in NEGATIVE_CONFIRMATIONS):
        return False
    return None


def _parse_time_component(user_input: str) -> Optional[tuple[int, int]]:
    match = TIME_RE.search(user_input)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = (match.group(3) or "").lower()

    if meridiem == "pm" and hour != 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0

    if hour > 23 or minute > 59:
        return None
    return hour, minute


def parse_preferred_datetime(user_input: str, now: Optional[datetime] = None) -> Optional[datetime]:
    now = now or datetime.now()
    lowered = user_input.lower()
    parsed_time = _parse_time_component(lowered)
    if parsed_time is None:
        return None

    hour, minute = parsed_time

    if "today" in lowered:
        target_date = now.date()
    elif "tomorrow" in lowered:
        target_date = (now + timedelta(days=1)).date()
    else:
        target_weekday = None
        for day_name, weekday in WEEKDAYS.items():
            if day_name in lowered:
                target_weekday = weekday
                break
        if target_weekday is None:
            return None

        days_ahead = (target_weekday - now.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        target_date = (now + timedelta(days=days_ahead)).date()

    return datetime.combine(target_date, datetime.min.time()).replace(hour=hour, minute=minute)


def format_datetime_label(value: datetime) -> str:
    label = value.strftime("%A at %I:%M %p")
    return label.replace(" 0", " ")


def validate_booking_datetime(appointment_datetime: datetime, is_emergency: bool = False, now: Optional[datetime] = None) -> tuple[bool, str]:
    now = now or datetime.now()

    if appointment_datetime <= now:
        return False, "You cannot book an appointment in the past. Please choose a future date and time."

    if appointment_datetime > now + timedelta(days=60):
        return False, "Appointments can only be booked up to 60 days in advance. Please choose an earlier date."

    if appointment_datetime < now + timedelta(hours=2) and not is_emergency:
        return False, "Appointments must be booked at least 2 hours in advance unless it is emergency care."

    hours = CLINIC_HOURS[appointment_datetime.weekday()]
    if hours is None:
        return False, "The clinic is closed on Sundays. Please choose another day."

    open_h, open_m, close_h, close_m = hours
    opening_minutes = open_h * 60 + open_m
    closing_minutes = close_h * 60 + close_m
    requested_minutes = appointment_datetime.hour * 60 + appointment_datetime.minute

    if requested_minutes < opening_minutes or requested_minutes > closing_minutes:
        if appointment_datetime.weekday() == 5:
            return False, "On Saturdays we are open from 9:00 AM to 12:30 PM. Please choose a time within those hours."
        return False, "We are open Monday to Friday from 9:00 AM to 4:30 PM. Please choose a time within those hours."

    return True, ""


def is_within_cancellation_window(appointment_datetime: datetime, now: Optional[datetime] = None) -> bool:
    now = now or datetime.now()
    return appointment_datetime - now < timedelta(hours=24)
