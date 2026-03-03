from enum import Enum
from datetime import datetime, timedelta

# ── Intents ────────────────────────────────────────────────────────────────────
class Intent(Enum):
    BOOK_APPOINTMENT       = "BOOK_APPOINTMENT"
    CANCEL_APPOINTMENT     = "CANCEL_APPOINTMENT"
    RESCHEDULE_APPOINTMENT = "RESCHEDULE_APPOINTMENT"
    CLINIC_INFO            = "CLINIC_INFO"
    OUT_OF_SCOPE           = "OUT_OF_SCOPE"

# ── States ─────────────────────────────────────────────────────────────────────
class BookingState(Enum):
    COLLECT_NAME     = "COLLECT_NAME"
    COLLECT_TYPE     = "COLLECT_TYPE"
    COLLECT_DATETIME = "COLLECT_DATETIME"
    COLLECT_CONTACT  = "COLLECT_CONTACT"
    CONFIRM          = "CONFIRM"
    COMPLETE         = "COMPLETE"

class CancelState(Enum):
    COLLECT_NAME        = "COLLECT_NAME"
    CONFIRM_APPOINTMENT = "CONFIRM_APPOINTMENT"
    APPLY_POLICY        = "APPLY_POLICY"
    CONFIRM_CANCEL      = "CONFIRM_CANCEL"
    COMPLETE            = "COMPLETE"

class RescheduleState(Enum):
    COLLECT_NAME         = "COLLECT_NAME"
    CONFIRM_APPOINTMENT  = "CONFIRM_APPOINTMENT"
    COLLECT_NEW_DATETIME = "COLLECT_NEW_DATETIME"
    CONFIRM              = "CONFIRM"
    COMPLETE             = "COMPLETE"

# ── Intent Detection ───────────────────────────────────────────────────────────
INTENT_KEYWORDS = {
    Intent.BOOK_APPOINTMENT:       ["book", "schedule", "make an appointment", "new appointment"],
    Intent.CANCEL_APPOINTMENT:     ["cancel", "cancellation", "cancel my appointment"],
    Intent.RESCHEDULE_APPOINTMENT: ["reschedule", "change my appointment", "move my appointment"],
    Intent.CLINIC_INFO:            ["hours", "location", "services", "insurance", "emergency", "where"],
}

def detect_intent(user_message: str) -> Intent:
    message_lower = user_message.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword in message_lower for keyword in keywords):
            return intent
    return Intent.OUT_OF_SCOPE

def get_initial_state(intent: Intent):
    if intent == Intent.BOOK_APPOINTMENT:
        return BookingState.COLLECT_NAME
    elif intent == Intent.CANCEL_APPOINTMENT:
        return CancelState.COLLECT_NAME
    elif intent == Intent.RESCHEDULE_APPOINTMENT:
        return RescheduleState.COLLECT_NAME
    return None

# ── Validation Logic (Python controls this, NOT the LLM) ──────────────────────

VALID_APPOINTMENT_TYPES = [
    "routine check-up", "cleaning", "filling", "teeth whitening",
    "orthodontic consultation", "root canal", "emergency"
]

CLINIC_HOURS = {
    "monday":    (9, 0, 16, 30),
    "tuesday":   (9, 0, 16, 30),
    "wednesday": (9, 0, 16, 30),
    "thursday":  (9, 0, 16, 30),
    "friday":    (9, 0, 16, 30),
    "saturday":  (9, 0, 12, 30),
    "sunday":    None  # closed
}

def is_valid_appointment_type(user_input: str) -> bool:
    user_input = user_input.lower()
    return any(t in user_input for t in VALID_APPOINTMENT_TYPES)

def is_valid_time(day: str, hour: int, minute: int) -> tuple[bool, str]:
    """Returns (is_valid, error_message)"""
    day = day.lower()
    if day not in CLINIC_HOURS:
        return False, "I didn't recognize that day. Please provide a valid day of the week."
    
    hours = CLINIC_HOURS[day]
    if hours is None:
        return False, "The clinic is closed on Sundays. Please choose another day."
    
    open_h, open_m, close_h, close_m = hours
    requested = hour * 60 + minute
    opening   = open_h * 60 + open_m
    closing   = close_h * 60 + close_m

    if requested < opening or requested > closing:
        if day == "saturday":
            return False, "On Saturdays we are open 9:00AM to 12:30PM. Please choose a time within those hours."
        return False, "We are open Monday to Friday 9:00AM to 4:30PM. Please choose a time within those hours."
    
    return True, ""

def is_within_booking_window(days_from_now: int, is_emergency: bool = False) -> tuple[bool, str]:
    """Returns (is_valid, error_message)"""
    if days_from_now < 0:
        return False, "You cannot book an appointment in the past. Please choose a future date."
    if days_from_now == 0 and not is_emergency:
        return False, "Appointments must be booked at least 2 hours in advance. Emergency care is exempt."
    if days_from_now > 60:
        return False, "Appointments can only be booked up to 60 days in advance. Please choose an earlier date."
    return True, ""

def is_within_cancellation_window(appointment_datetime: datetime) -> bool:
    """Returns True if appointment is within 24 hours (late cancellation)."""
    return appointment_datetime - datetime.now() < timedelta(hours=24)