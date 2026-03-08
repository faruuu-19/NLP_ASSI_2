from __future__ import annotations

import asyncio
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import AsyncGenerator, Dict, List, Optional

from llm_engine import LLMUnavailableError, OllamaLLM
from state_machine import (
    BookingState,
    CancelState,
    Intent,
    RescheduleState,
    TASK_INTENTS,
    detect_intent,
    format_datetime_label,
    get_initial_state,
    interpret_confirmation,
    is_valid_contact_info,
    is_within_cancellation_window,
    normalize_appointment_type,
    normalize_name,
    parse_preferred_datetime,
    validate_booking_datetime,
)
from system_prompt import build_general_prompt


def _parse_state(intent: Optional[Intent], state_value: Optional[str]):
    if not intent or not state_value:
        return None
    try:
        if intent == Intent.BOOK_APPOINTMENT:
            return BookingState(state_value)
        if intent == Intent.CANCEL_APPOINTMENT:
            return CancelState(state_value)
        if intent == Intent.RESCHEDULE_APPOINTMENT:
            return RescheduleState(state_value)
    except ValueError:
        return None
    return None


@dataclass
class SessionMemory:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    window_size: int = 12
    history: deque = field(default_factory=lambda: deque(maxlen=12))
    intent: Optional[Intent] = None
    state: Optional[object] = None
    entities: Dict[str, object] = field(default_factory=dict)
    policy_flags: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Optional[dict]) -> "SessionMemory":
        payload = payload or {}
        window_size = int(payload.get("window_size", 12) or 12)
        history = deque(payload.get("history", [])[-window_size:], maxlen=window_size)
        intent_value = payload.get("intent")
        intent = Intent(intent_value) if intent_value else None
        state = _parse_state(intent, payload.get("state"))
        return cls(
            session_id=payload.get("session_id") or str(uuid.uuid4()),
            window_size=window_size,
            history=history,
            intent=intent,
            state=state,
            entities=dict(payload.get("entities", {})),
            policy_flags=dict(payload.get("policy_flags", {})),
        )

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "window_size": self.window_size,
            "history": list(self.history),
            "intent": self.intent.value if self.intent else None,
            "state": self.state.value if self.state else None,
            "entities": self.entities,
            "policy_flags": self.policy_flags,
        }

    def add_turn(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def reset(self, new_session_id: bool = False) -> None:
        self.history.clear()
        self.intent = None
        self.state = None
        self.entities.clear()
        self.policy_flags.clear()
        if new_session_id:
            self.session_id = str(uuid.uuid4())

    def set_entity(self, key: str, value) -> None:
        self.entities[key] = value

    def get_entity(self, key: str, default=None):
        return self.entities.get(key, default)

    def snapshot(self) -> str:
        active_state = self.state.value if self.state else "idle"
        active_intent = self.intent.value if self.intent else "none"
        entity_pairs = ", ".join(f"{key}={value}" for key, value in self.entities.items()) or "none"
        return f"intent={active_intent}; state={active_state}; entities={entity_pairs}"


APPOINTMENT_DURATION_MINUTES = {
    "routine check-up": 30,
    "cleaning": 45,
    "filling": 60,
    "teeth whitening": 60,
    "orthodontic consultation": 45,
    "root canal": 90,
    "emergency care": 60,
}


def _get_appointments(session: SessionMemory) -> List[dict]:
    appointments = session.entities.setdefault("appointments", [])
    if not isinstance(appointments, list):
        appointments = []
        session.entities["appointments"] = appointments
    return appointments


def _active_appointments(session: SessionMemory) -> List[dict]:
    return [appointment for appointment in _get_appointments(session) if appointment.get("status") == "booked"]


def _appointments_for_name(session: SessionMemory, name: str) -> List[dict]:
    normalized_name = normalize_name(name)
    return [
        appointment
        for appointment in _active_appointments(session)
        if normalize_name(str(appointment.get("name", ""))) == normalized_name
    ]


def _appointment_datetime(appointment: dict) -> datetime:
    return datetime.fromisoformat(str(appointment["appointment_datetime"]))


def _appointment_duration_minutes(appointment_type: str) -> int:
    return APPOINTMENT_DURATION_MINUTES.get(appointment_type, 60)


def _appointment_label(appointment: dict) -> str:
    return f"{appointment['appointment_type']} on {format_datetime_label(_appointment_datetime(appointment))}"


def _find_appointment_by_id(session: SessionMemory, appointment_id: str) -> Optional[dict]:
    for appointment in _get_appointments(session):
        if appointment.get("id") == appointment_id:
            return appointment
    return None


def _find_matching_appointment(candidates: List[dict], user_message: str) -> Optional[dict]:
    lowered = user_message.lower()
    parsed_datetime = parse_preferred_datetime(user_message)
    matches = []

    for appointment in candidates:
        appointment_time = _appointment_datetime(appointment)
        if parsed_datetime and appointment_time == parsed_datetime:
            matches.append(appointment)
            continue
        if appointment["appointment_type"] in lowered:
            matches.append(appointment)
            continue
        if appointment_time.strftime("%A").lower() in lowered:
            matches.append(appointment)

    if len(matches) == 1:
        return matches[0]
    return None


def _has_conflict(session: SessionMemory, requested_datetime: datetime, appointment_type: str, exclude_appointment_id: Optional[str] = None) -> bool:
    requested_end = requested_datetime + timedelta(minutes=_appointment_duration_minutes(appointment_type))

    for appointment in _active_appointments(session):
        if appointment.get("id") == exclude_appointment_id:
            continue

        current_start = _appointment_datetime(appointment)
        current_end = current_start + timedelta(minutes=_appointment_duration_minutes(appointment["appointment_type"]))

        if requested_datetime < current_end and requested_end > current_start:
            return True

    return False


@dataclass
class AssistantReply:
    text: str
    intent: str
    state: str
    session: dict


class ConversationService:
    def __init__(self, llm: Optional[OllamaLLM] = None):
        self.llm = llm or OllamaLLM()

    async def reply(self, session: SessionMemory, user_message: str) -> AssistantReply:
        reply_text = await self._generate_reply(session, user_message)
        session.add_turn("user", user_message)
        session.add_turn("assistant", reply_text)
        return AssistantReply(
            text=reply_text,
            intent=session.intent.value if session.intent else Intent.OUT_OF_SCOPE.value,
            state=session.state.value if session.state else "IDLE",
            session=session.to_dict(),
        )

    async def stream_reply(self, session: SessionMemory, user_message: str) -> AsyncGenerator[str, None]:
        reply_text = await self._generate_reply(session, user_message)
        for token in self._chunk_text(reply_text):
            yield token
            await asyncio.sleep(0)
        session.add_turn("user", user_message)
        session.add_turn("assistant", reply_text)

    async def _generate_reply(self, session: SessionMemory, user_message: str) -> str:
        detected_intent = detect_intent(user_message)

        if session.state in {BookingState.COMPLETE, CancelState.COMPLETE, RescheduleState.COMPLETE} and detected_intent in TASK_INTENTS:
            self._start_new_flow(session, detected_intent)
            return self._initial_prompt_for_intent(session.intent)

        if self._should_switch_intent(session, user_message):
            self._start_new_flow(session, detected_intent)
            return self._initial_prompt_for_intent(session.intent)

        if session.intent is None:
            if detected_intent in TASK_INTENTS:
                self._start_new_flow(session, detected_intent)
                return self._initial_prompt_for_intent(session.intent)
            return await self._handle_general(session, user_message, detected_intent)

        if session.intent == Intent.BOOK_APPOINTMENT:
            return self._handle_booking(session, user_message)
        if session.intent == Intent.CANCEL_APPOINTMENT:
            return self._handle_cancel(session, user_message)
        if session.intent == Intent.RESCHEDULE_APPOINTMENT:
            return self._handle_reschedule(session, user_message)
        return await self._handle_general(session, user_message, detected_intent)

    def _should_switch_intent(self, session: SessionMemory, user_message: str) -> bool:
        detected_intent = detect_intent(user_message)
        confirmation = interpret_confirmation(user_message)
        if session.intent is None or detected_intent not in TASK_INTENTS:
            return False
        if session.state in {BookingState.COMPLETE, CancelState.COMPLETE, RescheduleState.COMPLETE}:
            return detected_intent != session.intent
        return detected_intent != session.intent and confirmation is None

    def _start_new_flow(self, session: SessionMemory, intent: Intent) -> None:
        persistent_appointments = list(_get_appointments(session))
        session.intent = intent
        session.state = get_initial_state(intent)
        session.entities = {"appointments": persistent_appointments}
        session.policy_flags = {}

    def _initial_prompt_for_intent(self, intent: Optional[Intent]) -> str:
        if intent == Intent.BOOK_APPOINTMENT:
            return "I'd be glad to help with that. May I have your full name, please?"
        if intent == Intent.CANCEL_APPOINTMENT:
            return "I can help cancel your appointment. Could I have the name the appointment is under?"
        if intent == Intent.RESCHEDULE_APPOINTMENT:
            return "I can help reschedule that. Could I have the name the appointment is under?"
        return "How can I help you with dental scheduling today?"

    def _handle_booking(self, session: SessionMemory, user_message: str) -> str:
        if session.state == BookingState.COLLECT_NAME:
            session.set_entity("name", normalize_name(user_message))
            session.state = BookingState.COLLECT_TYPE
            first_name = session.get_entity("name").split()[0]
            return (
                f"Thanks, {first_name}. What type of appointment do you need: routine check-up, cleaning, "
                "filling, teeth whitening, orthodontic consultation, root canal, or emergency care?"
            )

        if session.state == BookingState.COLLECT_TYPE:
            appointment_type = normalize_appointment_type(user_message)
            if appointment_type is None:
                return "I didn't recognize that appointment type. Please choose one of the listed dental services."
            session.set_entity("appointment_type", appointment_type)
            session.state = BookingState.COLLECT_DATETIME
            return "What date and time would you prefer? For example, Thursday at 10 AM."

        if session.state == BookingState.COLLECT_DATETIME:
            appointment_datetime = parse_preferred_datetime(user_message)
            if appointment_datetime is None:
                return "I didn't catch the requested slot. Please share the day and time, for example Thursday at 10 AM."

            is_emergency = session.get_entity("appointment_type") == "emergency care"
            valid, error_message = validate_booking_datetime(appointment_datetime, is_emergency=is_emergency)
            if not valid:
                return error_message

            if _has_conflict(session, appointment_datetime, session.get_entity("appointment_type")):
                return "That time is unavailable because it overlaps with another appointment already saved in this session. Please choose a different time."

            session.set_entity("appointment_datetime", appointment_datetime.isoformat())
            session.state = BookingState.COLLECT_CONTACT
            return "Could I get your email address or phone number for the confirmation?"

        if session.state == BookingState.COLLECT_CONTACT:
            if not is_valid_contact_info(user_message):
                return "Please share a valid email address or phone number so I can send the confirmation."
            session.set_entity("contact", user_message.strip())
            session.state = BookingState.CONFIRM
            appointment_datetime = datetime.fromisoformat(session.get_entity("appointment_datetime"))
            return (
                f"To confirm: {session.get_entity('appointment_type')} for {session.get_entity('name')} on "
                f"{format_datetime_label(appointment_datetime)}, with confirmation sent to {session.get_entity('contact')}. "
                "Should I book it?"
            )

        if session.state == BookingState.CONFIRM:
            confirmation = interpret_confirmation(user_message)
            if confirmation is True:
                appointment_datetime = datetime.fromisoformat(session.get_entity("appointment_datetime"))
                _get_appointments(session).append(
                    {
                        "id": str(uuid.uuid4()),
                        "name": session.get_entity("name"),
                        "appointment_type": session.get_entity("appointment_type"),
                        "appointment_datetime": appointment_datetime.isoformat(),
                        "contact": session.get_entity("contact"),
                        "status": "booked",
                    }
                )
                session.state = BookingState.COMPLETE
                return (
                    f"Your appointment is confirmed for {format_datetime_label(appointment_datetime)}. "
                    "Please arrive 10 minutes early. Is there anything else I can help you with?"
                )
            if confirmation is False:
                session.state = BookingState.COLLECT_DATETIME
                return "No problem. What new date and time would you prefer?"
            return "Please reply with yes to confirm the booking or no to change the time."

        session.intent = None
        session.state = None
        return "If you need anything else, I can help with another dental appointment request."

    def _handle_cancel(self, session: SessionMemory, user_message: str) -> str:
        if session.state == CancelState.COLLECT_NAME:
            patient_name = normalize_name(user_message)
            matches = _appointments_for_name(session, patient_name)
            session.set_entity("name", patient_name)

            if not matches:
                session.state = CancelState.COMPLETE
                return "I couldn't find an active appointment under that name in this session. Would you like to book a new appointment instead?"

            if len(matches) == 1:
                selected = matches[0]
                session.set_entity("selected_appointment_id", selected["id"])
                session.state = CancelState.CONFIRM_APPOINTMENT
                return f"I found your {_appointment_label(selected)}. Is that the appointment you'd like to cancel?"

            session.set_entity("candidate_appointment_ids", [appointment["id"] for appointment in matches])
            session.state = CancelState.CONFIRM_APPOINTMENT
            options = "; ".join(_appointment_label(appointment) for appointment in matches[:3])
            return f"I found multiple appointments for {patient_name}: {options}. Please tell me which one you'd like to cancel."

        if session.state == CancelState.CONFIRM_APPOINTMENT:
            candidate_ids = session.get_entity("candidate_appointment_ids", [])
            if candidate_ids and not session.get_entity("selected_appointment_id"):
                candidates = [
                    appointment
                    for appointment_id in candidate_ids
                    for appointment in [_find_appointment_by_id(session, appointment_id)]
                    if appointment and appointment.get("status") == "booked"
                ]
                selected = _find_matching_appointment(candidates, user_message)
                if selected is not None:
                    session.set_entity("selected_appointment_id", selected["id"])
                    session.entities.pop("candidate_appointment_ids", None)
                    return f"I found your {_appointment_label(selected)}. Is that the appointment you'd like to cancel?"
                return "Please tell me the appointment day, time, or type so I can identify which booking to cancel."

            selected = _find_appointment_by_id(session, session.get_entity("selected_appointment_id"))
            if selected is None or selected.get("status") != "booked":
                session.state = CancelState.COMPLETE
                return "I couldn't find an active appointment to cancel. Would you like help booking a new one?"

            confirmation = interpret_confirmation(user_message)
            if confirmation is True:
                appointment_datetime = _appointment_datetime(selected)
                if is_within_cancellation_window(appointment_datetime):
                    session.policy_flags["late_cancellation"] = True
                    session.state = CancelState.APPLY_POLICY
                    return "This appointment is within 24 hours, so a $25 late cancellation fee may apply. Would you still like to proceed?"
                selected["status"] = "cancelled"
                session.state = CancelState.CONFIRM_CANCEL
                return f"Your {_appointment_label(selected)} has been cancelled. Would you like to reschedule for a future date?"
            if confirmation is False:
                session.entities.pop("selected_appointment_id", None)
                return "Okay. Please tell me the correct appointment details or the name on the booking."
            return "Please reply yes if that is the appointment to cancel, or no if it is not."

        if session.state == CancelState.APPLY_POLICY:
            selected = _find_appointment_by_id(session, session.get_entity("selected_appointment_id"))
            if selected is None or selected.get("status") != "booked":
                session.state = CancelState.COMPLETE
                return "I couldn't find an active appointment to cancel. Would you like help booking a new one?"
            confirmation = interpret_confirmation(user_message)
            if confirmation is True:
                selected["status"] = "cancelled"
                session.state = CancelState.CONFIRM_CANCEL
                return f"Understood. Your {_appointment_label(selected)} has been cancelled. Would you like to reschedule for a future date?"
            if confirmation is False:
                session.state = CancelState.COMPLETE
                return "No problem. I have left the appointment in place. Is there anything else I can help you with?"
            return "Please reply yes to continue with the cancellation or no to keep the appointment."

        if session.state == CancelState.CONFIRM_CANCEL:
            confirmation = interpret_confirmation(user_message)
            if confirmation is True:
                selected = _find_appointment_by_id(session, session.get_entity("selected_appointment_id"))
                if selected is not None:
                    session.set_entity("name", selected["name"])
                    session.set_entity("appointment_type", selected["appointment_type"])
                    session.set_entity("contact", selected["contact"])
                session.entities.pop("candidate_appointment_ids", None)
                session.entities.pop("selected_appointment_id", None)
                session.intent = Intent.BOOK_APPOINTMENT
                session.state = BookingState.COLLECT_DATETIME
                return "Sure. What new date and time would you prefer?"
            session.state = CancelState.COMPLETE
            session.entities.pop("candidate_appointment_ids", None)
            session.entities.pop("selected_appointment_id", None)
            return "Understood. Reach out anytime if you'd like to book again."

        session.intent = None
        session.state = None
        return "If you need anything else, I can help with another dental appointment request."

    def _handle_reschedule(self, session: SessionMemory, user_message: str) -> str:
        if session.state == RescheduleState.COLLECT_NAME:
            patient_name = normalize_name(user_message)
            matches = _appointments_for_name(session, patient_name)
            session.set_entity("name", patient_name)

            if not matches:
                session.state = RescheduleState.COMPLETE
                return "I couldn't find an active appointment under that name in this session. Would you like to book a new one?"

            if len(matches) == 1:
                selected = matches[0]
                session.set_entity("selected_appointment_id", selected["id"])
                session.state = RescheduleState.CONFIRM_APPOINTMENT
                return f"I found your {_appointment_label(selected)}. Is that the one you'd like to reschedule?"

            session.set_entity("candidate_appointment_ids", [appointment["id"] for appointment in matches])
            session.state = RescheduleState.CONFIRM_APPOINTMENT
            options = "; ".join(_appointment_label(appointment) for appointment in matches[:3])
            return f"I found multiple appointments for {patient_name}: {options}. Please tell me which one you'd like to reschedule."

        if session.state == RescheduleState.CONFIRM_APPOINTMENT:
            candidate_ids = session.get_entity("candidate_appointment_ids", [])
            if candidate_ids and not session.get_entity("selected_appointment_id"):
                candidates = [
                    appointment
                    for appointment_id in candidate_ids
                    for appointment in [_find_appointment_by_id(session, appointment_id)]
                    if appointment and appointment.get("status") == "booked"
                ]
                selected = _find_matching_appointment(candidates, user_message)
                if selected is not None:
                    session.set_entity("selected_appointment_id", selected["id"])
                    session.entities.pop("candidate_appointment_ids", None)
                    return f"I found your {_appointment_label(selected)}. Is that the one you'd like to reschedule?"
                return "Please tell me the appointment day, time, or type so I can identify which booking to reschedule."

            confirmation = interpret_confirmation(user_message)
            if confirmation is True:
                selected = _find_appointment_by_id(session, session.get_entity("selected_appointment_id"))
                if selected is None or selected.get("status") != "booked":
                    session.state = RescheduleState.COMPLETE
                    return "I couldn't find an active appointment to reschedule. Would you like to book a new one instead?"
                session.state = RescheduleState.COLLECT_NEW_DATETIME
                return "What new date and time would you prefer?"
            if confirmation is False:
                return "Please tell me which appointment you'd like to reschedule."
            return "Please reply yes if that's the appointment, or no if it is not."

        if session.state == RescheduleState.COLLECT_NEW_DATETIME:
            new_datetime = parse_preferred_datetime(user_message)
            if new_datetime is None:
                return "I didn't catch the new slot. Please share the day and time, for example Monday at 2 PM."
            selected = _find_appointment_by_id(session, session.get_entity("selected_appointment_id"))
            if selected is None or selected.get("status") != "booked":
                session.state = RescheduleState.COMPLETE
                return "I couldn't find an active appointment to reschedule. Would you like to book a new one instead?"

            valid, error_message = validate_booking_datetime(
                new_datetime,
                is_emergency=selected["appointment_type"] == "emergency care",
            )
            if not valid:
                return error_message

            if _has_conflict(
                session,
                new_datetime,
                selected["appointment_type"],
                exclude_appointment_id=selected["id"],
            ):
                return "That time is unavailable because it overlaps with another appointment already saved in this session. Please choose a different time."
            session.set_entity("new_appointment_datetime", new_datetime.isoformat())
            session.state = RescheduleState.CONFIRM
            return f"To confirm, should I move your {_appointment_label(selected)} to {format_datetime_label(new_datetime)}?"

        if session.state == RescheduleState.CONFIRM:
            confirmation = interpret_confirmation(user_message)
            if confirmation is True:
                selected = _find_appointment_by_id(session, session.get_entity("selected_appointment_id"))
                if selected is None or selected.get("status") != "booked":
                    session.state = RescheduleState.COMPLETE
                    return "I couldn't find an active appointment to reschedule. Would you like to book a new one instead?"
                new_datetime = datetime.fromisoformat(session.get_entity("new_appointment_datetime"))
                selected["appointment_datetime"] = new_datetime.isoformat()
                session.state = RescheduleState.COMPLETE
                session.entities.pop("candidate_appointment_ids", None)
                session.entities.pop("selected_appointment_id", None)
                return f"Your appointment has been rescheduled to {format_datetime_label(new_datetime)}. A confirmation will be sent to your contact on file."
            if confirmation is False:
                session.state = RescheduleState.COLLECT_NEW_DATETIME
                return "No problem. What new date and time would you prefer instead?"
            return "Please reply yes to confirm the new slot or no to pick another time."

        session.intent = None
        session.state = None
        return "If you need anything else, I can help with another dental appointment request."

    async def _handle_general(self, session: SessionMemory, user_message: str, detected_intent: Intent) -> str:
        session.intent = None
        session.state = None
        fallback = self._fallback_general_response(user_message, detected_intent)
        try:
            response = await self.llm.chat(build_general_prompt(session.snapshot()), list(session.history), user_message)
            return response or fallback
        except LLMUnavailableError:
            return fallback

    def _fallback_general_response(self, user_message: str, detected_intent: Intent) -> str:
        lowered = user_message.lower()
        if detected_intent == Intent.CLINIC_INFO:
            if any(token in lowered for token in ["hours", "open", "close"]):
                return "SmileCare Dental Clinic is open Monday to Friday from 9:00 AM to 4:30 PM and Saturday from 9:00 AM to 12:30 PM. Would you like to book an appointment?"
            if any(token in lowered for token in ["service", "services"]):
                return "We offer routine check-ups, cleanings, fillings, teeth whitening, orthodontic consultations, root canal treatment, and emergency dental care. Would you like to book an appointment?"
            if any(token in lowered for token in ["insurance", "price", "pricing", "cost"]):
                return "For insurance and pricing details, please contact the clinic directly. Would you like to book an appointment?"
            if "emergency" in lowered:
                return "Emergency dental care and same-day appointments may be available during clinic hours. Would you like to book an appointment?"
            return "I can help with clinic hours, services, emergency dental care, and scheduling. Would you like to book an appointment?"
        return "I can only assist with dental appointment scheduling, rescheduling, cancellations, and clinic information. How can I help you with that today?"

    @staticmethod
    def _chunk_text(reply_text: str) -> List[str]:
        words = reply_text.split()
        if not words:
            return [reply_text]
        chunks = []
        current = []
        for word in words:
            current.append(word)
            if len(current) == 6:
                chunks.append(" ".join(current) + " ")
                current = []
        if current:
            chunks.append(" ".join(current) + (" " if len(chunks) else ""))
        return chunks


def new_session_memory(new_session_id: bool = True) -> SessionMemory:
    session = SessionMemory()
    if new_session_id:
        session.session_id = str(uuid.uuid4())
    return session


async def async_process_message(session: SessionMemory, user_message: str, service: Optional[ConversationService] = None) -> str:
    response = await (service or ConversationService()).reply(session, user_message)
    return response.text


def process_message(session: SessionMemory, user_message: str) -> str:
    return asyncio.run(async_process_message(session, user_message))


if __name__ == "__main__":
    service = ConversationService()
    session = new_session_memory()
    print("Dana Chatbot (type 'quit' to exit)\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        print(f"Dana: {asyncio.run(service.reply(session, user_input)).text}\n")
