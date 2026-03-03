import requests
from collections import deque
from datetime import datetime
from state_machine import (
    Intent, BookingState, CancelState, RescheduleState,
    detect_intent, get_initial_state,
    is_valid_appointment_type, is_valid_time,
    is_within_booking_window, is_within_cancellation_window,
    VALID_APPOINTMENT_TYPES
)
from system_prompt import DANA_SYSTEM_PROMPT

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL      = "qwen2.5:1.5b"

# ── LLM caller ─────────────────────────────────────────────────────────────────
def ask_llm(system: str, history: list, user_message: str) -> str:
    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.8
            }
        }
    )
    return response.json()["message"]["content"].strip()

# ── Session ─────────────────────────────────────────────────────────────────────
class SessionMemory:
    def __init__(self, window_size=6):
        self.history        = deque(maxlen=window_size)
        self.intent         = None
        self.state          = None
        self.entities       = {}
        self.policy_flags   = {}

    def add_turn(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def set_entity(self, key: str, value):
        self.entities[key] = value

    def get_entity(self, key: str, default=None):
        return self.entities.get(key, default)

# ── Hardcoded responses (no LLM needed) ────────────────────────────────────────
QUESTIONS = {
    BookingState.COLLECT_NAME:     "May I have your full name please?",
    BookingState.COLLECT_TYPE:     (
        "What type of appointment do you need? We offer: routine check-up, "
        "cleaning, filling, teeth whitening, orthodontic consultation, "
        "root canal, or emergency care."
    ),
    BookingState.COLLECT_DATETIME: "What date and time would you prefer? We are open Monday to Friday 9AM-5PM and Saturday 9AM-1PM.",
    BookingState.COLLECT_CONTACT:  "Could I get your email address or phone number for the confirmation?",
    BookingState.CONFIRM:          None,  # built dynamically
    CancelState.COLLECT_NAME:      "Could I have the name the appointment is under?",
    CancelState.CONFIRM_APPOINTMENT: None,  # built dynamically
    CancelState.APPLY_POLICY:      None,  # built dynamically
    CancelState.CONFIRM_CANCEL:    "Would you like to reschedule for a future date?",
    RescheduleState.COLLECT_NAME:      "Could I have the name the appointment is under?",
    RescheduleState.CONFIRM_APPOINTMENT: None,  # built dynamically
    RescheduleState.COLLECT_NEW_DATETIME: "What new date and time would you prefer?",
    RescheduleState.CONFIRM:       None,  # built dynamically
}

# ── Main router ────────────────────────────────────────────────────────────────
def process_message(session: SessionMemory, user_message: str) -> str:

    # Detect intent on first turn
    if session.intent is None or session.intent == Intent.OUT_OF_SCOPE:
        session.intent = detect_intent(user_message)
        session.state  = get_initial_state(session.intent)

        if session.intent == Intent.OUT_OF_SCOPE or session.intent == Intent.CLINIC_INFO:
            # Reset intent so next message can re-detect
            session.intent = None
            return handle_general(session, user_message)

    # Route to correct handler
    if session.intent == Intent.BOOK_APPOINTMENT:
        return handle_booking(session, user_message)
    elif session.intent == Intent.CANCEL_APPOINTMENT:
        return handle_cancel(session, user_message)
    elif session.intent == Intent.RESCHEDULE_APPOINTMENT:
        return handle_reschedule(session, user_message)
    else:
        return handle_general(session, user_message)


# ── Booking flow ───────────────────────────────────────────────────────────────
def handle_booking(session: SessionMemory, user_message: str) -> str:
    state = session.state

    if state == BookingState.COLLECT_NAME:
        session.set_entity("name", user_message.strip().title())
        session.state = BookingState.COLLECT_TYPE
        name = session.get_entity("name").split()[0]
        return f"Nice to meet you, {name}! " + QUESTIONS[BookingState.COLLECT_TYPE]

    elif state == BookingState.COLLECT_TYPE:
        if not is_valid_appointment_type(user_message):
            return (
                "I didn't recognize that appointment type. Please choose from: "
                "routine check-up, cleaning, filling, teeth whitening, "
                "orthodontic consultation, root canal, or emergency care."
            )
        session.set_entity("appointment_type", user_message.strip().lower())
        session.state = BookingState.COLLECT_DATETIME
        return QUESTIONS[BookingState.COLLECT_DATETIME]

    elif state == BookingState.COLLECT_DATETIME:
        # Let LLM extract day/time then validate in Python
        extraction_prompt = (
            f"Extract the day of week and time from this message: '{user_message}'. "
            f"Reply in exactly this format and nothing else:\n"
            f"DAY: <day>\nTIME: <HH:MM>"
        )
        extracted = ask_llm("You are a data extractor. Only reply in the exact format requested.", [], extraction_prompt)
        
        try:
            lines = extracted.strip().split("\n")
            day  = lines[0].split(":")[1].strip().lower()
            time_str = lines[1].split(":", 1)[1].strip()
            hour, minute = map(int, time_str.split(":"))

            valid, error = is_valid_time(day, hour, minute)
            if not valid:
                return error

            session.set_entity("day", day.capitalize())
            session.set_entity("time", time_str)
            session.state = BookingState.COLLECT_CONTACT
            return QUESTIONS[BookingState.COLLECT_CONTACT]

        except Exception:
            return "I didn't quite catch that. Could you please specify the day and time? For example: Thursday at 10AM."

    elif state == BookingState.COLLECT_CONTACT:
        session.set_entity("contact", user_message.strip())
        session.state = BookingState.CONFIRM
        name  = session.get_entity("name")
        atype = session.get_entity("appointment_type")
        day   = session.get_entity("day")
        time  = session.get_entity("time")
        contact = session.get_entity("contact")
        return (
            f"Let me confirm: a {atype} for {name} on {day} at {time}, "
            f"confirmation to {contact}. Shall I go ahead and book this?"
        )

    elif state == BookingState.CONFIRM:
        if any(word in user_message.lower() for word in ["yes", "confirm", "sure", "go ahead", "correct", "yeah", "yep"]):
            session.state = BookingState.COMPLETE
            name = session.get_entity("name").split()[0]
            day  = session.get_entity("day")
            time = session.get_entity("time")
            return (
                f"Perfect! Your appointment is confirmed. "
                f"We look forward to seeing you on {day} at {time}, {name}. "
                f"Please arrive 10 minutes early. Is there anything else I can help you with?"
            )
        else:
            session.state = BookingState.COLLECT_DATETIME
            return "No problem! Let's try again. What date and time would you prefer?"

    elif state == BookingState.COMPLETE:
        # New intent after booking
        session.intent = detect_intent(user_message)
        session.state  = get_initial_state(session.intent)
        return handle_general(session, user_message)

    return "I'm sorry, something went wrong. Let's start over."


# ── Cancel flow ────────────────────────────────────────────────────────────────
def handle_cancel(session: SessionMemory, user_message: str) -> str:
    state = session.state

    if state == CancelState.COLLECT_NAME:
        session.set_entity("name", user_message.strip().title())
        session.state = CancelState.CONFIRM_APPOINTMENT
        name = session.get_entity("name").split()[0]
        # Simulated appointment lookup
        return (
            f"Thank you, {name}. I have a filling appointment on Wednesday at 2:00PM on file. "
            f"Is that the one you'd like to cancel?"
        )

    elif state == CancelState.CONFIRM_APPOINTMENT:
        if any(word in user_message.lower() for word in ["yes", "correct", "that's it", "yeah", "yep"]):
            session.state = CancelState.APPLY_POLICY
            # Simulate checking if within 24 hours
            is_late = True  # hardcoded for simulation
            session.policy_flags["late_cancellation"] = is_late
            if is_late:
                return (
                    "I want to let you know that since your appointment is within 24 hours, "
                    "a $25 late cancellation fee may apply. Would you still like to proceed?"
                )
            else:
                session.state = CancelState.CONFIRM_CANCEL
                return "Your appointment has been cancelled. Would you like to reschedule for a future date?"
        else:
            return "Could you please provide the name or reference number for the appointment you'd like to cancel?"

    elif state == CancelState.APPLY_POLICY:
        if any(word in user_message.lower() for word in ["yes", "proceed", "sure", "go ahead", "yeah", "yep"]):
            session.state = CancelState.CONFIRM_CANCEL
            name = session.get_entity("name").split()[0]
            return (
                f"Understood, {name}. Your appointment has been cancelled and a notice "
                f"will be sent to your contact on file. Would you like to reschedule?"
            )
        else:
            return "No problem. Your appointment has not been cancelled. Is there anything else I can help you with?"

    elif state == CancelState.CONFIRM_CANCEL:
        if any(word in user_message.lower() for word in ["yes", "sure", "yeah", "yep"]):
            session.intent = Intent.BOOK_APPOINTMENT
            session.state  = BookingState.COLLECT_DATETIME
            name = session.get_entity("name").split()[0]
            return f"Of course, {name}! What date and time would you prefer for your new appointment?"
        else:
            session.state = CancelState.COMPLETE
            name = session.get_entity("name").split()[0]
            return f"No problem, {name}. Feel free to reach out whenever you're ready to rebook. Take care!"

    return "I'm sorry, something went wrong. Let's start over."


# ── Reschedule flow ────────────────────────────────────────────────────────────
def handle_reschedule(session: SessionMemory, user_message: str) -> str:
    state = session.state

    if state == RescheduleState.COLLECT_NAME:
        session.set_entity("name", user_message.strip().title())
        session.state = RescheduleState.CONFIRM_APPOINTMENT
        name = session.get_entity("name").split()[0]
        return (
            f"Hi {name}! I have a teeth whitening consultation on Friday at 11:00AM on file. "
            f"Is that the one you'd like to reschedule?"
        )

    elif state == RescheduleState.CONFIRM_APPOINTMENT:
        if any(word in user_message.lower() for word in ["yes", "correct", "that's it", "yeah", "yep"]):
            session.state = RescheduleState.COLLECT_NEW_DATETIME
            return QUESTIONS[RescheduleState.COLLECT_NEW_DATETIME]
        else:
            return "Could you clarify which appointment you'd like to reschedule?"

    elif state == RescheduleState.COLLECT_NEW_DATETIME:
        extraction_prompt = (
            f"Extract the day of week and time from this message: '{user_message}'. "
            f"Reply in exactly this format and nothing else:\n"
            f"DAY: <day>\nTIME: <HH:MM>"
        )
        extracted = ask_llm("You are a data extractor. Only reply in the exact format requested.", [], extraction_prompt)

        try:
            lines    = extracted.strip().split("\n")
            day      = lines[0].split(":")[1].strip().lower()
            time_str = lines[1].split(":", 1)[1].strip()
            hour, minute = map(int, time_str.split(":"))

            valid, error = is_valid_time(day, hour, minute)
            if not valid:
                return error

            session.set_entity("new_day", day.capitalize())
            session.set_entity("new_time", time_str)
            session.state = RescheduleState.CONFIRM
            name  = session.get_entity("name")
            return (
                f"To confirm: rescheduling your appointment to {day.capitalize()} "
                f"at {time_str} for {name}. Shall I go ahead?"
            )
        except Exception:
            return "I didn't quite catch that. Could you specify the day and time? For example: Monday at 2PM."

    elif state == RescheduleState.CONFIRM:
        if any(word in user_message.lower() for word in ["yes", "confirm", "sure", "go ahead", "yeah", "yep"]):
            session.state = RescheduleState.COMPLETE
            name = session.get_entity("name").split()[0]
            day  = session.get_entity("new_day")
            time = session.get_entity("new_time")
            return (
                f"Done! Your appointment has been rescheduled to {day} at {time}. "
                f"A confirmation will be sent to your contact on file. "
                f"See you then, {name}!"
            )
        else:
            session.state = RescheduleState.COLLECT_NEW_DATETIME
            return "No problem. What date and time would you prefer instead?"

    return "I'm sorry, something went wrong. Let's start over."


# ── General / info / out of scope ──────────────────────────────────────────────
def handle_general(session: SessionMemory, user_message: str) -> str:
    system = DANA_SYSTEM_PROMPT + """
You can answer questions about:
- Clinic hours: Monday-Friday 9AM-5PM, Saturday 9AM-1PM, Sunday closed
- Services: routine check-ups, cleanings, fillings, teeth whitening, 
  orthodontic consultations, root canal, emergency dental care
- Emergency: same-day care available
- Pricing and insurance: tell patient to contact the clinic directly
- Out of scope: politely redirect to dental scheduling

Never make up information. Keep response to 1-2 sentences.
After answering, offer to book an appointment.
"""
    return ask_llm(system, list(session.history), user_message)


# ── Main chat loop ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    session = SessionMemory()
    print("Dana Chatbot (type 'quit' to exit)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break

        reply = process_message(session, user_input)
        session.add_turn("user", user_input)
        session.add_turn("assistant", reply)
        print(f"Dana: {reply}\n")