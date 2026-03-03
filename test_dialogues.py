from conversation_manager import SessionMemory, process_message

def run_dialogue(name: str, turns: list):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    session = SessionMemory()
    for user_msg in turns:
        reply = process_message(session, user_msg)
        session.add_turn("user", user_msg)
        session.add_turn("assistant", reply)
        print(f"You:  {user_msg}")
        print(f"Dana: {reply}\n")

if __name__ == "__main__":

    run_dialogue("Happy Path Booking", [
        "I want to book an appointment",
        "Maria Gonzalez",
        "routine check-up",
        "Thursday at 10AM",
        "maria.g@email.com",
        "yes"
    ])

    run_dialogue("Invalid Time Booking", [
        "book me an appointment",
        "John Smith",
        "filling",
        "Sunday at 3PM",
        "Monday at 9AM",
        "john@email.com",
        "yes"
    ])

    run_dialogue("Cancellation with Late Fee", [
        "I need to cancel my appointment",
        "James Park",
        "yes",
        "yes",
        "no"
    ])

    run_dialogue("Reschedule", [
        "I want to reschedule my appointment",
        "Sarah Chen",
        "yes",
        "Monday at 2PM",
        "yes"
    ])

    run_dialogue("Out of Scope", [
        "can you recommend a restaurant?",
        "what services do you offer?"
    ])

# python conversation_manager.pyhu