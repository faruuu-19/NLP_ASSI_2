from __future__ import annotations

import asyncio
import unittest

from fastapi.testclient import TestClient

from conversation_manager import ConversationService, SessionMemory, new_session_memory
from llm_engine import LLMUnavailableError
from main import app


class StubLLM:
    async def chat(self, system_prompt: str, history: list[dict], user_message: str) -> str:
        raise LLMUnavailableError("stubbed for tests")


class DialogueFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ConversationService(llm=StubLLM())
        self.session = new_session_memory()

    def ask(self, message: str) -> str:
        return asyncio.run(self.service.reply(self.session, message)).text

    def test_booking_happy_path(self) -> None:
        self.assertIn("full name", self.ask("I want to book an appointment").lower())
        self.assertIn("type of appointment", self.ask("Maria Gonzalez").lower())
        self.assertIn("date and time", self.ask("routine check-up").lower())
        self.assertIn("email address or phone number", self.ask("Thursday at 10 AM").lower())
        self.assertIn("should i book it", self.ask("maria@example.com").lower())
        self.assertIn("appointment is confirmed", self.ask("yes").lower())

    def test_invalid_time_is_rejected(self) -> None:
        self.ask("book me an appointment")
        self.ask("John Smith")
        self.ask("filling")
        reply = self.ask("Sunday at 3 PM")
        self.assertIn("closed on sundays", reply.lower())

    def test_cancellation_uses_the_booked_appointment_from_session_memory(self) -> None:
        self.ask("I want to book an appointment")
        self.ask("Amna Hameed")
        self.ask("routine check-up")
        self.ask("Monday at 3 PM")
        self.ask("amna@example.com")
        self.ask("yes")

        self.ask("I want to cancel an appointment")
        reply = self.ask("Amna Hameed")
        self.assertIn("routine check-up", reply.lower())
        self.assertIn("monday at 3:00 pm", reply.lower())

        cancel_reply = self.ask("yes").lower()
        if "late cancellation fee" in cancel_reply:
            cancel_reply = self.ask("yes").lower()
        self.assertIn("has been cancelled", cancel_reply)
        self.assertEqual(self.session.entities["appointments"][0]["status"], "cancelled")

    def test_overlapping_slot_is_rejected(self) -> None:
        self.ask("I want to book an appointment")
        self.ask("Amna Hameed")
        self.ask("cleaning")
        self.ask("Monday at 2 PM")
        self.ask("amna@example.com")
        self.ask("yes")

        self.ask("I want to book an appointment")
        self.ask("Amna Hameed")
        self.ask("root canal")
        reply = self.ask("Monday at 2 PM")
        self.assertIn("overlaps with another appointment", reply.lower())

    def test_reschedule_uses_memory_and_checks_conflicts(self) -> None:
        self.ask("I want to book an appointment")
        self.ask("Amna Hameed")
        self.ask("cleaning")
        self.ask("Monday at 2 PM")
        self.ask("amna@example.com")
        self.ask("yes")

        self.ask("I want to book an appointment")
        self.ask("Amna Hameed")
        self.ask("routine check-up")
        self.ask("Monday at 4 PM")
        self.ask("amna@example.com")
        self.ask("yes")

        self.ask("I want to reschedule my appointment")
        confirm_reply = self.ask("Amna Hameed")
        self.assertIn("multiple appointments", confirm_reply.lower())
        self.assertIn("routine check-up", self.ask("routine check-up").lower())
        self.assertIn("what new date and time", self.ask("yes").lower())
        conflict_reply = self.ask("Monday at 2 PM")
        self.assertIn("overlaps with another appointment", conflict_reply.lower())

    def test_out_of_scope_redirect(self) -> None:
        reply = self.ask("Can you recommend a restaurant nearby?")
        self.assertIn("can only assist", reply.lower())


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_chat_endpoint_is_stateless(self) -> None:
        session = self.client.post("/sessions").json()["session"]
        response = self.client.post("/chat", json={"message": "I want to book an appointment", "session": session})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("session", payload)
        self.assertEqual(payload["session"]["session_id"], session["session_id"])
        self.assertGreater(len(payload["session"]["history"]), 0)

    def test_websocket_streaming(self) -> None:
        with self.client.websocket_connect("/ws/chat") as websocket:
            session_message = websocket.receive_json()
            self.assertEqual(session_message["type"], "session")
            websocket.send_json({"message": "I want to book an appointment", "session": session_message["session"]})
            start = websocket.receive_json()
            self.assertEqual(start["type"], "start")
            deltas = []
            while True:
                event = websocket.receive_json()
                if event["type"] == "delta":
                    deltas.append(event["content"])
                if event["type"] == "end":
                    self.assertTrue("".join(deltas).strip())
                    self.assertIn("session", event)
                    break


if __name__ == "__main__":
    unittest.main()
