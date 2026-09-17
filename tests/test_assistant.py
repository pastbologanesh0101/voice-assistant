"""Unit tests for the AI Voice Assistant "brain".

These tests exercise Assistant.handle() directly with plain text
commands (as MockTranscriber would produce) and never touch real
audio hardware. The optional Speaker/TTS wrapper is exercised once,
with enabled=False, purely to confirm it never raises -- no audio
playback happens in this test suite.
"""

import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant import (  # noqa: E402
    Assistant,
    FALLBACK_RESPONSE,
    MockTranscriber,
    Speaker,
    TodoList,
)


class AssistantTestCase(unittest.TestCase):
    def setUp(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(path)  # Assistant/TodoList should create it lazily.
        self.todo_path = path
        self.assistant = Assistant(todo_path=self.todo_path)

    def tearDown(self):
        if os.path.exists(self.todo_path):
            os.remove(self.todo_path)

    # -- time / date ------------------------------------------------------
    def test_time_query_matches(self):
        response = self.assistant.handle("what time is it")
        self.assertIn("time", response.lower())

    def test_date_query_matches(self):
        response = self.assistant.handle("what's the date today")
        self.assertIn("date", response.lower() + " date")  # sanity, always true-ish
        self.assertTrue(response.startswith("Today's date is"))

    # -- calculator ---------------------------------------------------------
    def test_calculator_addition(self):
        response = self.assistant.handle("what is 5 plus 3")
        self.assertEqual(response, "The answer is 8.")

    def test_calculator_subtraction(self):
        response = self.assistant.handle("what is 10 minus 4")
        self.assertEqual(response, "The answer is 6.")

    def test_calculator_multiplication(self):
        response = self.assistant.handle("what is 12 times 7")
        self.assertEqual(response, "The answer is 84.")

    def test_calculator_division(self):
        response = self.assistant.handle("what is 20 divided by 4")
        self.assertEqual(response, "The answer is 5.")

    def test_calculator_divide_by_zero(self):
        response = self.assistant.handle("what is 10 divided by 0")
        self.assertEqual(response, "I can't divide by zero.")

    # -- unit conversion ------------------------------------------------------
    def test_unit_conversion_km_to_miles(self):
        response = self.assistant.handle("convert 10 km to miles")
        match = re.search(r"([\d.]+)\s*mi\b", response)
        self.assertIsNotNone(match)
        self.assertAlmostEqual(float(match.group(1)), 6.21371, places=3)

    def test_unit_conversion_celsius_to_fahrenheit(self):
        response = self.assistant.handle("convert 100 celsius to fahrenheit")
        self.assertEqual(response, "100 c is 212 f.")

    def test_unit_conversion_kg_to_lbs(self):
        response = self.assistant.handle("convert 10 kg to lbs")
        match = re.search(r"([\d.]+)\s*lb\b", response)
        self.assertIsNotNone(match)
        self.assertAlmostEqual(float(match.group(1)), 22.0462, places=3)

    # -- to-do list --------------------------------------------------------
    def test_todo_add_and_list(self):
        add_response = self.assistant.handle("add a todo to buy milk")
        self.assertIn("buy milk", add_response)
        list_response = self.assistant.handle("list my todos")
        self.assertIn("buy milk", list_response)
        self.assertIn("not done", list_response)

    def test_todo_complete(self):
        self.assistant.handle("add a todo to walk the dog")
        response = self.assistant.handle("complete todo 1")
        self.assertIn("complete", response.lower())
        list_response = self.assistant.handle("list my todos")
        self.assertIn("(done)", list_response)

    def test_todo_remove(self):
        self.assistant.handle("add a todo to wash the car")
        response = self.assistant.handle("remove todo 1")
        self.assertIn("Removed", response)
        list_response = self.assistant.handle("list my todos")
        self.assertEqual(list_response, "Your to-do list is empty.")

    def test_todo_persists_across_instances(self):
        self.assistant.handle("add a todo to read a book")
        second_assistant = Assistant(todo_path=self.todo_path)
        list_response = second_assistant.handle("list my todos")
        self.assertIn("read a book", list_response)

    # -- timer -------------------------------------------------------------
    def test_timer_acknowledgement(self):
        response = self.assistant.handle("set a timer for 5 minutes")
        self.assertEqual(response, "Timer set for 5 minutes.")

    # -- Q&A knowledge base --------------------------------------------------
    def test_qa_lookup(self):
        response = self.assistant.handle("what is the capital of france")
        self.assertIn("Paris", response)

    def test_qa_lookup_second_fact(self):
        response = self.assistant.handle("what is the largest planet")
        self.assertIn("Jupiter", response)

    # -- fallback ------------------------------------------------------------
    def test_fallback_for_unmatched_command(self):
        response = self.assistant.handle("purple elephants dance sideways")
        self.assertEqual(response, FALLBACK_RESPONSE)

    def test_fallback_for_empty_command(self):
        response = self.assistant.handle("")
        self.assertEqual(response, FALLBACK_RESPONSE)

    # -- transcriber extension point -----------------------------------------
    def test_mock_transcriber_passthrough(self):
        transcriber = MockTranscriber()
        text = transcriber.transcribe("what is 2 plus 2")
        response = self.assistant.handle(text)
        self.assertEqual(response, "The answer is 4.")

    # -- speaker never raises, even without audio hardware --------------------
    def test_speaker_disabled_does_not_raise(self):
        speaker = Speaker(enabled=False)
        self.assertFalse(speaker.available)
        speaker.speak("this should be a silent no-op")  # must not raise

    def test_todo_list_direct_helper(self):
        todos = TodoList(self.todo_path)
        todos.add("item one")
        todos.add("item two")
        self.assertEqual(len(todos.list()), 2)
        self.assertTrue(todos.complete(1))
        self.assertFalse(todos.complete(99))
        self.assertTrue(todos.remove(2))
        self.assertEqual(len(todos.list()), 1)


if __name__ == "__main__":
    unittest.main()
