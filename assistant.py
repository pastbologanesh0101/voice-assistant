"""AI Voice Assistant
=====================

A small, fully offline, rule-based voice-assistant command engine.

Architecture
------------
The project is split into two clearly separated halves:

1. The "brain" (this module, minus the ``main()`` REPL): the
   :class:`Assistant` class and its skills. It is pure text-in/text-out
   logic with no dependency on microphones, speakers, or network access,
   which makes it trivially unit-testable.

2. Optional real voice I/O, which is layered on top of the brain and can
   be swapped out or disabled entirely:

   - Speech *output* uses `pyttsx3 <https://pypi.org/project/pyttsx3/>`_,
     which drives the operating system's built-in offline TTS engine.
     It is a *soft* dependency: if it is not installed, or if there is
     no audio device available (e.g. in CI), the assistant silently
     falls back to text-only output.
   - Speech *input* is represented by the :class:`Transcriber`
     extension point (see below). By default the CLI simply reads
     already-"transcribed" text from stdin (i.e. you type what you
     would have said). A real speech-to-text engine can be plugged in
     by implementing :meth:`Transcriber.transcribe`.

No cloud API keys are required anywhere in this project.

Skills
------
The :class:`Assistant` matches an incoming command string against a
fixed, ordered list of rule-based skills (regex/keyword patterns, not
ML). The first skill whose pattern matches handles the command. If none
match, a graceful fallback response is returned.

Skills, in matching order:

1. **Time/date query** -- "what time is it", "what's the date"
2. **Calculator** -- "what is 12 times 7", "calculate 9 / 3"
3. **Unit conversion** -- km/miles, celsius/fahrenheit, kg/lbs
4. **To-do list** -- add / list / complete / remove items, persisted
   to a local JSON file
5. **Timer** -- "set a timer for 5 minutes" (acknowledgement only,
   does not actually block/wait)
6. **Q&A knowledge base** -- ~18 canned facts matched by keyword
7. **Fallback** -- "Sorry, I didn't understand that."

Extension point for real speech-to-text
----------------------------------------
See :class:`Transcriber` below. Implement ``transcribe(audio) -> str``
against a real engine (e.g. the offline `Vosk
<https://alphacephei.com/vosk/>`_ toolkit, or a cloud STT API) and pass
an instance of it to whatever captures your audio; the rest of the
pipeline (the :class:`Assistant`) does not need to change at all,
since it only ever consumes plain text commands.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

DEFAULT_TODO_FILE = "todo.json"

FALLBACK_RESPONSE = "Sorry, I didn't understand that."


# ---------------------------------------------------------------------------
# Speech-to-text extension point
# ---------------------------------------------------------------------------
class Transcriber(ABC):
    """Extension point for plugging in real speech-to-text.

    The assistant's "brain" only ever consumes plain text. Anything that
    turns audio into text implements this interface. To use a real
    offline engine such as Vosk::

        from vosk import Model, KaldiRecognizer

        class VoskTranscriber(Transcriber):
            def __init__(self, model_path: str):
                self._model = Model(model_path)

            def transcribe(self, audio: bytes) -> str:
                recognizer = KaldiRecognizer(self._model, 16000)
                recognizer.AcceptWaveform(audio)
                result = json.loads(recognizer.Result())
                return result.get("text", "")

    Or wrap a cloud STT API in the same way. Whatever the source, feed
    the returned string straight into ``Assistant.handle(text)``.
    """

    @abstractmethod
    def transcribe(self, audio) -> str:
        """Convert raw audio into a plain text command."""
        raise NotImplementedError


class MockTranscriber(Transcriber):
    """A trivial transcriber for tests and offline development.

    ``audio`` is simply expected to already be a string (representing
    "the text the user would have spoken"), and it is returned as-is.
    This lets tests and the default CLI exercise the exact same code
    path a real transcriber would, without needing a microphone.
    """

    def transcribe(self, audio) -> str:
        return str(audio)


# ---------------------------------------------------------------------------
# Optional offline text-to-speech
# ---------------------------------------------------------------------------
class Speaker:
    """Thin, fail-safe wrapper around pyttsx3.

    pyttsx3 is a soft dependency: if it isn't installed, or if engine
    initialization fails (e.g. no audio device, which is the normal
    case in CI/containers), :meth:`speak` becomes a silent no-op
    instead of raising.
    """

    def __init__(self, enabled: bool = True):
        self._engine = None
        if not enabled:
            return
        try:
            import pyttsx3  # type: ignore

            self._engine = pyttsx3.init()
        except Exception:
            # No pyttsx3 installed, or no audio backend available.
            self._engine = None

    @property
    def available(self) -> bool:
        return self._engine is not None

    def speak(self, text: str) -> None:
        if not self._engine:
            return
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception:
            # Never let a speech failure crash the assistant.
            pass


# ---------------------------------------------------------------------------
# To-do list, persisted to a local JSON file
# ---------------------------------------------------------------------------
class TodoList:
    """A local to-do list persisted to a JSON file on disk.

    Each item is stored as ``{"text": str, "done": bool}``. The file is
    read fresh on every mutation-free query and rewritten on every
    mutation, so two separate ``TodoList``/``Assistant`` instances
    pointed at the same file will see each other's changes.
    """

    def __init__(self, path: str = DEFAULT_TODO_FILE):
        self.path = path

    def _load(self) -> List[Dict]:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self, items: List[Dict]) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(items, fh, indent=2)

    def add(self, text: str) -> int:
        items = self._load()
        items.append({"text": text, "done": False})
        self._save(items)
        return len(items)

    def list(self) -> List[Dict]:
        return self._load()

    def complete(self, index: int) -> bool:
        items = self._load()
        if 1 <= index <= len(items):
            items[index - 1]["done"] = True
            self._save(items)
            return True
        return False

    def remove(self, index: int) -> bool:
        items = self._load()
        if 1 <= index <= len(items):
            items.pop(index - 1)
            self._save(items)
            return True
        return False


# ---------------------------------------------------------------------------
# Small Q&A knowledge base (~18 canned facts, matched by keyword)
# ---------------------------------------------------------------------------
QA_FACTS: List[Tuple[str, str]] = [
    ("capital of france", "The capital of France is Paris."),
    ("capital of japan", "The capital of Japan is Tokyo."),
    ("capital of india", "The capital of India is New Delhi."),
    ("capital of the united states", "The capital of the United States is Washington, D.C."),
    ("capital of australia", "The capital of Australia is Canberra."),
    ("largest planet", "Jupiter is the largest planet in our solar system."),
    ("smallest planet", "Mercury is the smallest planet in our solar system."),
    ("speed of light", "The speed of light in a vacuum is about 299,792 kilometers per second."),
    ("tallest mountain", "Mount Everest is the tallest mountain above sea level."),
    ("longest river", "The Nile is generally considered the longest river in the world."),
    ("who wrote romeo and juliet", "Romeo and Juliet was written by William Shakespeare."),
    ("who painted the mona lisa", "The Mona Lisa was painted by Leonardo da Vinci."),
    ("boiling point of water", "Water boils at 100 degrees Celsius at sea level."),
    ("freezing point of water", "Water freezes at 0 degrees Celsius at sea level."),
    ("how many continents", "There are seven continents on Earth."),
    ("how many planets", "There are eight planets in our solar system."),
    ("who is the founder of python", "Python was created by Guido van Rossum."),
    ("what is the chemical symbol for gold", "The chemical symbol for gold is Au."),
    ("what is the chemical symbol for water", "The chemical symbol for water is H2O."),
    ("largest ocean", "The Pacific Ocean is the largest ocean on Earth."),
]


# ---------------------------------------------------------------------------
# The Assistant "brain"
# ---------------------------------------------------------------------------
_NUMBER = r"-?\d+(?:\.\d+)?"

_CALC_OPERATORS = {
    "plus": "+",
    "add": "+",
    "added to": "+",
    "minus": "-",
    "subtract": "-",
    "subtracted from": "-",
    "times": "*",
    "multiplied by": "*",
    "multiply": "*",
    "x": "*",
    "divided by": "/",
    "divide": "/",
    "over": "/",
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
}
# Longest keys first so e.g. "divided by" matches before "divide".
_CALC_OP_PATTERN = "|".join(
    re.escape(op) for op in sorted(_CALC_OPERATORS, key=len, reverse=True)
)
_CALC_RE = re.compile(
    rf"({_NUMBER})\s*(?:{_CALC_OP_PATTERN})\s*({_NUMBER})", re.IGNORECASE
)
# Used only to detect *which* operator word was used (calc_re above is
# operator-agnostic on purpose so " x " style expressions also work).
_CALC_OP_FINDER = re.compile(
    rf"{_NUMBER}\s*({_CALC_OP_PATTERN})\s*{_NUMBER}", re.IGNORECASE
)

_UNIT_ALIASES = {
    "km": "km", "kilometer": "km", "kilometers": "km", "kilometre": "km", "kilometres": "km",
    "mi": "mi", "mile": "mi", "miles": "mi",
    "c": "c", "celsius": "c",
    "f": "f", "fahrenheit": "f",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
}
_UNIT_WORD = "|".join(sorted(_UNIT_ALIASES, key=len, reverse=True))
_CONVERT_RE = re.compile(
    rf"({_NUMBER})\s*({_UNIT_WORD})\s*(?:to|in|into)\s*({_UNIT_WORD})", re.IGNORECASE
)


class Assistant:
    """The rule-based voice-assistant "brain".

    Feed it plain text commands (already-transcribed speech) via
    :meth:`handle`, and it returns a plain text response. It has no
    dependency on audio hardware whatsoever.
    """

    def __init__(self, todo_path: str = DEFAULT_TODO_FILE):
        self.todos = TodoList(todo_path)
        # Ordered list of (skill_name, handler). The first handler that
        # returns a non-None response wins.
        self._skills: List[Tuple[str, Callable[[str], Optional[str]]]] = [
            ("time_date", self._skill_time_date),
            ("calculator", self._skill_calculator),
            ("unit_conversion", self._skill_unit_conversion),
            ("todo", self._skill_todo),
            ("timer", self._skill_timer),
            ("qa", self._skill_qa),
        ]

    def handle(self, command: str) -> str:
        """Match ``command`` against skills in order; return the response."""
        text = (command or "").strip()
        if not text:
            return FALLBACK_RESPONSE
        for _name, handler in self._skills:
            response = handler(text)
            if response is not None:
                return response
        return FALLBACK_RESPONSE

    # -- Skill: time / date --------------------------------------------
    def _skill_time_date(self, text: str) -> Optional[str]:
        lowered = text.lower()
        wants_time = bool(re.search(r"\b(what(?:'s| is)?\s+the\s+)?time\b", lowered))
        wants_date = bool(re.search(r"\b(what(?:'s| is)?\s+the\s+)?date\b|\btoday'?s date\b", lowered))
        if not (wants_time or wants_date):
            return None
        now = datetime.now()
        if wants_time and wants_date:
            return f"It's currently {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d, %Y')}."
        if wants_time:
            return f"The current time is {now.strftime('%I:%M %p')}."
        return f"Today's date is {now.strftime('%A, %B %d, %Y')}."

    # -- Skill: calculator ------------------------------------------------
    def _skill_calculator(self, text: str) -> Optional[str]:
        lowered = text.lower()
        match = _CALC_RE.search(lowered)
        if not match:
            return None
        op_match = _CALC_OP_FINDER.search(lowered)
        if not op_match:
            return None
        left = float(match.group(1))
        right = float(match.group(2))
        op_word = op_match.group(1)
        op = _CALC_OPERATORS.get(op_word)
        if op is None:
            return None

        if op == "+":
            result = left + right
        elif op == "-":
            result = left - right
        elif op == "*":
            result = left * right
        elif op == "/":
            if right == 0:
                return "I can't divide by zero."
            result = left / right
        else:
            return None

        if result == int(result):
            result_str = str(int(result))
        else:
            result_str = str(round(result, 6))
        return f"The answer is {result_str}."

    # -- Skill: unit conversion -------------------------------------------
    def _skill_unit_conversion(self, text: str) -> Optional[str]:
        lowered = text.lower()
        match = _CONVERT_RE.search(lowered)
        if not match:
            return None
        value = float(match.group(1))
        from_unit = _UNIT_ALIASES[match.group(2)]
        to_unit = _UNIT_ALIASES[match.group(3)]

        result = self._convert(value, from_unit, to_unit)
        if result is None:
            return (
                f"I don't know how to convert {from_unit} to {to_unit}."
            )
        return f"{self._fmt(value)} {from_unit} is {self._fmt(result)} {to_unit}."

    @staticmethod
    def _fmt(n: float) -> str:
        if n == int(n):
            return str(int(n))
        return str(round(n, 4))

    @staticmethod
    def _convert(value: float, from_unit: str, to_unit: str) -> Optional[float]:
        if from_unit == to_unit:
            return value
        if from_unit == "km" and to_unit == "mi":
            return value * 0.621371
        if from_unit == "mi" and to_unit == "km":
            return value / 0.621371
        if from_unit == "c" and to_unit == "f":
            return value * 9 / 5 + 32
        if from_unit == "f" and to_unit == "c":
            return (value - 32) * 5 / 9
        if from_unit == "kg" and to_unit == "lb":
            return value * 2.20462
        if from_unit == "lb" and to_unit == "kg":
            return value / 2.20462
        return None

    # -- Skill: to-do list --------------------------------------------------
    def _skill_todo(self, text: str) -> Optional[str]:
        lowered = text.lower()

        add_match = re.match(
            r"add\s+(?:a\s+)?(?:todo|task|item)?\s*(?:to\s+)?(.+)", lowered
        )
        if lowered.startswith("add") and add_match and add_match.group(1).strip():
            item_text = add_match.group(1).strip()
            position = self.todos.add(item_text)
            return f"Added '{item_text}' to your to-do list as item {position}."

        if re.search(r"\b(list|show)\b.*\b(todo|task)s?\b", lowered) or re.search(
            r"what'?s on my to-?do list", lowered
        ):
            items = self.todos.list()
            if not items:
                return "Your to-do list is empty."
            lines = []
            for idx, item in enumerate(items, start=1):
                status = "done" if item.get("done") else "not done"
                lines.append(f"{idx}. {item['text']} ({status})")
            return "Here's your to-do list: " + "; ".join(lines)

        complete_match = re.search(
            r"\b(complete|finish|mark)\b(?:\s+(?:todo|task|item))?\s*(?:number\s*)?(\d+)",
            lowered,
        )
        if complete_match:
            index = int(complete_match.group(2))
            if self.todos.complete(index):
                return f"Marked item {index} as complete."
            return f"I couldn't find item {index} on your to-do list."

        remove_match = re.search(
            r"\b(remove|delete)\b(?:\s+(?:todo|task|item))?\s*(?:number\s*)?(\d+)",
            lowered,
        )
        if remove_match:
            index = int(remove_match.group(2))
            if self.todos.remove(index):
                return f"Removed item {index} from your to-do list."
            return f"I couldn't find item {index} on your to-do list."

        return None

    # -- Skill: timer --------------------------------------------------------
    def _skill_timer(self, text: str) -> Optional[str]:
        lowered = text.lower()
        match = re.search(
            r"set\s+(?:a\s+)?timer\s+for\s+(\d+(?:\.\d+)?)\s*(hour|hr|minute|min|second|sec)s?",
            lowered,
        )
        if not match:
            return None
        amount = match.group(1)
        unit_raw = match.group(2)
        unit_map = {
            "hour": "hour", "hr": "hour",
            "minute": "minute", "min": "minute",
            "second": "second", "sec": "second",
        }
        unit = unit_map[unit_raw]
        plural = "" if amount == "1" else "s"
        return f"Timer set for {amount} {unit}{plural}."

    # -- Skill: Q&A knowledge base -------------------------------------------
    def _skill_qa(self, text: str) -> Optional[str]:
        lowered = text.lower()
        for keyword, answer in QA_FACTS:
            if keyword in lowered:
                return answer
        return None


# ---------------------------------------------------------------------------
# CLI REPL
# ---------------------------------------------------------------------------
def main() -> None:
    assistant = Assistant()
    speaker = Speaker(enabled=True)

    print("AI Voice Assistant (type 'quit' or 'exit' to leave)")
    if not speaker.available:
        print("(Text-to-speech unavailable -- running in text-only mode.)")

    while True:
        try:
            command = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if command.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        response = assistant.handle(command)
        print(response)
        speaker.speak(response)


if __name__ == "__main__":
    main()
