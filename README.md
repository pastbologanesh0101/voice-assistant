# AI Voice Assistant

A small, fully offline, rule-based voice-assistant command engine written in
Python. No cloud speech APIs and no API keys are required for any part of
this project.

## Architecture

The project is deliberately split into two halves:

1. **The "brain"** — the `Assistant` class in `assistant.py`. It takes a
   plain text command string (representing already-transcribed speech) and
   returns a plain text response, by matching the command against an
   ordered list of rule-based skills (regex/keyword patterns — no ML). The
   first skill whose pattern matches wins; if nothing matches, a graceful
   fallback response is returned. The brain has **zero** dependency on
   audio hardware, which is what makes it fully unit-testable.

2. **Optional real voice I/O**, layered on top of the brain:
   - **Speech output** uses [`pyttsx3`](https://pypi.org/project/pyttsx3/),
     which drives the operating system's built-in *offline* TTS engine
     (no network, no API key). It is a soft/optional dependency — the
     `Speaker` wrapper class fails gracefully to a silent no-op if
     `pyttsx3` isn't installed or if there's no audio device available
     (e.g. in CI or a container).
   - **Speech input** is represented by the `Transcriber` extension
     point. By default, the CLI just reads text typed at a prompt (i.e.
     you type what you would have said) via `MockTranscriber`, which is
     also what the test suite uses. See below for how to plug in a real
     speech-to-text engine.

```
 audio in --> [Transcriber] --> text --> [Assistant.handle] --> text --> [Speaker] --> audio out
              (pluggable,                (pure logic,                    (optional,
               default: typed              fully tested)                  soft dependency)
               text via MockTranscriber)
```

## Skills

Commands are matched in this order; the first match wins:

1. **Time / date query** — "what time is it", "what's the date"
2. **Calculator** — "what is 12 times 7", "what is 20 divided by 4"
   (supports `+ - * /`, word forms like *plus/minus/times/divided by*,
   and reports a friendly error on divide-by-zero)
3. **Unit conversion** — km ↔ miles, celsius ↔ fahrenheit, kg ↔ lbs
   ("convert 10 km to miles", "100 celsius to fahrenheit")
4. **To-do list** — add / list / complete / remove items, persisted to a
   local JSON file (`todo.json` by default) so it survives restarts
5. **Timer** — "set a timer for 5 minutes" (acknowledges and confirms
   the parsed duration; does not actually block/wait)
6. **Q&A knowledge base** — ~20 canned facts (capitals, science trivia,
   etc.) matched by keyword, e.g. "what is the capital of France"
7. **Fallback** — "Sorry, I didn't understand that."

## Running it

```bash
# Optional: install pyttsx3 for spoken responses
pip install -r requirements.txt

python assistant.py
```

If `pyttsx3` isn't installed, or no audio output device is available, the
assistant automatically runs in text-only mode.

## Example session transcript

```
AI Voice Assistant (type 'quit' or 'exit' to leave)
> what time is it
The current time is 03:41 PM.
> what is 12 times 7
The answer is 84.
> convert 10 km to miles
10 km is 6.2137 mi.
> add a todo to buy milk
Added 'buy milk' to your to-do list as item 1.
> add a todo to walk the dog
Added 'walk the dog' to your to-do list as item 2.
> list my todos
Here's your to-do list: 1. buy milk (not done); 2. walk the dog (not done)
> complete todo 1
Marked item 1 as complete.
> set a timer for 5 minutes
Timer set for 5 minutes.
> what is the capital of france
The capital of France is Paris.
> asdkj nonsense request
Sorry, I didn't understand that.
> quit
Goodbye!
```

## Running the tests

```bash
pip install pytest
pytest -v
```

The test suite (`tests/test_assistant.py`) covers every skill, the
fallback path, calculator edge cases (including divide-by-zero), and
to-do list persistence across separate `Assistant` instances pointed at
the same JSON file. It never touches real audio hardware — the one test
that exercises `Speaker` does so with `enabled=False`.

A GitHub Actions workflow (`.github/workflows/tests.yml`) runs the full
suite on Python 3.11 and 3.12 on every push and pull request, without
installing `pyttsx3` (it isn't needed for the tests and CI has no audio
hardware).

## Plugging in real speech-to-text

The `Transcriber` abstract base class in `assistant.py` is the extension
point:

```python
class Transcriber(ABC):
    @abstractmethod
    def transcribe(self, audio) -> str:
        """Convert raw audio into a plain text command."""
```

`MockTranscriber` (used by default and in tests) just returns the input
as-is, treating it as already-transcribed text. To use a real, fully
offline speech recognizer such as [Vosk](https://alphacephei.com/vosk/):

```python
import json
from vosk import Model, KaldiRecognizer
from assistant import Transcriber, Assistant

class VoskTranscriber(Transcriber):
    def __init__(self, model_path: str):
        self._model = Model(model_path)

    def transcribe(self, audio: bytes) -> str:
        recognizer = KaldiRecognizer(self._model, 16000)
        recognizer.AcceptWaveform(audio)
        return json.loads(recognizer.Result()).get("text", "")

transcriber = VoskTranscriber("path/to/vosk-model")
assistant = Assistant()
text = transcriber.transcribe(audio_bytes)
print(assistant.handle(text))
```

A cloud STT API can be wrapped the exact same way — implement
`transcribe(audio) -> str` and hand the returned text to
`Assistant.handle()`. The assistant's core logic never needs to change.

## Project layout

```
assistant.py              # Assistant "brain", skills, Transcriber/Speaker, CLI
tests/test_assistant.py   # Unit tests (unittest, run via pytest)
requirements.txt          # Optional pyttsx3 dependency
.github/workflows/tests.yml
LICENSE                   # MIT
```

## License

MIT — see [LICENSE](LICENSE).
