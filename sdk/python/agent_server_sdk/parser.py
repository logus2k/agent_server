"""Streaming + whole-response parser for agent_server output.

agent_server folds a model's reasoning and spoken-summary channels INTO the
content stream as literal tags:

    <think>...reasoning...</think><voice>...spoken summary...</voice>answer body

Every client has to separate those three channels. Hand-rolling that parse is
exactly where bugs live (an unclosed ``</voice>`` once made a CV client read a
whole answer aloud and speak the literal closing tag). This module is the one
tested implementation so no client has to reinvent it.

Two entry points:

* :class:`StreamParser` — feed streamed ``content`` deltas, get typed events
  (``thinking`` / ``voice`` / ``answer``) as soon as each is known.
* :func:`split_response` — split a complete (non-streamed) content string into
  ``(thinking, voice, answer)``.

:func:`sanitize_for_tts` is whitespace/format tolerant: it strips even a
malformed or truncated ``<voice>`` / ``<think>`` tag so a stray tag can never
be spoken aloud.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, List, Literal, Optional, Tuple

__all__ = ["StreamEvent", "StreamParser", "split_response", "sanitize_for_tts"]

EventKind = Literal["thinking", "voice", "answer"]


@dataclass
class StreamEvent:
    """One parsed fragment of the stream.

    ``kind`` is ``"thinking"``, ``"voice"`` or ``"answer"``. ``text`` is the
    fragment for that channel. ``final`` is True on the closing event of the
    ``voice`` channel (the moment the full spoken summary is known and safe to
    hand to TTS).
    """

    kind: EventKind
    text: str
    final: bool = False


# A complete <think>...</think> block (whitespace tolerant on the tags).
_THINK_BLOCK = re.compile(r"<\s*think\s*>[\s\S]*?<\s*/\s*think\s*>", re.IGNORECASE)
_VOICE_BLOCK = re.compile(r"<\s*voice\s*>([\s\S]*?)<\s*/\s*voice\s*>", re.IGNORECASE)

# Tag-ish fragments used by the TTS sanitiser.
_ANY_THINK_VOICE_TAG = re.compile(r"<\s*/?\s*(?:think|voice)\s*>", re.IGNORECASE)
_TRAILING_PARTIAL_TAG = re.compile(r"<\s*/?\s*(?:think|voice)\b[^>]*$", re.IGNORECASE)
_OPEN_THINK = re.compile(r"<\s*think\s*>", re.IGNORECASE)
_OPEN_VOICE = re.compile(r"<\s*voice\s*>", re.IGNORECASE)
_CLOSE_THINK = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)
_CLOSE_VOICE = re.compile(r"<\s*/\s*voice\s*>", re.IGNORECASE)


def sanitize_for_tts(text: str) -> str:
    """Return ``text`` safe to send to a TTS engine.

    Strips complete and *malformed/truncated* ``<think>``/``<voice>`` tags
    (e.g. ``</ voice>``, ``</voice``) so a leaked tag is never spoken as
    "slash voice", plus markdown/citation noise that would be read literally.
    """
    t = str(text)
    t = re.sub(r"<\s*think\s*>[\s\S]*?<\s*/\s*think\s*>", "", t, flags=re.IGNORECASE)
    t = re.sub(r"<\s*think\s*>[\s\S]*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"<\s*voice\s*>[\s\S]*?<\s*/\s*voice\s*>", "", t, flags=re.IGNORECASE)
    t = re.sub(r"<\s*voice\s*>[\s\S]*$", "", t, flags=re.IGNORECASE)
    t = _ANY_THINK_VOICE_TAG.sub("", t)
    t = _TRAILING_PARTIAL_TAG.sub("", t)
    t = re.sub(r"```[\s\S]*?```", " ", t)                       # code fences
    t = re.sub(r"\[(?:markdown_chunk|E|R):[^\]]+\]", "", t)     # citation tags
    t = re.sub(r"\[C\d+\]", "", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)              # md links -> text
    t = re.sub(r"^\s*#{1,6}\s+", "", t, flags=re.MULTILINE)     # headings
    t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.MULTILINE)      # bullets
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"`([^`]+)`", r"\1", t)
    t = re.sub(r"\n{2,}", ". ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def split_response(content: str) -> Tuple[str, str, str]:
    """Split a *complete* content string into ``(thinking, voice, answer)``.

    ``thinking`` and ``voice`` are the inner text of the first such block
    (empty string if absent). ``answer`` is everything else, with both blocks
    removed and surrounding whitespace trimmed.
    """
    content = content or ""
    think = ""
    m = _THINK_BLOCK.search(content)
    if m:
        inner = re.sub(r"^<\s*think\s*>|<\s*/\s*think\s*>$", "", m.group(0), flags=re.IGNORECASE)
        think = inner.strip()
    voice = ""
    mv = _VOICE_BLOCK.search(content)
    if mv:
        voice = mv.group(1).strip()
    answer = _THINK_BLOCK.sub("", content)
    answer = _VOICE_BLOCK.sub("", answer)
    # Drop any dangling unclosed opener and its tail (defensive).
    answer = re.sub(r"<\s*think\s*>[\s\S]*$", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"<\s*voice\s*>[\s\S]*$", "", answer, flags=re.IGNORECASE)
    return think, voice, answer.strip()


@dataclass
class StreamParser:
    """Incremental parser. Feed ``content`` deltas, receive typed events.

    Example::

        parser = StreamParser()
        for delta in stream_of_content_deltas:
            for ev in parser.feed(delta):
                if ev.kind == "answer":
                    render(ev.text)
                elif ev.kind == "voice" and ev.final:
                    speak(sanitize_for_tts(ev.text))
        for ev in parser.flush():
            ...

    State machine over a small carry buffer: ``answer`` -> (``thinking`` |
    ``voice``) -> ``answer``. A partial tag at a chunk boundary is held back in
    the buffer until the next ``feed``.
    """

    _buf: str = ""
    _mode: EventKind = "answer"
    _voice_pending: str = ""  # voice content held back until </voice> is seen
    voice_text: str = field(default="")  #: accumulated spoken-summary text
    thinking_text: str = field(default="")  #: accumulated reasoning text
    answer_text: str = field(default="")  #: accumulated user-visible answer

    # The longest tag we might be mid-way through receiving.
    _MAX_TAG = len("</voice >")

    def feed(self, delta: str) -> List[StreamEvent]:
        """Consume one content delta, return the events it completed."""
        self._buf += delta or ""
        events: List[StreamEvent] = []
        while True:
            progressed = False
            if self._mode == "answer":
                progressed = self._scan_answer(events)
            elif self._mode == "thinking":
                progressed = self._scan_thinking(events)
            elif self._mode == "voice":
                progressed = self._scan_voice(events)
            if not progressed:
                break
        return events

    def flush(self) -> List[StreamEvent]:
        """Emit whatever remains after the stream ends.

        An unclosed ``<voice>`` is treated as the answer body (NEVER spoken) —
        the safe failure mode for a forgotten closer.
        """
        events: List[StreamEvent] = []
        if self._mode == "thinking":
            if self._buf:
                self.thinking_text += self._buf
                events.append(StreamEvent("thinking", self._buf))
        elif self._mode == "voice":
            # Forgotten closer: surface the held-back voice + buffer as ANSWER.
            leftover = self._voice_pending + self._buf
            self._voice_pending = ""
            if leftover:
                self.answer_text += leftover
                events.append(StreamEvent("answer", leftover))
        elif self._buf:
            self.answer_text += self._buf
            events.append(StreamEvent("answer", self._buf))
        self._buf = ""
        self._mode = "answer"
        return events

    # -- internals ---------------------------------------------------------
    def _scan_answer(self, events: List[StreamEvent]) -> bool:
        ot = _OPEN_THINK.search(self._buf)
        ov = _OPEN_VOICE.search(self._buf)
        nxt = min([m for m in (ot, ov) if m], key=lambda m: m.start(), default=None)
        if nxt is None:
            # No opener. Emit all but a possible partial-tag tail.
            return self._emit_safe_prefix(events, "answer")
        before = self._buf[: nxt.start()]
        if before:
            self.answer_text += before
            events.append(StreamEvent("answer", before))
        self._mode = "thinking" if nxt.re is _OPEN_THINK else "voice"
        self._buf = self._buf[nxt.end():]
        return True

    def _scan_thinking(self, events: List[StreamEvent]) -> bool:
        # Reasoning streams incrementally (clients show it live).
        m = _CLOSE_THINK.search(self._buf)
        if m is None:
            return self._emit_safe_prefix(events, "thinking")
        inner = self._buf[: m.start()]
        if inner:
            self.thinking_text += inner
            events.append(StreamEvent("thinking", inner))
        events.append(StreamEvent("thinking", "", final=True))
        self._buf = self._buf[m.end():]
        self._mode = "answer"
        return True

    def _scan_voice(self, events: List[StreamEvent]) -> bool:
        # Voice is BUFFERED until </voice>; emitted only on close (or, if never
        # closed, redirected to the answer at flush). This guarantees a partial
        # or forgotten voice block is never handed to TTS.
        m = _CLOSE_VOICE.search(self._buf)
        if m is None:
            # Move all but a possible partial closer into the pending buffer.
            keep = 0
            idx = self._buf.rfind("<", max(0, len(self._buf) - self._MAX_TAG))
            if idx != -1 and self._is_partial_tag(self._buf[idx:]):
                keep = len(self._buf) - idx
            move = self._buf if keep == 0 else self._buf[:-keep]
            if move:
                self._voice_pending += move
                self._buf = self._buf[len(move):]
            return False
        full = self._voice_pending + self._buf[: m.start()]
        self._voice_pending = ""
        self.voice_text += full
        # One event carries the whole spoken summary, marked final (speakable).
        events.append(StreamEvent("voice", full, final=True))
        self._buf = self._buf[m.end():]
        self._mode = "answer"
        return True

    # Canonical tag spellings a trailing '<...' fragment might be growing into.
    _TAGS = ("<think>", "</think>", "<voice>", "</voice>")

    @classmethod
    def _is_partial_tag(cls, frag: str) -> bool:
        """True if ``frag`` (starting with '<') is a non-empty prefix of one of
        the channel tags — i.e. a tag cut by a chunk boundary."""
        f = frag.replace(" ", "").lower()
        return any(t.startswith(f) and f != t for t in cls._TAGS)

    def _emit_safe_prefix(self, events: List[StreamEvent], kind: EventKind) -> bool:
        """Emit buffer content except a trailing fragment that may be a partial
        tag (``<``, ``<v``, ``</voi`` ...). Returns False — no further progress
        is possible until the next ``feed`` supplies the rest of the tag."""
        if not self._buf:
            return False
        keep = 0
        idx = self._buf.rfind("<", max(0, len(self._buf) - self._MAX_TAG))
        if idx != -1 and self._is_partial_tag(self._buf[idx:]):
            keep = len(self._buf) - idx
        emit = self._buf if keep == 0 else self._buf[:-keep]
        if emit:
            if kind == "thinking":
                self.thinking_text += emit
            elif kind == "voice":
                self.voice_text += emit
            else:
                self.answer_text += emit
            events.append(StreamEvent(kind, emit))
            self._buf = self._buf[len(emit):]
        return False
