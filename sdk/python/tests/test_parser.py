"""Tests for the channel parser + TTS sanitiser.

Run: pytest -q   (from sdk/python/)
"""
import random

from agent_server_sdk.parser import StreamParser, split_response, sanitize_for_tts


def _stream(parser: StreamParser, text: str, chunk: int = 1):
    """Feed `text` to `parser` in fixed-size chunks, return collected events."""
    events = []
    for i in range(0, len(text), chunk):
        events.extend(parser.feed(text[i:i + chunk]))
    events.extend(parser.flush())
    return events


def _channels(events):
    out = {"thinking": "", "voice": "", "answer": ""}
    for ev in events:
        out[ev.kind] += ev.text
    return out


def test_split_complete():
    t, v, a = split_response("<think>reasoning here</think><voice>spoken gist</voice>The answer body.")
    assert t == "reasoning here"
    assert v == "spoken gist"
    assert a == "The answer body."


def test_split_no_blocks():
    t, v, a = split_response("Just a plain answer.")
    assert (t, v) == ("", "")
    assert a == "Just a plain answer."


def test_stream_char_by_char_matches_split():
    text = "<think>thinking</think><voice>say this</voice>Visible answer with detail."
    for chunk in (1, 2, 3, 7, 1000):
        p = StreamParser()
        ch = _channels(_stream(p, text, chunk))
        assert ch["thinking"] == "thinking", chunk
        assert ch["voice"] == "say this", chunk
        assert ch["answer"] == "Visible answer with detail.", chunk
        assert p.voice_text == "say this"


def test_voice_final_event_marks_complete():
    p = StreamParser()
    evs = _stream(p, "<voice>hi</voice>body")
    finals = [e for e in evs if e.kind == "voice" and e.final]
    assert len(finals) == 1


def test_unclosed_voice_becomes_answer_never_spoken():
    # Forgotten </voice>: the safe failure mode is to surface it as ANSWER,
    # so TTS is never fed the whole body.
    p = StreamParser()
    ch = _channels(_stream(p, "<voice>this never closes and runs into the body"))
    assert ch["voice"] == ""             # nothing committed to the voice channel
    assert "runs into the body" in ch["answer"]


def test_partial_tag_across_boundary_not_leaked():
    # '<voice>' split across two feeds must not leak '<voi' as answer text.
    p = StreamParser()
    evs = p.feed("Answer <voi")
    evs += p.feed("ce>spoken</voice>more")
    evs += p.flush()
    ch = _channels(evs)
    assert "<voi" not in ch["answer"]
    assert ch["voice"] == "spoken"
    assert ch["answer"] == "Answer more"


def test_sanitize_strips_malformed_closers():
    assert sanitize_for_tts("António was CTO.</voice>") == "António was CTO."
    assert sanitize_for_tts("António was CTO.</ voice>") == "António was CTO."
    assert sanitize_for_tts("António was CTO.</voice >") == "António was CTO."
    assert sanitize_for_tts("António was CTO.</voice") == "António was CTO."


def test_sanitize_preserves_legit_text():
    assert sanitize_for_tts("He processed every invoice on time.") == "He processed every invoice on time."
    assert sanitize_for_tts("António led the Orchestra Platform.") == "António led the Orchestra Platform."


def test_sanitize_strips_think_and_citations():
    s = sanitize_for_tts("<think>secret</think>Answer [markdown_chunk:ab12] here.")
    assert "secret" not in s and "markdown_chunk" not in s
    assert "Answer" in s and "here" in s


def test_fuzz_random_chunking_is_stable():
    text = "<think>r</think><voice>v</voice>answer-text"
    rng = random.Random(7)
    for _ in range(50):
        p = StreamParser()
        evs, i = [], 0
        while i < len(text):
            n = rng.randint(1, 5)
            evs += p.feed(text[i:i + n])
            i += n
        evs += p.flush()
        ch = _channels(evs)
        assert ch == {"thinking": "r", "voice": "v", "answer": "answer-text"}
