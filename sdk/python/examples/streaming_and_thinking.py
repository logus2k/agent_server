#!/usr/bin/env python3
"""Streaming + thinking on/off, and separating the reasoning channel.

Shows:
  * live streaming of the ANSWER channel only (reasoning kept separate);
  * the spoken-summary channel handed to a (mock) TTS sink, sanitised;
  * toggling thinking generation on/off per request (--no-think).

Run:
    python streaming_and_thinking.py "Explain his Vision-Box work"
    python streaming_and_thinking.py --no-think "2+2?"
"""
import os
import sys

from agent_server_sdk import AgentServerClient, sanitize_for_tts

URL = os.environ.get("AGENT_SERVER_URL", "http://localhost:7701")
AGENT = os.environ.get("AGENT", "cv_assistant")


def main() -> None:
    args = sys.argv[1:]
    thinking = True
    if args and args[0] == "--no-think":
        thinking = False
        args = args[1:]
    question = " ".join(args) or "Tell me about his career."

    with AgentServerClient(URL) as ac:
        # The active model's family decides the right thinking kwarg; pass it so
        # granite/ministral are handled correctly too.
        active = ac.active_model()
        family = active.family if active else ""

        reasoning, voice, voice_done = [], [], False
        print(f"# thinking={'on' if thinking else 'off'}  answer stream:\n")
        for ev in ac.chat_stream(AGENT, question, thinking=thinking, thinking_family=family):
            if ev.kind == "answer":
                sys.stdout.write(ev.text)
                sys.stdout.flush()
            elif ev.kind == "thinking":
                reasoning.append(ev.text)
            elif ev.kind == "voice":
                voice.append(ev.text)            # fragments accumulate
                if ev.final:
                    voice_done = True            # spoken summary complete

        print("\n")
        if voice_done:
            # In a real client this is the ONLY text you'd send to TTS.
            print("TTS would speak:", repr(sanitize_for_tts("".join(voice))))
        if reasoning and thinking:
            print("\n--- reasoning (collapse/hide in your UI) ---")
            print("".join(reasoning))


if __name__ == "__main__":
    main()
