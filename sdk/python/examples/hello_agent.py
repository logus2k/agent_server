#!/usr/bin/env python3
"""Minimal 'hello agent' — the smallest correct agent_server integration.

Run (from the host, with the stack up):
    pip install -e ../        # installs agent_server_sdk
    python hello_agent.py "Who is António?"

Set AGENT_SERVER_URL / AGENT to point elsewhere.
"""
import os
import sys

from agent_server_sdk import AgentServerClient

URL = os.environ.get("AGENT_SERVER_URL", "http://localhost:7701")
AGENT = os.environ.get("AGENT", "cv_assistant")


def main() -> None:
    question = " ".join(sys.argv[1:]) or "Give me a one-sentence intro."
    with AgentServerClient(URL) as ac:
        # Discovery: what's active + what agents exist (optional, for sanity).
        active = ac.active_model()
        print(f"# active model: {active.id if active else '??'} "
              f"(family={active.family if active else '?'})\n")

        result = ac.chat(AGENT, question)
        print("ANSWER:\n" + result.answer)
        if result.thinking:
            print("\n--- reasoning (hidden channel) ---\n" + result.thinking)
        if result.voice:
            print("\n--- spoken summary ---\n" + result.voice)
        if result.usage:
            print(f"\n[tokens: {result.usage}]")


if __name__ == "__main__":
    main()
