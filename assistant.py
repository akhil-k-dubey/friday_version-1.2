"""
assistant.py — FRIDAY agent controller.

Upgrades:
  - Skill dispatch: intercepts [ACTION:...] tags and executes skills
  - Less redundant hard-coded responses (let LLM handle most things)
  - Conversation mode tracking
  - Optional always-on mode (skip standby)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import re

from brain import Brain
from memory import ShortMemory, PersistentMemory
from skills import SkillEngine


@dataclass
class Friday:
    """Voice agent — routes natural language to llama3 via Brain, dispatches skills."""

    name: str = "friday"
    awake: bool = True
    activated: bool = False
    always_on: bool = False
    memory: ShortMemory = field(default_factory=ShortMemory)
    persistent_memory: PersistentMemory = field(default_factory=PersistentMemory)
    brain: Optional[Brain] = None
    skills: SkillEngine = field(default_factory=SkillEngine)

    def __post_init__(self):
        if self.always_on:
            self.activated = True

        # Wire skill prompt into brain
        if self.brain:
            self.brain.skill_prompt = self.skills.get_skill_prompt()
            ctx = self.persistent_memory.get_context_summary()
            if ctx:
                self.brain.memory_context = ctx

    def standby(self) -> str:
        if self.always_on:
            self.activated = True
            return f"{self.name.title()} online and listening."
        return f"{self.name.title()} in standby. Say 'wake up {self.name}' to activate."

    def wake(self) -> str:
        self.activated = True
        now = datetime.now().strftime("%I:%M %p")
        return f"{self.name.title()} online. Systems restored at {now}."

    def handle(self, user_input: str) -> str:
        """
        Handle user input — only for hard commands.
        Everything else should go through the LLM streaming path in main.py.
        """
        message = user_input.strip()
        if not message:
            return "I am listening."

        normalized = message.lower()
        cleaned = re.sub(r"[^a-z0-9\s]", " ", normalized)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Store in short term memory
        self.memory.add(message)

        # ── Hard commands — these never go to LLM ───────────────────────────

        if any(w in normalized for w in {"exit", "quit", "shutdown"}):
            self.awake = False
            return "Powering down. Until next time."

        if cleaned in {"help", "commands"}:
            return (
                "I can help with many things. Ask me to search the web, "
                "open apps, check system status, set timers, do math, "
                "get weather, take screenshots, and much more. "
                "Say 'list skills' to see all capabilities. "
                "Or just talk to me naturally."
            )

        if cleaned in {"list skills", "show skills", "what can you do"}:
            return self.skills.list_skills()

        if cleaned in {"status", "system status"}:
            from skills import skill_system_info
            return skill_system_info("")

        if cleaned in {"memory", "show memory", "short memory"}:
            if self.memory.count() == 0:
                return "Memory is empty."
            return "Recent memory: " + " | ".join(self.memory.recent(5))

        if cleaned in {
            "clear memory", "delete memory",
            "reset memory", "clean memory",
        }:
            self.memory.clear()
            if self.brain is not None:
                self.brain.clear_history()
            self.persistent_memory.clear_session()
            return "Memory cleared."

        if cleaned in {"time", "current time"}:
            return "Current time: " + datetime.now().strftime("%I:%M %p, %A %d %B %Y")

        # ── Everything else → LLM (handled by main.py streaming path) ────

        if self.brain is not None and self.brain.enabled:
            llm_reply = self.brain.generate(
                message,
                memory_hint=" | ".join(self.memory.recent(5))
            )
            if llm_reply:
                # Check for skill actions in the response
                cleaned_reply, skill_results = self.skills.extract_and_execute(llm_reply)
                if skill_results:
                    # Combine LLM text with skill results
                    combined = cleaned_reply
                    for result in skill_results:
                        if result:
                            combined += " " + result
                    return combined.strip()
                return cleaned_reply
            return llm_reply or "I didn't catch that. Could you rephrase?"

        return (
            "My brain is offline. Make sure Ollama is running, "
            "then restart me."
        )

    def process_skill_actions(self, response: str) -> tuple[str, list[str]]:
        """
        Extract and execute any [ACTION:...] tags in an LLM response.
        Returns (cleaned_response, list_of_skill_results).
        """
        return self.skills.extract_and_execute(response)