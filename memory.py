"""
memory.py — Short-term + long-term persistent memory for FRIDAY.

ShortMemory    : in-session rolling string buffer (for memory_hint).
PersistentMemory: saves facts + full conversation history to disk.
                  On init, loads last N turns so Friday remembers
                  previous sessions without any extra infrastructure.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


# ── Paths ──────────────────────────────────────────────────────────────────

_DEFAULT_DIR   = os.path.join(os.path.expanduser("~"), ".friday")
_MEMORY_FILE   = os.path.join(_DEFAULT_DIR, "memory.json")
_HISTORY_FILE  = os.path.join(_DEFAULT_DIR, "history.jsonl")

# How many past turns to reload into conversation_history at startup
_RELOAD_TURNS  = 20


# ── Short-term memory (rolling string buffer for memory_hint) ──────────────

@dataclass
class ShortMemory:
    """In-session string buffer — passed as memory_hint to the LLM."""
    messages: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.messages.append(message)

    def clear(self) -> None:
        self.messages.clear()

    def recent(self, count: int = 5) -> list[str]:
        return self.messages[-count:]

    def count(self) -> int:
        return len(self.messages)


# ── Persistent memory ──────────────────────────────────────────────────────

@dataclass
class PersistentMemory:
    """
    Long-term memory persisted to ~/.friday/.

    On startup loads the last _RELOAD_TURNS conversation turns from
    history.jsonl so Friday can resume context across sessions.

    get_history_for_llm() returns those turns formatted as
    [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]
    ready to prepend into brain.conversation_history.
    """
    memory_dir:   str = _DEFAULT_DIR
    memory_file:  str = _MEMORY_FILE
    history_file: str = _HISTORY_FILE

    user_facts:      Dict[str, str]  = field(default_factory=dict)
    preferences:     Dict[str, str]  = field(default_factory=dict)
    session_log:     List[dict]      = field(default_factory=list)
    max_session_log: int             = 200

    # Turns loaded from previous sessions — injected into LLM history
    _loaded_history: List[dict] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        os.makedirs(self.memory_dir, exist_ok=True)
        self._load_facts()
        self._load_history()

    # ── Disk I/O ──────────────────────────────────────────────────────────

    def _load_facts(self) -> None:
        """Load persisted facts/preferences from memory.json."""
        if not os.path.exists(self.memory_file):
            return
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.user_facts  = data.get("user_facts",  {})
            self.preferences = data.get("preferences", {})
        except (json.JSONDecodeError, IOError):
            pass

    def _save_facts(self) -> None:
        """Persist facts/preferences to memory.json."""
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump({
                    "user_facts":    self.user_facts,
                    "preferences":   self.preferences,
                    "last_updated":  datetime.now().isoformat(),
                }, f, indent=2, ensure_ascii=False)
        except IOError:
            pass

    def _load_history(self) -> None:
        """
        Read the last _RELOAD_TURNS turns from history.jsonl.
        Skips partial/interrupted entries (assistant starts with '[partial]').
        """
        if not os.path.exists(self.history_file):
            return
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except IOError:
            return

        turns = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            user_text = entry.get("user", "")
            asst_text = entry.get("assistant", "")
            # Skip skill results, partial answers, and empty turns
            if user_text.startswith("[skill_result]"):
                continue
            if asst_text.startswith("[partial]"):
                continue
            if not user_text or not asst_text:
                continue
            turns.append(entry)
            if len(turns) >= _RELOAD_TURNS:
                break

        # Restore chronological order
        self._loaded_history = list(reversed(turns))

    # ── LLM history injection ──────────────────────────────────────────────

    def get_history_for_llm(self) -> List[dict]:
        """
        Return loaded history formatted for brain.conversation_history.
        Call this once at startup and prepend to brain.conversation_history.
        """
        messages = []
        for entry in self._loaded_history:
            messages.append({"role": "user",      "content": entry["user"]})
            messages.append({"role": "assistant",  "content": entry["assistant"]})
        return messages

    # ── Interaction logging ────────────────────────────────────────────────

    def log_interaction(self, user_input: str, assistant_response: str) -> None:
        """Log a turn to session_log (in-memory) and append to history.jsonl."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user":      user_input,
            "assistant": assistant_response,
        }
        self.session_log.append(entry)
        if len(self.session_log) > self.max_session_log:
            self.session_log = self.session_log[-self.max_session_log:]

        try:
            with open(self.history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except IOError:
            pass

        # Auto-extract facts from the conversation
        self._auto_extract_facts(user_input, assistant_response)

    # ── Fact auto-extraction ───────────────────────────────────────────────

    _FACT_PATTERNS = [
        (r"my name is ([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)",  "name"),
        (r"i(?:'m| am) ([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", "name"),
        (r"i(?:'m| am) (\d+) years? old",                 "age"),
        (r"i(?:'m| am) a ([a-z]+ (?:student|engineer|developer|designer|researcher))",
                                                          "profession"),
        (r"i (?:live|am) in ([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", "location"),
        (r"i(?:'m| am) from ([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", "location"),
        (r"i (?:like|love|prefer) ([a-z]+(?:\s[a-z]+)?)", "preference"),
    ]

    def _auto_extract_facts(self, user_text: str, _asst: str) -> None:
        """Scan user text for self-identifying statements and save as facts."""
        text_lower = user_text.lower()
        changed = False
        for pattern, key in self._FACT_PATTERNS:
            m = re.search(pattern, text_lower)
            if m:
                value = m.group(1).strip()
                if value and self.user_facts.get(key) != value:
                    self.user_facts[key] = value
                    changed = True
        if changed:
            self._save_facts()

    # ── Fact/preference API ────────────────────────────────────────────────

    def save_fact(self, key: str, value: str) -> None:
        self.user_facts[key.strip().lower()] = value.strip()
        self._save_facts()

    def get_fact(self, key: str) -> Optional[str]:
        return self.user_facts.get(key.strip().lower())

    def save_preference(self, key: str, value: str) -> None:
        self.preferences[key.strip().lower()] = value.strip()
        self._save_facts()

    def get_preference(self, key: str) -> Optional[str]:
        return self.preferences.get(key.strip().lower())

    # ── System prompt context ──────────────────────────────────────────────

    def get_context_summary(self) -> str:
        """
        Build a short context string for the LLM system prompt.
        Includes known facts and recent session context.
        """
        parts = []
        if self.user_facts:
            facts = "; ".join(
                f"{k}: {v}" for k, v in list(self.user_facts.items())[:10]
            )
            parts.append(f"Known facts about user: {facts}")
        if self.preferences:
            prefs = "; ".join(
                f"{k}: {v}" for k, v in list(self.preferences.items())[:5]
            )
            parts.append(f"User preferences: {prefs}")
        if self._loaded_history:
            parts.append(
                f"Note: {len(self._loaded_history)} previous conversation turns "
                f"have been loaded into context from past sessions."
            )
        return ". ".join(parts) if parts else ""

    # ── Cleanup ────────────────────────────────────────────────────────────

    def clear_facts(self) -> None:
        self.user_facts.clear()
        self.preferences.clear()
        self._save_facts()

    def clear_session(self) -> None:
        self.session_log.clear()

    def recent_history(self, count: int = 5) -> List[dict]:
        """Last N turns from this session."""
        return self.session_log[-count:]

    def clear_all_history(self) -> None:
        """Wipe the history file (useful for a fresh start)."""
        try:
            open(self.history_file, "w").close()
        except IOError:
            pass
        self._loaded_history = []
        self.session_log = []
