"""
commands.py — Wake word and shutdown command detection.

Upgrades:
  - More natural shutdown phrases
  - More wake word variants
  - Fuzzy matching with better thresholds
"""
from __future__ import annotations

from difflib import SequenceMatcher
import re


def is_shutdown_command(text: str) -> bool:
    """Detect shutdown intent from user input."""
    normalized = text.strip().lower()
    # Exact matches
    if normalized in {
        "exit", "quit", "shutdown", "shut down",
        "power off", "power down", "goodbye", "good bye",
        "bye friday", "bye bye", "go to sleep",
        "turn off", "stop", "end session",
    }:
        return True
    # Partial matches
    return any(word in normalized for word in {
        "shutdown", "shut down", "power off", "power down",
    })


def is_wake_command(text: str, assistant_name: str) -> bool:
    """Detect wake word from user input."""
    normalized = text.strip().lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", normalized)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    name = assistant_name.lower()

    def name_matches(spoken_name: str) -> bool:
        candidate = spoken_name.strip()
        if not candidate:
            return False
        if candidate == name:
            return True
        return SequenceMatcher(None, candidate, name).ratio() >= 0.70

    # Wake phrases
    prefixes = ["wake up ", "hey ", "hello ", "hi ", "yo ", "okay "]
    suffixes = [" come online", " wake up", " start", " activate"]

    # Check prefixes
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            return name_matches(cleaned[len(prefix):])

    # Check suffixes
    for suffix in suffixes:
        if cleaned.endswith(suffix):
            return name_matches(cleaned[:-len(suffix)])

    # Just the name alone
    if name_matches(cleaned):
        return True

    return False
