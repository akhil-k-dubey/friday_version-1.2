"""
agentic.py — FRIDAY Agentic Task Loop (Phase 4)

Architecture:
  User task (multi-step request)
      → Plan   : LLM breaks task into ordered steps
      → Execute : run each step's skill/code action
      → Verify  : check output, decide retry / next step / done
      → Done   : speak concise summary

Safety:
  - Max 8 steps per task (hard ceiling)
  - Confirmation gate: destructive ops (file_write, run_code with mutations)
    ask the user before executing
  - Results truncated to 500 chars before being spoken
  - All skill errors are caught and reported cleanly — never crash the loop
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Any

from skills import SkillEngine


# ── Step status constants ──────────────────────────────────────────────────
STEP_OK      = "ok"
STEP_RETRY   = "retry"
STEP_FAIL    = "fail"
STEP_SKIP    = "skip"
STEP_CONFIRM = "needs_confirm"
STEP_PENDING = "pending"

# ── Destructive skills that require user confirmation ──────────────────────
_DESTRUCTIVE_SKILLS = {"file_write", "run_code", "close_app"}

# Max retries per step before marking it failed
_MAX_STEP_RETRIES = 2

# Max total steps per agentic task
MAX_STEPS = 8

# Result truncation for voice
_RESULT_VOICE_LIMIT = 500


def _truncate_voice(text: str) -> str:
    """Trim result to voice-safe length."""
    text = text.strip()
    if len(text) > _RESULT_VOICE_LIMIT:
        return text[:_RESULT_VOICE_LIMIT] + "… (truncated)"
    return text


@dataclass
class AgentStep:
    """One planned step in the agentic task."""
    index:       int
    description: str         # human-readable description
    skill:       str         # skill name to call
    args:        str         # arguments for the skill
    result:      str  = ""
    status:      str  = STEP_PENDING
    retries:     int  = 0
    hint:        str  = ""


@dataclass
class AgenticTaskLoop:
    """
    Agentic execution loop for multi-step tasks.

    Usage:
        loop = AgenticTaskLoop(
            brain=brain,
            skill_engine=skill_engine,
            speak_fn=voice.speak,
            confirm_fn=get_user_input,    # callable that returns user text
            debug=True,
        )
        summary = loop.run("create a python hello world file and run it")

    The loop:
        1. PLAN   — sends user task to LLM asking for a JSON step list
        2. EXECUTE — for each step, calls the corresponding skill
        3. VERIFY  — feeds result back to LLM: ok / retry / next / done?
        4. DONE   — collects all results, asks LLM for a short summary
    """

    brain:       Any                      # Brain instance
    skill_engine: SkillEngine
    speak_fn:    Callable[[str], None]    # voice.speak
    print_fn:    Callable[[str], None] = print
    confirm_fn:  Optional[Callable[[str], Optional[str]]] = None  # prompts user
    debug:       bool = False

    def _log(self, msg: str) -> None:
        if self.debug:
            self.print_fn(f"[agentic] {msg}")

    # ── Phase 1: PLAN ─────────────────────────────────────────────────────

    def _plan(self, task: str) -> List[AgentStep]:
        """
        Ask the LLM to decompose the task into ordered steps.
        Returns a list of AgentStep objects.
        Falls back to a single skill step if planning fails.
        """
        plan_prompt = (
            f"You are FRIDAY's task planner. Break this task into steps:\n"
            f"TASK: {task}\n\n"
            f"Available skills: "
            f"file_read, file_write, run_code, git_query, "
            f"open_app, close_app, screenshot, web_search, "
            f"system_info, calculator, weather, datetime, timer, "
            f"smart_open, youtube_search, browser_search, find_and_open, "
            f"clipboard_read, clipboard_write, volume, remind\n\n"
            f"Reply ONLY with a JSON array of steps. Each step:\n"
            f'  {{"step": 1, "description": "what this step does", '
            f'"skill": "skill_name", "args": "skill arguments"}}\n\n'
            f"Rules:\n"
            f"- Max {MAX_STEPS} steps\n"
            f"- For file_write: args = path||content\n"
            f"- For run_code: args = python:code_here\n"
            f"- For file_read: args = path\n"
            f"- For git_query: args = status | diff | log | branch\n"
            f"- Use only ONE skill per step\n"
            f"- If the task needs no skills (pure Q&A), return an empty array []\n"
            f"Output ONLY the JSON array, nothing else."
        )

        self._log(f"Planning task: {task[:80]}")
        raw = self.brain.generate_simple(plan_prompt, max_tokens=600, temperature=0.2)
        if not raw:
            self._log("Plan: LLM returned empty")
            return []

        # Extract JSON array using json.JSONDecoder (handles nested brackets in args)
        start_idx = raw.find('[')
        if start_idx == -1:
            self._log(f"Plan: no JSON array found in: {raw[:200]}")
            return []

        try:
            decoder = json.JSONDecoder()
            steps_raw, _ = decoder.raw_decode(raw, start_idx)
        except json.JSONDecodeError as exc:
            self._log(f"Plan: JSON parse failed: {exc}")
            return []

        steps = []
        for i, s in enumerate(steps_raw[:MAX_STEPS]):
            steps.append(AgentStep(
                index=i + 1,
                description=str(s.get("description", f"Step {i+1}")),
                skill=str(s.get("skill", "")).strip(),
                args=str(s.get("args", "")).strip(),
            ))

        self._log(f"Plan: {len(steps)} steps parsed")
        return steps

    # ── Phase 2: EXECUTE ──────────────────────────────────────────────────

    def _needs_confirmation(self, step: AgentStep) -> bool:
        """Return True if this step touches destructive actions."""
        return step.skill in _DESTRUCTIVE_SKILLS

    def _confirm_with_user(self, step: AgentStep) -> bool:
        """
        Ask the user for confirmation before a destructive step.
        Returns True if confirmed, False if denied/skipped.
        """
        prompt = (
            f"Step {step.index}: {step.description}. "
            f"This will use '{step.skill}' — shall I proceed? Say yes or no."
        )
        self.speak_fn(prompt)
        self.print_fn(f"[agentic] Confirm: {prompt}")

        if self.confirm_fn is None:
            return True   # no confirm channel — auto-approve

        response = self.confirm_fn("Confirm > ")
        if response is None:
            return False
        return response.strip().lower() in {"yes", "y", "yeah", "yep", "proceed", "do it", "ok", "okay"}

    def _execute_step(self, step: AgentStep) -> str:
        """Run the skill for this step. Returns result string."""
        fn = self.skill_engine.skills.get(step.skill)
        if fn is None:
            return f"Unknown skill: {step.skill}"
        try:
            result = fn(step.args)
            return _truncate_voice(result or "(no output)")
        except Exception as exc:
            return f"Skill error: {exc}"

    # ── Phase 3: VERIFY ───────────────────────────────────────────────────

    def _verify(self, step: AgentStep, task: str, results_so_far: List[str]) -> str:
        """
        Ask the LLM: given this step's output, what should happen next?
        Returns: 'ok' | 'retry:<hint>' | 'fail:<reason>' | 'done'
        """
        context = "\n".join(
            f"Step {i+1} result: {r[:200]}" for i, r in enumerate(results_so_far)
        )
        verify_prompt = (
            f"You are verifying a step in an agentic task.\n"
            f"TASK: {task}\n"
            f"Current step ({step.index}): {step.description}\n"
            f"Step result: {step.result[:300]}\n"
            f"Previous results:\n{context}\n\n"
            f"Is this step successful? Reply with exactly one of:\n"
            f"  OK          — step succeeded, continue to next\n"
            f"  RETRY:<hint> — step failed but retryable, hint = what to change\n"
            f"  FAIL:<reason> — unrecoverable failure\n"
            f"  DONE        — task is fully complete (no more steps needed)\n"
            f"Reply ONLY with one of the above, nothing else."
        )

        verdict = self.brain.generate_simple(verify_prompt, max_tokens=80, temperature=0.1)
        if not verdict:
            return STEP_OK   # assume ok if LLM silent

        verdict = verdict.strip().upper()
        self._log(f"Verify step {step.index}: {verdict[:60]}")

        if verdict.startswith("DONE"):
            return "done"
        if verdict.startswith("OK"):
            return STEP_OK
        if verdict.startswith("RETRY"):
            hint = verdict[6:].strip(": ") if ":" in verdict else ""
            return f"retry:{hint}"
        if verdict.startswith("FAIL"):
            reason = verdict[5:].strip(": ") if ":" in verdict else "unknown"
            return f"fail:{reason}"
        return STEP_OK   # default

    # ── Phase 4: DONE / SUMMARY ───────────────────────────────────────────

    def _summarize(self, task: str, steps: List[AgentStep]) -> str:
        """Ask LLM for a short voice-friendly summary of what was done."""
        executed  = [s for s in steps if s.status != STEP_PENDING]
        completed = [s for s in steps if s.status in (STEP_OK, "done")]
        failed    = [s for s in steps if s.status == STEP_FAIL]

        steps_detail = "\n".join(
            f"Step {s.index} ({s.skill}): {s.result[:150]}"
            for s in executed
        )
        summary_prompt = (
            f"Summarize what FRIDAY accomplished for the user in 1-2 short sentences.\n"
            f"TASK: {task}\n"
            f"Steps completed: {len(completed)}/{len(executed)}\n"
            f"Results:\n{steps_detail}\n\n"
            f"Rules: no markdown, no lists, speak naturally, be concise."
        )
        summary = self.brain.generate_simple(summary_prompt, max_tokens=120, temperature=0.4)
        if not summary:
            if failed:
                return f"I completed {len(completed)} of {len(executed)} steps. Some steps could not be finished."
            return f"Done. Completed {len(completed)} step{'s' if len(completed) != 1 else ''}."
        return summary.strip()

    # ── Main run loop ──────────────────────────────────────────────────────

    def run(self, task: str) -> str:
        """
        Execute the full Plan→Execute→Verify→Done agentic loop.
        Returns a summary string (also spoken via speak_fn).
        """
        self.print_fn(f"\n[agentic] ─── Starting agentic task ───")
        self.print_fn(f"[agentic] Task: {task}")

        # ── Plan ──────────────────────────────────────────────────────────
        self.speak_fn("Understood. Let me plan this out.")
        steps = self._plan(task)

        if not steps:
            self._log("No steps planned — falling back to direct answer")
            return ""   # caller will handle as normal LLM response

        self.print_fn(f"[agentic] Plan: {len(steps)} steps")
        plan_preview = ". ".join(
            f"Step {s.index}: {s.description}" for s in steps[:3]
        )
        if len(steps) > 3:
            plan_preview += f". And {len(steps)-3} more steps."
        self.speak_fn(f"I have a {len(steps)}-step plan. {plan_preview}")

        # ── Execute + Verify loop ─────────────────────────────────────────
        results_so_far: List[str] = []
        early_done = False

        for step in steps:
            self.print_fn(f"\n[agentic] Step {step.index}/{len(steps)}: {step.description}")
            self.print_fn(f"[agentic]   skill={step.skill}  args={step.args[:80]}")
            self.speak_fn(f"Step {step.index}: {step.description}.")

            # Confirmation gate for destructive ops
            if self._needs_confirmation(step):
                confirmed = self._confirm_with_user(step)
                if not confirmed:
                    step.status = STEP_SKIP
                    step.result = "Skipped by user."
                    results_so_far.append(step.result)
                    self.speak_fn(f"Skipping step {step.index}.")
                    continue

            # Execute with retry support
            for attempt in range(_MAX_STEP_RETRIES + 1):
                step.result = self._execute_step(step)
                self.print_fn(f"[agentic]   result: {step.result[:120]}")

                # Verify
                verdict = self._verify(step, task, results_so_far)

                if verdict == "done":
                    step.status = STEP_OK
                    results_so_far.append(step.result)
                    early_done = True
                    break
                elif verdict == STEP_OK:
                    step.status = STEP_OK
                    results_so_far.append(step.result)
                    break
                elif verdict.startswith("retry") and attempt < _MAX_STEP_RETRIES:
                    hint = verdict[6:] if ":" in verdict else ""
                    self._log(f"  Retrying (attempt {attempt+1}): {hint}")
                    step.retries += 1
                    if hint:
                        step.hint = hint
                elif verdict.startswith("fail") or attempt >= _MAX_STEP_RETRIES:
                    reason = verdict[5:] if ":" in verdict else "unknown"
                    step.status = STEP_FAIL
                    step.result = f"Failed: {reason}"
                    results_so_far.append(step.result)
                    self.speak_fn(f"Step {step.index} could not be completed.")
                    break

            if early_done:
                self._log("Early exit: task complete before all steps ran")
                break

        # ── Summary ───────────────────────────────────────────────────────
        self.print_fn(f"\n[agentic] ─── Task complete ───")
        summary = self._summarize(task, steps)
        self.print_fn(f"[agentic] Summary: {summary}")
        self.speak_fn(summary)
        return summary


# ── Agentic task detector ──────────────────────────────────────────────────

_AGENTIC_PATTERNS = [
    r'\b(create|write|make|build)\s+.+\s+(and|then)\s+(run|execute|test|open)',
    r'\b(find|search)\s+.+\s+(and|then)\s+(open|read|show|edit)',
    r'\bstep by step\b',
    r'\bmultiple steps?\b',
    r'\b(first|then|after that|finally)\b.{10,}',
    r'\b(read|open)\s+.+\s+(and|then)\s+(edit|modify|change|fix|write)',
    r'\bgit\s+(status|diff|log)\s+(and|then)',
    r'\brun\s+(the\s+)?(code|script|file)\b',
    r'\bcheck\s+.+\s+(and|then)\s+(fix|update|change)',
    r'\bdo\s+everything\b',
    r'\bautomate\b',
]

_AGENTIC_RE = re.compile("|".join(_AGENTIC_PATTERNS), re.IGNORECASE)


def is_agentic_task(user_input: str) -> bool:
    """
    Heuristic: does this user request require multi-step agentic execution?
    Returns True if the input matches agentic patterns.
    """
    return bool(_AGENTIC_RE.search(user_input))
