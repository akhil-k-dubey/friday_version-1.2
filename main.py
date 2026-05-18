"""
main.py — FRIDAY Advanced AI Assistant.

Architecture:
  1. LLM streams tokens → accumulates sentences
  2. First sentence pre-gen starts immediately (latency hiding)
  3. TTS plays sentence-by-sentence with minimal gaps
  4. Skill actions are detected, executed, and results spoken
  5. Persistent memory logs every interaction

All tools are FREE FOREVER — no API keys required.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from typing import Optional

# Silence pkg_resources deprecation from webrtcvad
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

from assistant import Friday
from brain import Brain
from commands import is_shutdown_command, is_wake_command
from memory import PersistentMemory
from monitor import ProactiveMonitor
from skills import SkillEngine
from voice import VoiceIO
from agentic import AgenticTaskLoop, is_agentic_task


def run(
    default_voice_mode: bool = True,
    debug_voice: bool = False,
    whisper_model_size: str = "small.en",
    stt_language: str = "en-IN",
    llm_backend: str = "ollama",
    llm_model: str = "auto",
    llm_base_url: str = "http://127.0.0.1:11434",
    llm_timeout: float = 60.0,
    llm_enabled: bool = True,
    always_on: bool = False,
    monitor_enabled: bool = True,
    agentic_enabled: bool = True,   # NEW: enable agentic task routing
) -> None:
    # ── Initialize components ─────────────────────────────────────────────
    persistent_memory = PersistentMemory()
    skill_engine = SkillEngine(
        debug_fn=lambda m: print(f"[skill-debug] {m}") if debug_voice else None
    )

    brain = Brain(
        enabled=llm_enabled and llm_backend == "ollama",
        base_url=llm_base_url,
        model=llm_model,
        timeout_s=llm_timeout,
        debug=debug_voice,
    )

    # ── Inject skill prompt so LLM knows how to invoke tools ──────────────
    brain.skill_prompt = skill_engine.get_skill_prompt()

    # ── Inject long-term memory into LLM conversation history ────────────
    # Load last N turns from disk so Friday remembers previous sessions.
    loaded_turns = persistent_memory.get_history_for_llm()
    if loaded_turns:
        brain.conversation_history = loaded_turns
        print(f"[memory] Loaded {len(loaded_turns)//2} turns from previous sessions.")

    # Inject known facts into system prompt (name, preferences, etc.)
    ctx = persistent_memory.get_context_summary()
    if ctx:
        brain.memory_context = ctx
    brain._build_system_prompt()  # rebuild with skill prompt + memory injected

    agent = Friday(
        name="friday",
        brain=brain,
        always_on=always_on,
        persistent_memory=persistent_memory,
        skills=skill_engine,
    )

    voice = VoiceIO(
        debug=debug_voice,
        whisper_model_size=whisper_model_size,
        stt_language=stt_language,
    )
    voice.voice_mode = default_voice_mode

    # Wire TTS into skills so timers/reminders speak aloud when they fire
    import skills as _skills_mod
    _skills_mod._announce_fn = voice.speak

    # ── Agentic task loop ─────────────────────────────────────────────────
    # Provides voice.listen() as confirm_fn so agentic loop can ask user
    # for confirmation before destructive ops (file_write, run_code, etc.)
    _agentic_loop = AgenticTaskLoop(
        brain=brain,
        skill_engine=skill_engine,
        speak_fn=voice.speak,
        print_fn=print,
        confirm_fn=voice.listen if (default_voice_mode and voice.mic_enabled) else input,
        debug=debug_voice,
    ) if agentic_enabled else None

    # ── Boot messages (mic muted for entire block to prevent echo) ──────
    online_message = (
        f"{agent.name.title()} online. "
        f"Systems restored at {datetime.now().strftime('%I:%M %p')}."
    )
    print(online_message)
    voice.mute_mic()  # ← mute mic for ALL boot speech
    voice.speak(online_message)

    if not always_on:
        standby_message = agent.standby()
        print(standby_message)
        voice.speak(standby_message)
    else:
        agent.activated = True
        ready_msg = f"{agent.name.title()} is ready and listening."
        print(ready_msg)
    voice.unmute_mic()  # ← unmute + flush + settle after all boot TTS done

    # ── Start ProactiveMonitor (after voice is live so alerts can speak) ──
    monitor = ProactiveMonitor(
        announce_fn=voice.speak,
        debug=debug_voice,
    )
    if monitor_enabled:
        monitor.start()

    # ── Status line ───────────────────────────────────────────────────────
    if voice.voice_mode and voice.mic_enabled and voice.tts_enabled:
        status_line = (
            f"Voice: READY "
            f"(mic={voice.mic_backend}, STT={voice.stt_backend}, TTS={voice.tts_effective_backend})"
        )
    elif voice.voice_mode and (voice.mic_enabled or voice.tts_enabled):
        status_line = (
            "Voice: PARTIAL "
            f"(mic={voice.mic_backend}, STT={voice.stt_backend}, "
            f"TTS={voice.tts_effective_backend if voice.tts_enabled else 'off'})"
        )
    elif voice.voice_mode:
        status_line = "Voice: FALLBACK (text-only; mic/TTS unavailable)"
    else:
        status_line = "Voice: DISABLED (text-only mode)"
    print(status_line)

    model_name = brain.active_model or brain.model
    print(f"Brain: {'Ollama (' + model_name + ')' if brain.enabled else 'OFFLINE (rule-based)'}")
    print(f"Skills: {len(skill_engine.skills)} available")
    print("Say 'help' for commands or just talk naturally.\n")
    if debug_voice:
        print("[debug] Debug mode enabled.")

    # ── Main loop ─────────────────────────────────────────────────────────
    while agent.awake:
        user_input: Optional[str] = None

        # Voice input (streaming Silero VAD + whisper GPU)
        if voice.voice_mode and voice.mic_enabled:
            spoken = voice.listen()
            if spoken:
                print(f"You (voice) > {spoken}")
                user_input = spoken

        # Text input fallback (when mic unavailable or voice mode off)
        if not user_input and (not voice.voice_mode or not voice.mic_enabled):
            try:
                user_input = input("You > ")
            except (EOFError, KeyboardInterrupt):
                print("\nPowering down. Until next time.")
                break

        if not user_input or not user_input.strip():
            continue

        normalized = user_input.strip().lower()

        # ── Shutdown ──────────────────────────────────────────────────────
        if is_shutdown_command(normalized):
            response = "Powering down. Until next time."
            agent.awake = False
            print(f"{agent.name.title()} > {response}")
            voice.speak(response)
            persistent_memory.log_interaction(user_input, response)
            break

        # ── Wake word (standby mode) ──────────────────────────────────────
        if not agent.activated:
            if is_wake_command(normalized, agent.name):
                response = agent.wake()
                print(f"{agent.name.title()} > {response}")
                voice.speak(response)
            else:
                response = f"In standby. Say 'wake up {agent.name}' to activate."
                print(f"{agent.name.title()} > {response}")
                voice.speak(response)
            continue

        # ── Already active, wake word repeated ───────────────────────────
        if is_wake_command(normalized, agent.name):
            response = "I am already online and listening."
            print(f"{agent.name.title()} > {response}")
            voice.speak(response)
            continue

        # ── Voice mode toggles ────────────────────────────────────────────
        if normalized in ("voice on", "enable voice", "turn voice on"):
            voice.voice_mode = True
            response = "Voice mode enabled."
            print(f"{agent.name.title()} > {response}")
            voice.speak(response)
            continue

        if normalized in ("voice off", "disable voice", "turn voice off"):
            voice.voice_mode = False
            response = "Voice mode disabled."
            print(f"{agent.name.title()} > {response}")
            continue

        # ── LLM or rule-based response ────────────────────────────────────
        if agent.brain and agent.brain.enabled:
            # Route to agentic loop for multi-step tasks
            if _agentic_loop is not None and is_agentic_task(user_input):
                _handle_agentic_response(
                    agent, voice, user_input, persistent_memory, _agentic_loop
                )
            else:
                _handle_brain_response(agent, voice, user_input, persistent_memory)
        else:
            response = agent.handle(user_input)
            print(f"{agent.name.title()} > {response}")
            voice.speak(response)
            persistent_memory.log_interaction(user_input, response)

        if not agent.awake:
            break

    # ── Cleanup ───────────────────────────────────────────────────────────
    monitor.stop()
    voice.shutdown()
    print("FRIDAY shut down cleanly.")


def _handle_brain_response(
    agent: "Friday",
    voice: "VoiceIO",
    user_input: str,
    persistent_memory: "PersistentMemory",
) -> None:
    """
    Stream LLM response sentence-by-sentence with pipelined TTS.

    Skills are dispatched IMMEDIATELY as their [ACTION:...] tags are
    yielded by the LLM, so tasks complete before or during speech,
    not after all speech finishes.

    Barge-in recovery is ITERATIVE (not recursive) — prevents RecursionError
    and the post-barge-in crash when the user interrupts multiple times.
    """
    current_input = user_input
    MAX_BARGE_IN_CHAIN = 5  # safety cap — prevents infinite loop on stuck mic

    for _turn in range(MAX_BARGE_IN_CHAIN + 1):
        # ── Per-turn state (use dict so closure captures by reference) ────
        _state: dict = {"full_response": "", "skill_results": [], "sentences": []}

        sentence_gen = agent.brain.stream_sentences(
            current_input,
            memory_hint=" | ".join(agent.memory.recent(5)),
        )

        print(f"{agent.name.title()} > ", end="", flush=True)

        def _sentence_stream(state=_state, sg=sentence_gen):
            """Yield speakable sentences; execute skills instantly."""
            for sentence in sg:
                sentence = sentence.strip()
                if not sentence:
                    continue

                state["full_response"] = (
                    state["full_response"] + " " + sentence
                ).strip()

                # Execute skill IMMEDIATELY if [ACTION:] tag found
                if "[ACTION:" in sentence:
                    cleaned, results = agent.process_skill_actions(sentence)
                    if results:
                        for result in results:
                            print(f"\n  [Skill] {result}")
                            state["skill_results"].append(result)
                            # Truncate long error messages (file paths etc.)
                            # before TTS — prevents 7-second error reads.
                            spoken = result if len(result) <= 120 else result[:117] + "..."
                            yield spoken
                else:
                    state["sentences"].append(sentence)
                    yield sentence

        # 3-thread pipeline: LLM feeder | TTS generator | Player — all parallel
        voice.speak_pipeline(_sentence_stream())

        full_response = _state["full_response"]
        skill_results = _state["skill_results"]

        # ── Memory for this turn ─────────────────────────────────────────
        agent.memory.add(current_input)

        was_interrupted = voice._barge_in_speech_pending
        if full_response:
            if was_interrupted:
                persistent_memory.log_interaction(
                    current_input, f"[partial] {full_response}"
                )
            else:
                persistent_memory.log_interaction(current_input, full_response)

        for result in skill_results:
            if result:
                persistent_memory.log_interaction("[skill_result]", result)

        # ── Barge-in recovery: listen then loop (never recurse) ──────────
        if not was_interrupted:
            break  # normal turn complete

        recovered = voice.listen()
        voice._barge_in_speech_pending = False

        if not recovered or not recovered.strip():
            break  # barge-in but silence — done

        print(f"You (voice/barge-in) > {recovered}")
        current_input = recovered  # next iteration processes this input


def _handle_agentic_response(
    agent: "Friday",
    voice: "VoiceIO",
    user_input: str,
    persistent_memory: "PersistentMemory",
    agentic_loop: "AgenticTaskLoop",
) -> None:
    """
    Route a multi-step request through the agentic Plan→Execute→Verify→Done loop.

    If the loop returns empty (task was pure Q&A, no steps planned), falls back
    to the normal streaming brain response so the user always gets an answer.
    """
    print(f"[agentic] Routing to agentic loop: {user_input[:80]}")

    # Mute mic during agentic execution to prevent false wake words
    # from confirmation prompts being misheard as new commands.
    voice.mute_mic()
    try:
        summary = agentic_loop.run(user_input)
    except Exception as exc:
        print(f"[agentic] Loop error: {exc}")
        summary = ""
    finally:
        voice.unmute_mic()

    if not summary:
        # Agentic loop found no steps — treat as normal LLM request
        print("[agentic] No steps planned — falling back to normal response")
        _handle_brain_response(agent, voice, user_input, persistent_memory)
        return

    # Log the agentic task + summary to persistent memory
    agent.memory.add(user_input)
    persistent_memory.log_interaction(user_input, f"[agentic] {summary}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FRIDAY — Advanced AI Voice Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py                  # Voice mode, Ollama LLM\n"
            "  python main.py --always-on      # Skip standby, always listen\n"
            "  python main.py --no-voice       # Text-only mode\n"
            "  python main.py --debug-voice    # Show diagnostics\n"
            "  python main.py --no-llm         # Rule-based, no Ollama needed\n"
            "  python main.py --no-agentic     # Disable agentic loop (single-turn only)\n"
        ),
    )
    parser.add_argument("--voice",    dest="voice", action="store_true",  help="Voice mode (default).")
    parser.add_argument("--no-voice", dest="voice", action="store_false", help="Text-only mode.")
    parser.add_argument("--always-on",  action="store_true", help="Skip standby, always listening.")
    parser.add_argument("--debug-voice", action="store_true", help="Print voice diagnostics.")
    parser.add_argument(
        "--whisper-model", default="small.en",
        help="Whisper model size: tiny.en, base.en, small.en (default), medium.en",
    )
    parser.add_argument(
        "--stt-language", default="en-IN",
        help="STT language locale (en-IN, en-US, hi-IN).",
    )
    parser.add_argument(
        "--llm-backend", choices=["ollama", "none"], default="ollama",
        help="LLM backend.",
    )
    parser.add_argument("--llm-model",    default="auto",                    help="Ollama model name.")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:11434", help="Ollama server URL.")
    parser.add_argument("--llm-timeout",  type=float, default=60.0,          help="LLM timeout (seconds).")
    parser.add_argument("--no-llm",       action="store_true",               help="Disable LLM.")
    parser.add_argument("--list-skills",  action="store_true",               help="List skills and exit.")
    parser.add_argument("--no-monitor",   action="store_true",               help="Disable ProactiveMonitor.")
    parser.add_argument("--no-agentic",   action="store_true",               help="Disable agentic loop.")
    parser.set_defaults(voice=True)

    args = parser.parse_args(sys.argv[1:])

    if args.list_skills:
        print(SkillEngine().list_skills())
        sys.exit(0)

    run(
        default_voice_mode=args.voice,
        debug_voice=args.debug_voice,
        whisper_model_size=args.whisper_model,
        stt_language=args.stt_language,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_timeout=args.llm_timeout,
        llm_enabled=not args.no_llm,
        always_on=args.always_on,
        monitor_enabled=not args.no_monitor,
        agentic_enabled=not args.no_agentic,
    )