"""
voice.py — Voice I/O pipeline for FRIDAY.

IN  → sounddevice → Silero VAD → faster-whisper (RTX 4050) → text
OUT → Kokoro ONNX (local neural TTS, ~50-150ms/sentence) → sounddevice

Why Kokoro instead of Edge TTS:
  - 100% local — no network, no rate limits, no "No audio received" failures
  - ~50-150ms generation after first call (model stays warm in memory)
  - No ffplay subprocess needed — audio played directly via sounddevice
  - Same kokoro-v1.0.onnx + voices-v1.0.bin you already downloaded
"""
from __future__ import annotations

import os
import queue
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Iterator

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

from ear import (
    np, sd, sr,
    setup_mic, setup_stt_backend,
    StreamingVoiceCapture, _try_import,
)

# ── Kokoro ONNX local TTS ─────────────────────────────────────────────────
_Kokoro = _try_import("kokoro_onnx", "Kokoro")

# Paths to local model files (already downloaded)
_KOKORO_MODEL  = os.path.join(os.path.dirname(__file__), "kokoro-v1.0.onnx")
_KOKORO_VOICES = os.path.join(os.path.dirname(__file__), "voices-v1.0.bin")

# Voice selection — Irish female (closest to Emily) or American female
# Available en voices: af_sky, af_bella, af_sarah, af_nicole, am_adam,
#                      am_michael, bf_emma, bf_isabella, bm_george, bm_lewis
FRIDAY_VOICE = "bf_emma"   # British female — sounds great, change if you like
FRIDAY_SPEED = 1.1          # Slightly faster than normal for snappier feel
FRIDAY_LANG  = "en-gb"     # British English (matches bf_emma perfectly)


@dataclass
class VoiceIO:
    """
    Full-duplex voice pipeline.
    IN  → StreamingVoiceCapture (Silero VAD + Whisper GPU)
    OUT → Kokoro ONNX (local neural TTS) → sounddevice
    """

    voice_mode:            bool  = True
    mic_enabled:           bool  = False
    tts_enabled:           bool  = True
    recognizer:            Optional[object] = None
    mic_backend:           str   = "none"
    stt_backend:           str   = "whisper"
    whisper_model_size:    str   = "small.en"
    stt_language:          str   = "en-IN"
    post_tts_cooldown_s:   float = 1.2
    debug:                 bool  = False
    whisper_model:         Optional[object] = None
    tts_effective_backend: str   = "none"

    _last_tts_time:          float           = field(default=0.0,                repr=False)
    _playback_stop:          threading.Event = field(default_factory=threading.Event, repr=False)
    _streaming_capture:      Optional[StreamingVoiceCapture] = field(default=None, repr=False)
    _kokoro:                 Optional[object] = field(default=None, repr=False)
    _kokoro_lock:            threading.Lock  = field(default_factory=threading.Lock, repr=False)
    _current_playback:       Optional[threading.Thread] = field(default=None, repr=False)
    _barge_in_speech_pending: bool           = field(default=False, repr=False)

    def __post_init__(self) -> None:
        # ── Mic setup ──
        self.recognizer, self.mic_enabled, self.mic_backend = setup_mic(
            self.recognizer, self._debug
        )
        self.stt_backend, self.whisper_model = setup_stt_backend(
            self.whisper_model_size, self._debug
        )

        # ── Kokoro TTS setup ──
        self._init_kokoro()

        # ── Streaming capture ──
        if self.mic_enabled and self.mic_backend == "sounddevice_stream":
            self._streaming_capture = StreamingVoiceCapture(
                stt_backend=self.stt_backend,
                whisper_model=self.whisper_model,
                recognizer=self.recognizer,
                stt_language=self.stt_language,
                debug=self._debug,
            )
            self._streaming_capture.start()
            self._debug("Streaming capture initialized.")

            # Wait for Silero VAD to be ready so we never miss the wake word
            self._debug("Waiting for Silero VAD...")
            ready = self._streaming_capture.vad.wait_ready(timeout=10)
            if ready:
                self._debug("Silero VAD ready.")
            else:
                self._debug("Silero VAD not ready (RMS fallback active).")

    def _init_kokoro(self) -> None:
        """Load Kokoro ONNX model from local files. Keeps model in memory."""
        if _Kokoro is None:
            self._debug("kokoro-onnx not installed — TTS disabled. Run: pip install kokoro-onnx")
            self.tts_enabled = False
            self.tts_effective_backend = "none"
            return

        if not os.path.exists(_KOKORO_MODEL):
            self._debug(f"Kokoro model not found: {_KOKORO_MODEL}")
            self.tts_enabled = False
            self.tts_effective_backend = "none"
            return

        if not os.path.exists(_KOKORO_VOICES):
            self._debug(f"Kokoro voices not found: {_KOKORO_VOICES}")
            self.tts_enabled = False
            self.tts_effective_backend = "none"
            return

        try:
            self._debug("Loading Kokoro ONNX TTS model...")
            t0 = time.monotonic()
            self._kokoro = _Kokoro(_KOKORO_MODEL, _KOKORO_VOICES)
            elapsed = time.monotonic() - t0
            self._debug(f"TTS: Kokoro ONNX ({FRIDAY_VOICE}) loaded in {elapsed:.2f}s [OK]")
            self.tts_enabled = True
            self.tts_effective_backend = "kokoro"

            # Warm-up call so the first real speak() is fast
            self._debug("Warming up Kokoro TTS...")
            self._kokoro.create("Ready.", voice=FRIDAY_VOICE, speed=FRIDAY_SPEED, lang=FRIDAY_LANG)
            self._debug("Kokoro TTS warm-up complete.")

        except Exception as e:
            self._debug(f"Kokoro TTS failed to load: {e}")
            self._kokoro = None
            self.tts_enabled = False
            self.tts_effective_backend = "none"

    def _debug(self, msg: str) -> None:
        if self.debug:
            print(f"[voice-debug] {msg}")

    # ── Shutdown ──────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        if self._streaming_capture:
            self._streaming_capture.stop()
            self._streaming_capture = None
        self._playback_stop.set()
        self.stop_playback()

    # ── Mic mute (for multi-message TTS blocks like boot) ─────────────

    def mute_mic(self) -> None:
        """Manually mute mic capture. Use for multi-speak blocks (boot, etc.).
        While muted, _play_audio() skips its own pause/resume.
        Call unmute_mic() when the block is done."""
        if self._streaming_capture:
            self._streaming_capture.pause_for_tts()

    def unmute_mic(self) -> None:
        """Unmute mic after a mute_mic() block. Flushes stale audio + settles."""
        if self._streaming_capture:
            self._streaming_capture.resume_after_tts()
        self._last_tts_time = time.monotonic()

    # ── Listen ────────────────────────────────────────────────────────────

    def listen(self) -> Optional[str]:
        """Block until user speaks; return transcript or None."""
        # Streaming path does not need sr at all — sr is only used by the
        # speech_recognition_mic fallback below. Checking sr is None here
        # would silently kill the streaming path if SpeechRecognition is
        # missing/broken, making Friday permanently deaf on sounddevice_stream.
        if not self.voice_mode or not self.mic_enabled:
            return None
        if time.monotonic() - self._last_tts_time < self.post_tts_cooldown_s:
            return None

        if self._streaming_capture:
            return self._streaming_capture.listen_for_utterance()

        # Fallback: speech_recognition mic (PyAudio backend)
        if sr is None:
            return None
        if self.mic_backend == "speech_recognition_mic" and self.recognizer is not None:
            try:
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=8)
                transcript = self.recognizer.recognize_google(audio, language=self.stt_language)
                return transcript.strip() if transcript else None
            except Exception as e:
                self._debug(f"STT mic fallback failed: {e}")
        return None

    # ── Core TTS generation ───────────────────────────────────────────────

    def _generate_tts(self, text: str) -> Optional[tuple]:
        """
        Generate audio via Kokoro ONNX — fully local, ~50-150ms per sentence.
        Returns (samples_float32, sample_rate) or None on failure.
        No network, no rate limits, no temp files.
        """
        if self._kokoro is None or not text.strip():
            return None

        # Strip characters that Kokoro doesn't handle well
        clean = text.replace("[", "").replace("]", "").replace("*", "").strip()
        if not clean:
            return None

        try:
            with self._kokoro_lock:
                t0 = time.monotonic()
                samples, sample_rate = self._kokoro.create(
                    clean,
                    voice=FRIDAY_VOICE,
                    speed=FRIDAY_SPEED,
                    lang=FRIDAY_LANG,
                )
                elapsed = (time.monotonic() - t0) * 1000
                self._debug(f"TTS Kokoro ({elapsed:.0f}ms): {clean[:50]!r}")
            return (samples, sample_rate)
        except Exception as e:
            self._debug(f"TTS Kokoro failed: {e}")
            return None

    # ── Playback ──────────────────────────────────────────────────────────

    def _play_audio(self, samples, sample_rate: int) -> None:
        """Play float32 audio array via sounddevice. Blocks until done."""
        if sd is None or samples is None:
            return
        # If mic is already muted (e.g., boot block or speak_pipeline),
        # skip our own pause/resume to avoid double-unmuting mid-sequence.
        already_muted = (
            self._streaming_capture is not None
            and self._streaming_capture._speaking
        )
        if self._streaming_capture and not already_muted:
            self._streaming_capture.pause_for_tts()
        _out_stream = None
        try:
            # Use an explicit OutputStream so we can abort ONLY playback
            # without touching the mic InputStream.
            # sd.play() + sd.stop() kills ALL streams including the mic —
            # that's the root cause of the post-barge-in hang.
            import numpy as _np
            _out_stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1 if samples.ndim == 1 else samples.shape[1],
                dtype='float32',
            )
            _out_stream.start()
            # Write in chunks so we can respect the stop event
            chunk_size = sample_rate // 10  # 100ms chunks
            flat = samples.flatten() if samples.ndim > 1 else samples
            offset = 0
            play_duration = len(flat) / sample_rate
            deadline = time.monotonic() + play_duration + 2.0
            while offset < len(flat):
                if self._playback_stop.is_set():
                    break
                if time.monotonic() > deadline:
                    self._debug("[play_audio] playback timeout — force stopping output")
                    break
                end = min(offset + chunk_size, len(flat))
                _out_stream.write(flat[offset:end].reshape(-1, 1) if _out_stream.channels == 1 else flat[offset:end])
                offset = end
        except Exception as e:
            self._debug(f"Audio playback failed: {e}")
        finally:
            if _out_stream is not None:
                try:
                    _out_stream.stop()
                    _out_stream.close()
                except Exception:
                    pass
            if self._streaming_capture and not already_muted:
                self._streaming_capture.resume_after_tts()

    # ── speak() — single-shot ─────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Generate and play TTS synchronously (boot messages, short replies)."""
        if not self.voice_mode or not text.strip() or not self.tts_enabled:
            return
        result = self._generate_tts(text)
        if result:
            samples, sr_rate = result
            try:
                self._play_audio(samples, sr_rate)
                self._last_tts_time = time.monotonic()
                self._debug(f"Spoke: {text[:60]}")
            except Exception as e:
                self._debug(f"speak() playback failed: {e}")
        else:
            self._debug(f"speak() TTS failed for: {text[:40]!r}")

    # ── Barge-in monitor ─────────────────────────────────────────────────

    def _barge_in_monitor(self, stop_event: threading.Event, playback_started: threading.Event) -> None:
        """
        Daemon thread: watches the mic via Silero VAD while speak_pipeline() runs.

        Trigger rule: 4 consecutive 32ms chunks satisfying BOTH:
          (a) speech_prob ≥ BARGE_IN_THRESHOLD (0.85)
          (b) chunk RMS ≥ loopback_baseline_ema × RMS_GATE_MULTIPLIER (2.5×)
        → fires stop_playback() to cut TTS immediately.

        WHY DUAL GATE:
        The stream stays live during TTS (single-stream WASAPI design), so the
        mic picks up TTS loopback from speakers. Silero scores loopback as
        speech (correctly — it IS speech phonemes). Two discriminators separate
        loopback from real barge-in:

          1. RMS relative gate: loopback through room air is attenuated vs the
             user's voice directly into the mic. The EMA baseline tracks the
             loopback energy level; real speech must be louder (× RMS_GATE_MULTIPLIER).

          2. Duration (CONFIRM_CHUNKS=4 = 128ms): loopback spikes from a single
             TTS word are brief (2-3 chunks); deliberate user interruption sustains
             for 200ms+ (≥4 chunks for any normally spoken word).

        Tune RMS_GATE_MULTIPLIER lower (2.0) if barge-in is hard to trigger,
        higher (3.5) if speaker loopback still causes false positives.
        """
        if np is None or self._streaming_capture is None:
            return

        BARGE_IN_THRESHOLD  = 0.88   # raise slightly — more selective for human voice
        CONFIRM_CHUNKS      = 4      # 4×32ms=128ms — still deliberate, less than fan sustain
        CHUNK_SIZE          = 512    # 32ms at 16kHz — Silero requirement
        RMS_GATE_MULTIPLIER = 3.0    # slightly easier for real user voice to beat
        RMS_EMA_ALPHA       = 0.12   # EMA smoothing for loopback baseline
        RMS_FLOOR           = 0.012  # raise floor — prevents 0.010 start being too easy
        MAX_BASELINE        = 0.025  # cap: need≥ never exceeds 0.025×3.0=0.075
        #   Without this cap the EMA tracks loud laptop loopback (0.06–0.09 RMS)
        #   and pushes need≥ to 0.17+, which real user voice cannot beat.
        #   With the cap: loopback mostly rejected; user speaking clearly at
        #   0.10+ RMS for 160ms triggers reliably.

        vad              = self._streaming_capture.vad
        consecutive      = 0
        rms_baseline_ema = RMS_FLOOR  # start AT floor — never below the minimum gate
        _barge_in_confirmed = False
        # Adaptive confirm target — updated per chunk based on RMS confidence.
        _confirm_target  = CONFIRM_CHUNKS
        # Speech chunks that passed BOTH gates — saved to pre-buffer on confirm.
        # Reset whenever consecutive drops to 0 (loopback broke the run).
        speech_chunks: list = []

        # Wire up the queue — callback will feed it while _speaking=True
        #
        # Unbounded (maxsize=0): the monitor processes each chunk in ~1-3ms
        # (Silero CPU inference); chunks arrive every 32ms (31/sec at 48kHz).
        # The monitor runs at 10-30× the input rate — no backlog can build.
        # A bounded queue with put_nowait() would silently drop NEW chunks
        # while keeping STALE ones — exactly wrong for real-time detection.
        # The queue is GC'd when the pipeline ends, so unbounded is safe.
        barge_q: queue.Queue = queue.Queue()
        self._streaming_capture._barge_in_queue = barge_q

        # Wait for actual audio before watching for interrupts
        # If pipeline ends before audio ever plays (LLM failed etc), exit cleanly
        playback_started.wait(timeout=15.0)
        if stop_event.is_set():
            self._streaming_capture._barge_in_queue = None
            return

        # Collect baseline samples first to warm up the EMA
        WARMUP_CHUNKS = 8   # ~256ms of loopback to seed the EMA
        warmed = 0
        while warmed < WARMUP_CHUNKS and not stop_event.is_set():
            try:
                raw = barge_q.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if self._streaming_capture._denoiser is not None:
                chunk = raw[::self._streaming_capture._DECIMATE]
            else:
                chunk = raw

            if len(chunk) < CHUNK_SIZE:
                chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))
            elif len(chunk) > CHUNK_SIZE:
                chunk = chunk[:CHUNK_SIZE]

            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms > 0.001:   # only update from non-silent chunks (real loopback)
                rms_baseline_ema = (
                    RMS_EMA_ALPHA * rms + (1 - RMS_EMA_ALPHA) * rms_baseline_ema
                )
            warmed += 1

        self._debug(f"[barge-in] baseline warmed: {rms_baseline_ema:.4f}")

        try:
            while not stop_event.is_set():
                try:
                    raw = barge_q.get(timeout=0.05)
                except queue.Empty:
                    continue

                # ── Decimate to 16kHz (inline, no shared DSP state) ─────────
                # We intentionally skip _process_chunk() / RNNoise here.
                # _denoiser.denoise_chunk() mutates a stateful GRU — calling it
                # from this thread concurrently with listen_for_utterance() would
                # be a data race on the RNNoise C context (corruption / crash).
                # Barge-in only needs a rough speech probability, not clean audio;
                # a plain decimate slice is sufficient and touches zero shared state.
                if self._streaming_capture._denoiser is not None:
                    chunk = raw[::self._streaming_capture._DECIMATE]  # 48kHz → 16kHz
                else:
                    chunk = raw  # stream is already 16kHz

                # Pad / trim to exactly 512 samples for Silero
                if len(chunk) < CHUNK_SIZE:
                    chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))
                elif len(chunk) > CHUNK_SIZE:
                    chunk = chunk[:CHUNK_SIZE]

                prob = vad.speech_prob(chunk)
                rms  = float(np.sqrt(np.mean(chunk ** 2)))
                rms_required = max(RMS_FLOOR, rms_baseline_ema) * RMS_GATE_MULTIPLIER

                if prob >= BARGE_IN_THRESHOLD:
                    # ── RMS relative gate ──────────────────────────────────
                    rms_ok = rms >= rms_required
                    if rms_ok:
                        # ── Adaptive confirmation window ───────────────────
                        ratio = rms / max(rms_required, 1e-9)
                        if ratio >= 2.0:
                            _confirm_target = max(3, CONFIRM_CHUNKS - 1)
                        elif ratio >= 1.3:
                            _confirm_target = CONFIRM_CHUNKS
                        else:
                            _confirm_target = CONFIRM_CHUNKS + 1
                        speech_chunks.append(chunk)   # save for pre-buffer
                        consecutive += 1
                        self._debug(
                            f"[barge-in] prob={prob:.2f} rms={rms:.4f} "
                            f"need≥{rms_required:.4f} HIT {consecutive}/{_confirm_target}"
                        )
                        if consecutive >= _confirm_target:
                            if stop_event.is_set():
                                return
                            self._debug("[barge-in] CONFIRMED — saving audio + stopping playback")
                            _barge_in_confirmed = True
                            # ── Save confirmed chunks + drain 200ms more ──────────
                            # The confirmed chunks ARE the word onset. Drain for
                            # ~200ms more to catch the rest of the syllable/word
                            # before the queue is deregistered and audio is lost.
                            drain_until = time.monotonic() + 0.20
                            while time.monotonic() < drain_until:
                                try:
                                    raw2 = barge_q.get(timeout=0.02)
                                except queue.Empty:
                                    break
                                if self._streaming_capture._denoiser is not None:
                                    c2 = raw2[::self._streaming_capture._DECIMATE]
                                else:
                                    c2 = raw2
                                if len(c2) < CHUNK_SIZE:
                                    c2 = np.pad(c2, (0, CHUNK_SIZE - len(c2)))
                                elif len(c2) > CHUNK_SIZE:
                                    c2 = c2[:CHUNK_SIZE]
                                speech_chunks.append(c2)
                            self._streaming_capture._barge_in_audio_buffer = list(speech_chunks)
                            self._debug(
                                f"[barge-in] pre-buffer: {len(speech_chunks)} chunks "
                                f"({len(speech_chunks)*32}ms) saved"
                            )
                            self.stop_playback()
                            return
                    else:
                        consecutive = 0
                        speech_chunks = []   # reset — this run was loopback
                        self._debug(
                            f"[barge-in] prob={prob:.2f} rms={rms:.4f} "
                            f"need≥{rms_required:.4f} loopback"
                        )
                        # Update baseline from high-prob loopback (confirmed speaker output).
                        rms_baseline_ema = min(
                            MAX_BASELINE,
                            RMS_EMA_ALPHA * rms + (1 - RMS_EMA_ALPHA) * rms_baseline_ema
                        )
                elif prob >= 0.50:
                    # Medium-probability chunk: Silero is uncertain — could be
                    # attenuated loopback or a soft consonant. Update baseline
                    # only if RMS is below the gate (i.e. not a real user voice).
                    consecutive = 0
                    speech_chunks = []
                    if rms < rms_required:
                        rms_baseline_ema = min(
                            MAX_BASELINE,
                            RMS_EMA_ALPHA * 0.5 * rms + (1 - RMS_EMA_ALPHA * 0.5) * rms_baseline_ema
                        )
                else:
                    consecutive = 0
                    speech_chunks = []
                    # Quiet/noise chunk: do NOT update baseline.
        finally:
            # Only zero the cooldown on a NATURAL pipeline end where the user
            # was mid-word (consecutive>=1 but not yet confirmed). On a confirmed
            # barge-in, stop_playback() already zeroed _last_tts_time.
            if not _barge_in_confirmed and consecutive >= 1:
                self._last_tts_time = 0.0
                self._debug("[barge-in] partial speech at pipeline end — cooldown cleared")
            # Deregister queue — callback reverts to dropping audio when _speaking=True
            self._streaming_capture._barge_in_queue = None

    # ── speak_pipeline() — 3-thread parallel pipeline ────────────────────

    def speak_pipeline(self, sentences: Iterator[str]) -> None:
        """
        TRUE 3-thread parallel pipeline:

          Thread 1 (llm-feeder): pulls text from LLM generator → sentence_q
          Thread 2 (tts-worker): sentence_q → Kokoro generate → audio_q
          Main thread (player):  audio_q → sounddevice.play()

        While sentence N is PLAYING, sentence N+1 is being TTS-generated.
        Kokoro is ~50-150ms vs Edge TTS ~1500ms → massively lower latency.

        Filters [ACTION:...] tags — handled by skill dispatch.

        Mic lifecycle (pipeline-level, not per-sentence):
          pause_for_tts()  ← called ONCE here, before any thread starts
              guarantees _speaking=True before barge_q is registered,
              so the callback feeds the monitor from the very first chunk.
              _play_audio() sees already_muted=True and skips its own
              pause/resume — no per-sentence churn.
          resume_after_tts() ← called ONCE after pipeline exits (normal or
              barge-in) — single clean handoff to listen_for_utterance().
        """
        if not self.voice_mode or not self.tts_enabled:
            return

        self._playback_stop.clear()
        sentence_q: queue.Queue = queue.Queue()
        audio_q:    queue.Queue = queue.Queue()

        # ── Mute mic NOW — before the monitor thread even starts ──────────
        # This sets _speaking=True so that as soon as the monitor registers
        # barge_q, the callback immediately routes audio there.
        # No race window: monitor always has a live feed from chunk 1.
        if self._streaming_capture:
            self._streaming_capture.pause_for_tts()

        # ── Barge-in monitor — starts immediately, dies when pipeline ends ──
        _barge_in_stop = threading.Event()
        _playback_started = threading.Event()
        threading.Thread(
            target=self._barge_in_monitor,
            args=(_barge_in_stop, _playback_started),
            daemon=True,
            name="barge-in-monitor",
        ).start()

        # ── Thread 1: drain the LLM sentence generator ──
        def _sentence_feeder():
            for s in sentences:
                s = s.strip()
                if s and "[ACTION:" not in s:
                    sentence_q.put(s)
            sentence_q.put(None)  # sentinel

        threading.Thread(target=_sentence_feeder, daemon=True, name="llm-feeder").start()

        # ── Thread 2: TTS generation via Kokoro (~50-150ms each) ──
        def _tts_worker():
            try:
                while True:
                    try:
                        text = sentence_q.get(timeout=60.0)
                    except queue.Empty:
                        break
                    if text is None:
                        break  # text sentinel from feeder
                    if self._playback_stop.is_set():
                        break  # barge-in fired — exit without generating more TTS
                    result = self._generate_tts(text)
                    audio_q.put(result)  # None on failure — player skips it
            finally:
                audio_q.put(None)  # sentinel — exactly once, on every exit path

        threading.Thread(target=_tts_worker, daemon=True, name="tts-worker").start()

        # ── Main thread: playback ──
        # _play_audio() sees already_muted=True (we set _speaking above) and
        # skips its own pause_for_tts / resume_after_tts on every sentence.
        while True:
            try:
                result = audio_q.get(timeout=60.0)
            except queue.Empty:
                break
            if result is None:
                break  # sentinel
            if self._playback_stop.is_set():
                continue
            samples, sample_rate = result
            try:
                _playback_started.set()
                self._play_audio(samples, sample_rate)
                self._last_tts_time = time.monotonic()
            except Exception as e:
                self._debug(f"Pipeline play failed: {e}")

        # ── Single clean handoff — normal finish OR barge-in ─────────────
        # Stop monitor first (deregisters barge_q), then restore mic.
        # Pass barge_in=True when the user interrupted — resume skips the
        # 300ms echo-decay sleep so we don't clip the word they just spoke.
        _barge_in_stop.set()
        if self._streaming_capture:
            was_barge_in = self._streaming_capture._barge_in_fired
            self._streaming_capture.resume_after_tts(barge_in=was_barge_in)

    def stop_playback(self) -> None:
        """Barge-in: stop sounddevice playback immediately.

        Sets _playback_stop so the _play_audio write-loop exits cleanly.
        Does NOT call sd.stop() — that would kill the mic InputStream and
        cause listen_for_utterance() to hang forever after barge-in.
        """
        self._playback_stop.set()
        # Zero the cooldown so listen() doesn't wait 1.2s after barge-in.
        self._last_tts_time = 0.0
        self._barge_in_speech_pending = True
        # Signal the capture so resume_after_tts() skips its 300ms sleep.
        # The user is already speaking — any delay clips their first word.
        if self._streaming_capture is not None:
            self._streaming_capture._barge_in_fired = True