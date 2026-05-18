"""
ear.py — Microphone setup, streaming voice capture, and STT.

Architecture:
  - Silero VAD  : Neural voice activity detection (human vs noise)
  - faster-whisper : Local GPU-accelerated STT (RTX 4050, ~150-250ms)
  - Google STT  : Silent fallback if whisper CUDA unavailable
"""
from __future__ import annotations

import collections
import importlib
import io
import wave
import queue
import threading
import time
import warnings
from typing import Any, Optional, Callable

# Suppress pkg_resources deprecation warning emitted by webrtcvad on import
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)


# ── Optional dependency loader ─────────────────────────────────────────────
def _try_import(module_name: str, attribute: str | None = None) -> Any:
    """Import a module or attribute, returning None if unavailable."""
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, attribute) if attribute else mod
    except (ImportError, AttributeError):
        return None


sr          = _try_import("speech_recognition")
sd          = _try_import("sounddevice")
np          = _try_import("numpy")
WhisperModel = _try_import("faster_whisper", "WhisperModel")

# ── RNNoise — optional neural mic denoiser (<1ms per 32ms chunk) ──────────
# Install: pip install pyrnnoise
# Effect: strips fans, keyboard clicks, background TV before VAD + Whisper
try:
    import pyrnnoise as _pyrnnoise
    _RNNoise = _pyrnnoise.RNNoise
except ImportError:
    _RNNoise = None


# ──────────────────────────────────────────────────────────────────────────
# 1.  SILERO VAD — neural human-voice detector
# ──────────────────────────────────────────────────────────────────────────
class SileroVAD:
    """
    Neural voice activity detection using the silero-vad package.
    Distinguishes human speech from background noise (fans, keyboard,
    TV audio, door slams, etc.) with a 0.0–1.0 probability score.

    Loads asynchronously on init; falls back to RMS gate until ready.
    Exposes raw probability via speech_prob() for hysteresis logic.
    """

    SAMPLE_RATE   = 16000
    CHUNK_SAMPLES = 512  # ~32ms at 16kHz — Silero's required chunk size

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._rms_threshold = 0.008  # RMS fallback gate — updated by noise floor
        self._model    = None
        self._torch    = None
        self._ready    = False
        self._load_thread = threading.Thread(target=self._load, daemon=True)
        self._load_thread.start()

    def _load(self) -> None:
        try:
            import torch
            from silero_vad import load_silero_vad  # type: ignore
            self._model = load_silero_vad()
            self._torch = torch
            self._ready = True
        except Exception:
            self._ready = False

    def wait_ready(self, timeout: float = 15.0) -> bool:
        self._load_thread.join(timeout=timeout)
        return self._ready

    def speech_prob(self, chunk_float32: "np.ndarray") -> float:
        """
        Returns speech probability 0.0–1.0 for a 512-sample float32 chunk.
        Falls back to normalised RMS energy (0.0–1.0 scale) before Silero loads.
        """
        if not self._ready or self._model is None:
            # Map RMS to a 0–1 pseudo-probability:
            # RMS < rms_threshold → ~0.0, RMS = 3×threshold → ~1.0
            rms = float(np.sqrt(np.mean(chunk_float32 ** 2)))
            return min(1.0, rms / max(self._rms_threshold * 3.0, 1e-9))

        try:
            tensor = self._torch.from_numpy(chunk_float32).unsqueeze(0)
            return float(self._model(tensor, self.SAMPLE_RATE).item())
        except Exception:
            rms = float(np.sqrt(np.mean(chunk_float32 ** 2)))
            return min(1.0, rms / max(self._rms_threshold * 3.0, 1e-9))

    def is_speech(self, chunk_float32: "np.ndarray") -> bool:
        """Simple threshold gate (kept for backward compatibility)."""
        return self.speech_prob(chunk_float32) >= self.threshold

    def reset(self) -> None:
        """Reset Silero's internal GRU state between utterances."""
        if self._ready and self._model is not None:
            try:
                self._model.reset_states()
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────────
# 1b. DYNAMIC NOISE FLOOR ESTIMATOR
#     Adapts VAD threshold to ambient room noise in real time.
# ──────────────────────────────────────────────────────────────────────────
class DynamicNoiseFloor:
    """
    Background estimator that continuously tracks the ambient noise level
    and dynamically adjusts the Silero VAD threshold.

    How it works:
      - Every `interval_s` seconds, measures the average RMS energy of
        recently seen audio chunks (the "ambient window").
      - Maintains an exponential moving average (EMA) of the noise floor.
      - Sets VAD threshold = clamp(base + noise_floor * margin, min, max).

    Quiet room (library)  → noise floor ~0.001 → threshold drops  → catches whispers
    Noisy room (fan on)   → noise floor ~0.02  → threshold rises  → ignores background
    Speech                → energy spikes well above threshold    → triggers VAD
    """

    def __init__(
        self,
        vad: SileroVAD,
        capture: "StreamingVoiceCapture",
        debug: Callable,
        interval_s: float = 3.0,
        ema_alpha: float = 0.3,
        margin_multiplier: float = 3.0,
        min_threshold: float = 0.30,
        max_threshold: float = 0.70,
        base_threshold: float = 0.35,
        rms_margin: float = 4.0,
        min_rms: float = 0.004,
        max_rms: float = 0.030,
    ):
        self.vad = vad
        self.capture = capture  # back-ref to update hysteresis thresholds
        self.debug = debug
        self.interval_s = interval_s

        # EMA smoothing factor (0→slow adaptation, 1→instant)
        self.ema_alpha = ema_alpha

        # Silero threshold = base + noise_floor_ema * margin, clamped to [min, max]
        self.margin_multiplier = margin_multiplier
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.base_threshold = base_threshold

        # RMS fallback threshold parameters
        self.rms_margin = rms_margin
        self.min_rms = min_rms
        self.max_rms = max_rms

        # State
        self._noise_floor_ema: float = 0.005  # start with a reasonable guess
        # Fixed-size rolling window — deque auto-evicts oldest, never clears
        self._recent_rms: collections.deque[float] = collections.deque(maxlen=100)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def feed(self, chunk_float32: "np.ndarray") -> None:
        """
        Feed a processed 16kHz ambient-only chunk.
        Called from the consumer thread ONLY when no speech is active.
        Speech chunks are excluded so your own voice doesn't inflate
        the noise floor estimate.
        """
        if np is None:
            return
        rms = float(np.sqrt(np.mean(chunk_float32 ** 2)))
        with self._lock:
            self._recent_rms.append(rms)  # deque(maxlen=100) auto-evicts oldest

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._update_loop, daemon=True, name="noise-floor"
        )
        self._thread.start()
        self.debug("Dynamic noise floor estimator started.")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _update_loop(self) -> None:
        """Background loop: every interval_s, recompute the noise floor."""
        while self._running:
            time.sleep(self.interval_s)
            if not self._running:
                break
            self._update_threshold()

    def _update_threshold(self) -> None:
        """
        Compute new noise floor from recent RMS samples.
        Uses the 25th percentile (Q1) of recent RMS values as the
        noise floor estimate — this filters out speech spikes and
        captures the true ambient level.
        """
        with self._lock:
            if len(self._recent_rms) < 10:
                return  # not enough data yet
            # Snapshot the rolling window — don't clear, deque keeps filling
            samples = list(self._recent_rms)

        # 25th percentile — robust to speech spikes
        sorted_rms = sorted(samples)
        q1_idx = len(sorted_rms) // 4
        ambient_rms = sorted_rms[q1_idx]

        # EMA smooth the noise floor
        self._noise_floor_ema = (
            self.ema_alpha * ambient_rms
            + (1 - self.ema_alpha) * self._noise_floor_ema
        )

        # ── Update Silero VAD threshold ──
        new_threshold = self.base_threshold + self._noise_floor_ema * self.margin_multiplier
        new_threshold = max(self.min_threshold, min(self.max_threshold, new_threshold))
        old_threshold = self.vad.threshold
        self.vad.threshold = new_threshold

        # ── Update RMS fallback threshold ──
        new_rms = self._noise_floor_ema * self.rms_margin
        new_rms = max(self.min_rms, min(self.max_rms, new_rms))
        self.vad._rms_threshold = new_rms

        if abs(new_threshold - old_threshold) >= 0.02:
            self.debug(
                f"Noise floor: {self._noise_floor_ema:.4f} RMS "
                f"→ VAD threshold: {old_threshold:.2f} → {new_threshold:.2f} "
                f"(enter={self.capture.enter_threshold:.2f}, "
                f"continue={self.capture.continue_threshold:.2f})"
            )

        # ── Scale hysteresis thresholds proportionally ──
        # noise_boost is the same delta applied to the base VAD threshold
        noise_boost = self._noise_floor_ema * self.margin_multiplier
        self.capture.enter_threshold = min(
            0.90,
            max(0.50, StreamingVoiceCapture.DEFAULT_ENTER_THRESHOLD + noise_boost * 0.5)
        )
        self.capture.continue_threshold = min(
            0.65,
            max(0.30, StreamingVoiceCapture.DEFAULT_CONTINUE_THRESHOLD + noise_boost * 0.5)
        )


# ──────────────────────────────────────────────────────────────────────────
# 2.  MIC SETUP
# ──────────────────────────────────────────────────────────────────────────
def setup_mic(
    recognizer: Any,
    debug: Callable,
    sample_rate: int = 16000,
    channels: int = 1,
) -> tuple[Any, bool, str]:
    """
    Probe for a working microphone input.
    Returns (recognizer_or_None, mic_available, backend_name).
    """
    if sr is None:
        debug("SpeechRecognition not installed.")
        return None, False, "none"

    if sd is not None and np is not None:
        # Try mono first, then stereo (AMD mic arrays expose 2ch only)
        for ch in (channels, 2):
            try:
                sd.check_input_settings(
                    device=None, channels=ch,
                    samplerate=sample_rate, dtype="float32",
                )
                debug(f"Mic backend: sounddevice_stream (channels={ch}).")
                return sr.Recognizer(), True, "sounddevice_stream"
            except Exception:
                continue
        debug("sounddevice: no compatible config found, trying PyAudio.")

    # PyAudio fallback (needs separate install)
    try:
        rec = sr.Recognizer()
        rec.energy_threshold = 300
        rec.dynamic_energy_threshold = True
        rec.pause_threshold = 0.5
        with sr.Microphone(sample_rate=sample_rate):
            pass
        debug("Mic backend: speech_recognition_mic.")
        return rec, True, "speech_recognition_mic"
    except OSError as e:
        debug(f"Mic setup failed (PyAudio unavailable): {e}")
    except Exception as e:
        debug(f"Mic setup failed: {e}")
    return None, False, "none"


# ──────────────────────────────────────────────────────────────────────────
# 3.  STT SETUP — faster-whisper on CUDA with google fallback
# ──────────────────────────────────────────────────────────────────────────
def setup_stt_backend(
    whisper_model_size: str,
    debug: Callable,
) -> tuple[str, Optional[object]]:
    """
    Try to load faster-whisper on CUDA (RTX 4050).
    Falls back to CPU whisper, then google if model load fails.
    Returns (backend_name, model_or_None).
    """
    if WhisperModel is None:
        debug("faster-whisper not installed -> falling back to google STT.")
        return "google", None

    # Try CUDA first, then CPU as last resort
    for device, compute in (("cuda", "int8_float16"), ("cpu", "int8")):
        try:
            debug(f"Loading whisper '{whisper_model_size}' on {device} ({compute})...")
            model = WhisperModel(
                whisper_model_size,
                device=device,
                compute_type=compute,
                cpu_threads=4,
                num_workers=1,
            )
            debug(f"STT: faster-whisper {whisper_model_size} on {device} [OK]")
            return "whisper", model
        except Exception as exc:
            debug(f"Whisper on {device} failed: {exc}")
            continue

    debug("Whisper unavailable -> falling back to google STT.")
    return "google", None


# ──────────────────────────────────────────────────────────────────────────
# 4.  STREAMING VOICE CAPTURE
#     Silero VAD → audio buffer → faster-whisper (GPU) → transcript
# ──────────────────────────────────────────────────────────────────────────
class StreamingVoiceCapture:
    """
    Real-time voice capture pipeline:
      sounddevice → Silero VAD (human detection) → whisper GPU transcription

    Flow:
      1. Audio comes in as 32ms chunks via sounddevice callback
      2. Each chunk is tested by Silero VAD (neural human-voice check)
      3. Once speech starts, chunks are buffered
      4. After 0.6s of silence post-speech, buffer is sent to Whisper
      5. Whisper returns transcript in ~150-250ms on RTX 4050
    """

    SAMPLE_RATE     = 16000   # rate seen by VAD + Whisper
    SAMPLE_RATE_MIC = 48000   # capture rate (pyrnnoise native, avoids resampling)
    BLOCKSIZE       = 512     # 32ms at 16kHz — Silero VAD requirement
    BLOCKSIZE_MIC   = 1536    # 32ms at 48kHz (512 * 3) — mic capture block
    CHANNELS        = 1
    _DECIMATE       = 3       # 48000 / 16000 = 3

    # ── Hysteresis thresholds (dynamically updated by noise floor) ──
    # ENTER speech:  prob > enter_threshold  (strict  — prevents false triggers)
    # STAY in speech: prob > continue_threshold (relaxed — survives quiet syllables)
    # EXIT speech:   prob < continue_threshold  (then silence timer starts)
    DEFAULT_ENTER_THRESHOLD    = 0.75
    DEFAULT_CONTINUE_THRESHOLD = 0.45

    def __init__(
        self,
        stt_backend: str,
        whisper_model: Optional[object],
        recognizer: Any,
        stt_language: str,
        debug: Callable,
        silence_seconds: float = 0.6,
        min_audio_duration: float = 0.3,
        vad_threshold: float = 0.5,
    ):
        self.stt_backend    = stt_backend
        self.whisper_model  = whisper_model
        self.recognizer     = recognizer
        self.stt_language   = stt_language
        self.debug          = debug
        self.silence_threshold  = silence_seconds
        self.min_audio_duration = min_audio_duration

        # Hysteresis thresholds — adapted by DynamicNoiseFloor
        self.enter_threshold    = self.DEFAULT_ENTER_THRESHOLD
        self.continue_threshold = self.DEFAULT_CONTINUE_THRESHOLD

        # ── RNNoise denoiser — strips background noise before VAD/Whisper ──
        # Runs at 48kHz (pyrnnoise native) — ~1.4ms per 32ms chunk, <1ms overhead
        # Captured at 48kHz → denoise → 3:1 decimate to 16kHz for VAD/Whisper
        if _RNNoise is not None:
            try:
                self._denoiser = _RNNoise(sample_rate=self.SAMPLE_RATE_MIC)  # 48 kHz
                debug("RNNoise denoiser: active (neural noise cancellation ~1ms/chunk)")
            except Exception as e:
                self._denoiser = None
                debug(f"RNNoise init failed (skipping): {e}")
        else:
            self._denoiser = None
            debug("RNNoise: not installed — pip install pyrnnoise for noise cancellation")

        # Neural VAD — loads async, RMS fallback until ready
        self.vad = SileroVAD(threshold=vad_threshold)
        debug("Silero VAD loading in background...")

        # Dynamic noise floor — adapts VAD + hysteresis thresholds to ambient room noise
        self._noise_floor = DynamicNoiseFloor(self.vad, self, debug)

        self._audio_queue: queue.Queue = queue.Queue(maxsize=100)
        self._stream  = None
        self._running = False
        self._speaking = False  # True while TTS is playing — audio routed to barge-in queue

        # Set by VoiceIO._barge_in_monitor while speak_pipeline() is active.
        # When _speaking=True the callback feeds this queue instead of _audio_queue,
        # so barge-in can watch the mic without opening a second InputStream.
        self._barge_in_queue: Optional[queue.Queue] = None

        # Set to True by VoiceIO.stop_playback() on a confirmed barge-in.
        # resume_after_tts() reads this to skip the echo-decay sleep — the
        # user is already talking, waiting 300ms would clip their first word.
        self._barge_in_fired: bool = False

        # Chunks saved by _barge_in_monitor on confirmation + 200ms drain.
        # listen_for_utterance() consumes this to recover the interrupted word.
        self._barge_in_audio_buffer: list = []

        # Set by resume_after_tts() after every TTS round. listen_for_utterance()
        # decrements it in IDLE and skips noise_floor.feed() while > 0, preventing
        # post-TTS reverb tail from inflating DynamicNoiseFloor over long sessions.
        self._post_tts_settle_chunks: int = 0

    # ── sounddevice callback (runs in audio thread) ──
    # MUST be ultra-lightweight: no numpy ops, no denoising, no blocking.
    # Just copy raw bytes into the queue; heavy work happens in the consumer.
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            # Suppress input_overflow — it's caused by CUDA/GIL contention during
            # Whisper inference and is not actionable. Logging it every 32ms during
            # transcription would flood the terminal. Other flags (xrun, etc.) log.
            if not status.input_overflow:
                self.debug(f"Stream status: {status}")
        if self._speaking:
            # TTS is playing: route to barge-in queue if monitor is active,
            # otherwise drop (loopback suppression via routing, not stream stop).
            biq = self._barge_in_queue
            if biq is not None:
                try:
                    biq.put_nowait(indata[:, 0].copy())
                except queue.Full:
                    pass
            return
        try:
            self._audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass  # drop frame rather than block the audio thread

    # ── RNNoise + decimate (called from consumer thread, not real-time) ──
    def _process_chunk(self, raw_chunk: "np.ndarray") -> "np.ndarray":
        """
        Denoise (if RNNoise active) and decimate 48kHz→16kHz.
        Called from listen_for_utterance on the voice thread.
        """
        if self._denoiser is not None and np is not None:
            try:
                denoised_parts = []
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    for _prob, frame in self._denoiser.denoise_chunk(
                        raw_chunk.reshape(1, -1).astype(np.float32)
                    ):
                        # frame is int16 from pyrnnoise; normalise to float32 [-1,1]
                        denoised_parts.append(
                            frame.flatten().astype(np.float32) / 32768.0
                        )
                if denoised_parts:
                    denoised_48k = np.concatenate(denoised_parts)
                    # 3:1 decimate 48kHz → 16kHz
                    return denoised_48k[::self._DECIMATE]
                return raw_chunk[::self._DECIMATE]
            except Exception:
                return raw_chunk[::self._DECIMATE]
        else:
            # No denoiser — stream is already 16kHz
            return raw_chunk

    # ── lifecycle ──
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        # Use 48kHz if RNNoise is active (its native rate, zero resampling overhead)
        # Fall back to 16kHz if RNNoise isn't installed
        mic_rate  = self.SAMPLE_RATE_MIC if self._denoiser is not None else self.SAMPLE_RATE
        mic_block = self.BLOCKSIZE_MIC   if self._denoiser is not None else self.BLOCKSIZE
        self._stream = sd.InputStream(
            samplerate=mic_rate,
            channels=self.CHANNELS,
            blocksize=mic_block,
            dtype="float32",
            callback=self._audio_callback,
            latency='high',
            extra_settings=None,
        )
        self._stream.start()
        self._noise_floor.start()
        self.debug("Streaming voice capture started.")

    def stop(self) -> None:
        self._running = False
        self._noise_floor.stop()
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.debug("Streaming voice capture stopped.")

    def pause_for_tts(self) -> None:
        """Signal that TTS is playing — routes mic audio to barge-in queue.

        The stream intentionally stays RUNNING so that barge-in can receive
        mic audio without opening a second InputStream (WASAPI dual-stream
        conflict). Loopback suppression is handled by the high VAD threshold
        (0.85) in _barge_in_monitor rather than at the stream level.
        """
        self._speaking = True

    def resume_after_tts(self, barge_in: bool = False) -> None:
        """Resume normal mic capture after TTS finishes.

        Normal path (barge_in=False):
          Set _speaking=False FIRST so audio starts routing to _audio_queue
          immediately (nothing is dropped during the reverb decay window).
          Then sleep 300ms for DAC + reverb tail. Then flush the reverb
          bleed that actually accumulated in _audio_queue during that window.

        Barge-in path (barge_in=True):
          User is already speaking — skip the sleep entirely.
          Preserve the pre-buffer through flush so listen_for_utterance
          can use the state signal.
        """
        saved_buffer = list(self._barge_in_audio_buffer) if barge_in else []

        # Set _speaking=False BEFORE any sleep so the audio callback routes
        # to _audio_queue immediately. On the normal path this means the
        # reverb tail accumulates in _audio_queue (not dropped), and flush()
        # below can actually clear it. Previously _speaking stayed True
        # during the 300ms sleep, audio went nowhere, and flush() was a no-op.
        self._speaking = False

        if not barge_in:
            time.sleep(0.30)  # reverb tail decays; audio accumulates in _audio_queue
            self.flush()      # now flush the actual loopback bleed
        else:
            self.flush()      # quick flush of any stale audio

        if saved_buffer:
            self._barge_in_audio_buffer = saved_buffer

        # Mark the next N chunks in listen_for_utterance as post-TTS settling.
        # These chunks may still contain reverb tail and must not be fed to
        # the noise floor estimator (Bug 5 — long-session drift).
        self._post_tts_settle_chunks = 10   # ~320ms at 32ms/chunk
        self._barge_in_fired = False

    def flush(self) -> None:
        """Drain buffered audio (call after TTS finishes to avoid echo)."""
        self._barge_in_audio_buffer = []  # discard stale barge-in seeds on flush
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        self.vad.reset()
        # Reset RNNoise GRU state so stale noise model doesn't bleed across utterances
        if self._denoiser is not None:
            try:
                self._denoiser.reset()
            except Exception:
                pass

    # ── main listen loop (blocking, called from voice thread) ──

    # Pre-speech ring buffer: ~100ms lookback at 32ms/chunk = 3 chunks
    _PRE_SPEECH_CHUNKS = 3

    def listen_for_utterance(self) -> Optional[str]:
        """
        Blocks until user finishes speaking, then returns transcript.

        4-STATE MACHINE with pre-speech ring buffer:

          IDLE      →  prob ≥ enter_threshold      →  PENDING
                       (ambient chunks feed noise floor estimator)
                       (ring buffer always keeps last 3 chunks ≈ 100ms)

          PENDING   →  2nd consecutive high chunk   →  CAPTURING
                       (2-chunk confirmation prevents single-frame spikes)
                       (ring buffer is prepended to main buffer for word onset)
                   →  prob < enter_threshold        →  IDLE (false alarm)

          CAPTURING →  prob ≥ continue_threshold    →  CAPTURING
                       (relaxed threshold survives quiet syllables)
                   →  prob < continue_threshold     →  TRAILING

          TRAILING  →  prob ≥ continue_threshold    →  CAPTURING (re-enters)
                   →  silence_threshold elapsed     →  DONE → Whisper

        Ring buffer ensures Whisper receives the word *onset*, not just
        audio from after detection — preventing clipped first syllables.
        """
        if np is None:
            return None

        # ── State ──
        IDLE, PENDING, CAPTURING, TRAILING = 0, 1, 2, 3
        state = IDLE

        buffer: list = []
        silence_start: Optional[float] = None
        pending_chunk: Optional["np.ndarray"] = None  # the first high-prob chunk

        # Pre-speech ring buffer — always keeps last N chunks (~100ms lookback)
        ring: collections.deque = collections.deque(maxlen=self._PRE_SPEECH_CHUNKS)

        # ── Barge-in pre-seed ──────────────────────────────────────────────
        # If the barge-in monitor confirmed speech, skip straight to CAPTURING.
        # IMPORTANT: do NOT put the pre-buffer chunks into the audio buffer —
        # they were captured while TTS was still playing and contain loopback
        # audio mixed with the user's voice. Feeding loopback to Whisper
        # produces hallucinations (e.g. "Ollama Ollama Ollama").
        # The pre-buffer is used ONLY as a state signal: we know the user is
        # speaking, so skip IDLE/PENDING and go straight to CAPTURING.
        # The actual clean audio arrives via _audio_queue after _speaking=False.
        if self._barge_in_audio_buffer:
            n_prebuf = len(self._barge_in_audio_buffer)
            self._barge_in_audio_buffer = []
            state = CAPTURING
            self.debug(
                f"[listen] barge-in detected ({n_prebuf} chunks) "
                f"— skipping to CAPTURING (clean audio only)"
            )

        # Watchdog: if mic stream dies (e.g. stale sd.stop() call), the queue
        # stays empty forever in CAPTURING/IDLE — bail after 5s of silence.
        _last_audio_t = time.perf_counter()
        _STREAM_DEAD_TIMEOUT = 5.0

        while self._running:
            try:
                raw = self._audio_queue.get(timeout=0.05)
                _last_audio_t = time.perf_counter()
            except queue.Empty:
                # In TRAILING, check if silence duration exceeded
                if state == TRAILING and silence_start is not None:
                    if (time.perf_counter() - silence_start) >= self.silence_threshold:
                        break
                # Dead stream watchdog
                if (time.perf_counter() - _last_audio_t) > _STREAM_DEAD_TIMEOUT:
                    self.debug("[listen] audio stream dead — returning None to recover")
                    return None
                continue

            # Denoise + decimate on consumer thread (not in audio callback)
            chunk = self._process_chunk(raw)

            # Pad/trim chunk to Silero's required 512 samples
            if len(chunk) < self.BLOCKSIZE:
                chunk = np.pad(chunk, (0, self.BLOCKSIZE - len(chunk)))
            elif len(chunk) > self.BLOCKSIZE:
                chunk = chunk[:self.BLOCKSIZE]

            prob = self.vad.speech_prob(chunk)

            # ── IDLE: listening for speech onset ──
            if state == IDLE:
                # Post-TTS settling: skip noise floor feeding for the first N
                # chunks after TTS ends. Those chunks may still contain reverb
                # tail which would inflate the DynamicNoiseFloor over time,
                # causing Friday to require progressively louder speech.
                if self._post_tts_settle_chunks > 0:
                    self._post_tts_settle_chunks -= 1
                else:
                    self._noise_floor.feed(chunk)  # only ambient feeds noise floor
                ring.append(chunk)  # always maintain lookback

                if prob >= self.enter_threshold:
                    # One high-prob chunk — move to PENDING for confirmation
                    state = PENDING
                    pending_chunk = chunk

            # ── PENDING: confirm speech with a 2nd consecutive chunk ──
            elif state == PENDING:
                if prob >= self.enter_threshold:
                    # Confirmed! Transition to CAPTURING
                    state = CAPTURING
                    silence_start = None

                    # Prepend ring buffer (pre-speech audio) for word onset
                    buffer.extend(ring)
                    ring.clear()

                    # Add the pending chunk and this confirming chunk
                    if pending_chunk is not None:
                        buffer.append(pending_chunk)
                        pending_chunk = None
                    buffer.append(chunk)
                else:
                    # False alarm — single spike, return to IDLE
                    state = IDLE
                    pending_chunk = None
                    ring.append(chunk)  # keep it in the ring

            # ── CAPTURING: active speech ──
            elif state == CAPTURING:
                buffer.append(chunk)
                if prob >= self.continue_threshold:
                    silence_start = None  # speech continues
                else:
                    # Dropped below continue threshold → start trailing
                    state = TRAILING
                    silence_start = time.perf_counter()

            # ── TRAILING: counting silence after speech ──
            elif state == TRAILING:
                buffer.append(chunk)  # keep trailing audio for natural boundaries
                if prob >= self.continue_threshold:
                    # Speech resumed — back to CAPTURING
                    state = CAPTURING
                    silence_start = None
                elif silence_start is not None:
                    if (time.perf_counter() - silence_start) >= self.silence_threshold:
                        break  # silence long enough → done

        if not buffer or state < CAPTURING:
            return None

        return self._transcribe(buffer)

    # ── transcription ──
    def _transcribe(self, buffer: list) -> Optional[str]:
        audio_np = np.concatenate(buffer).flatten()

        duration = len(audio_np) / self.SAMPLE_RATE
        if duration < self.min_audio_duration:
            self.debug(f"Audio too short ({duration:.2f}s) — skipped.")
            return None

        self.debug(f"Transcribing {duration:.2f}s of audio…")

        if self.stt_backend == "whisper" and self.whisper_model is not None:
            return self._transcribe_whisper(audio_np)

        if self.recognizer is not None:
            return self._transcribe_google(audio_np)

        self.debug("No STT backend available.")
        return None

    def _transcribe_whisper(self, audio_np: "np.ndarray") -> Optional[str]:
        """GPU-accelerated transcription via faster-whisper."""
        try:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(self.SAMPLE_RATE)
                wav.writeframes((audio_np * 32767).astype(np.int16).tobytes())
            buf.seek(0)

            t0 = time.perf_counter()
            segments, info = self.whisper_model.transcribe(
                buf,
                language=self.stt_language.split("-")[0],  # "en-IN" → "en"
                beam_size=5,              # better accuracy
                best_of=1,
                vad_filter=True,          # Whisper's own VAD as second pass
                vad_parameters={
                    "min_silence_duration_ms": 300,
                    "speech_pad_ms": 100,
                },
                condition_on_previous_text=False,
                temperature=0.0,          # greedy — fastest + most consistent
                # ── Anti-hallucination ──
                no_speech_threshold=0.5,  # was 0.6 — tighter: drop >50% no-speech
                log_prob_threshold=-0.6,  # was -0.8 — tighter: drop low-confidence
                hallucination_silence_threshold=0.3,  # was 0.5 — more aggressive
                # ── Domain vocabulary ──
                # Seeds the decoder so it knows these words exist.
                # IMPORTANT: do NOT include common English words or brand names
                # that Whisper might hallucinate on ambiguous/short audio.
                # 'Ollama' was removed — it caused "Ollama Ollama Ollama"
                # hallucinations on loopback-contaminated barge-in audio.
                initial_prompt=(
                    "Friday, wake up Friday, shut down Friday, "
                    "barge in, voice assistant, Akhil"
                ),
            )
            # Filter segments by confidence — drop low-quality ones
            parts = []
            for s in segments:
                text = s.text.strip()
                if not text:
                    continue
                # Skip segments with very low average log probability
                if s.avg_logprob < -0.8:
                    self.debug(f"  [skip low-conf: {text!r} logprob={s.avg_logprob:.2f}]")
                    continue
                # Skip segments with high no-speech probability
                if s.no_speech_prob > 0.4:
                    self.debug(f"  [skip no-speech: {text!r} no_speech={s.no_speech_prob:.2f}]")
                    continue
                parts.append(text)

            transcript = " ".join(parts).strip()
            elapsed = (time.perf_counter() - t0) * 1000

            if not transcript:
                self.debug(f"STT whisper ({elapsed:.0f}ms): <empty>")
                return None

            # ── Repetition detector — catches Whisper hallucination loops ──
            # Whisper hallucinates by repeating a single word (e.g. "open" ×112)
            # when audio is ambiguous or nearly silent. Any word appearing more
            # than 8 times = hallucination. Drop the whole transcript.
            words = transcript.lower().split()
            if words:
                most_common_count = max(words.count(w) for w in set(words))
                if most_common_count > 8:
                    self.debug(
                        f"STT whisper ({elapsed:.0f}ms): <hallucination — "
                        f"word repeated {most_common_count}×, dropping>"
                    )
                    return None

            self.debug(f"STT whisper ({elapsed:.0f}ms): {transcript}")
            return transcript

        except Exception as exc:
            self.debug(f"Whisper transcription failed: {exc}")
            return None

    def _transcribe_google(self, audio_np: "np.ndarray") -> Optional[str]:
        """Google STT fallback (requires internet)."""
        if sr is None or self.recognizer is None:
            return None
        try:
            int16 = (audio_np * 32767).astype(np.int16)
            audio_data = sr.AudioData(int16.tobytes(), self.SAMPLE_RATE, 2)
            transcript = self.recognizer.recognize_google(
                audio_data, language=self.stt_language
            )
            if transcript:
                self.debug(f"STT google: {transcript}")
                return transcript.strip()
        except Exception as exc:
            self.debug(f"STT google failed: {exc}")
        return None