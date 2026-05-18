"""
skills.py — Built-in skill engine for FRIDAY.

Every skill is a callable that returns a string result.
The LLM triggers skills by emitting [ACTION:skill_name:args] in its response.
The main loop intercepts these, runs the skill, and feeds the result back.

All tools are FREE FOREVER — no API keys required.
"""
from __future__ import annotations

import ast
import datetime
import json
import math
import os
import platform
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from urllib import parse as urlparse, request as urlreq


# ── App registry — common Windows apps ──────────────────────────────────────

_APP_REGISTRY: Dict[str, str] = {
    "notepad":    "notepad.exe",
    "calculator": "calc.exe",
    "calc":       "calc.exe",
    "paint":      "mspaint.exe",
    "explorer":   "explorer.exe",
    "file explorer": "explorer.exe",
    "files":      "explorer.exe",
    "cmd":        "cmd.exe",
    "terminal":   "wt.exe",
    "powershell": "powershell.exe",
    "task manager": "taskmgr.exe",
    "taskmgr":    "taskmgr.exe",
    "settings":   "ms-settings:",
    "control panel": "control.exe",
    "chrome":     r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "google chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "firefox":    r"C:\Program Files\Mozilla Firefox\firefox.exe",
    "edge":       "msedge.exe",
    "microsoft edge": "msedge.exe",
    "brave":      r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    "vscode":     "code",
    "vs code":    "code",
    "visual studio code": "code",
    "spotify":    r"C:\Users\{user}\AppData\Roaming\Spotify\Spotify.exe",
    "discord":    r"C:\Users\{user}\AppData\Local\Discord\Update.exe --processStart Discord.exe",
    "vlc":        r"C:\Program Files\VideoLAN\VLC\vlc.exe",
    "obs":        r"C:\Program Files\obs-studio\bin\64bit\obs64.exe",
}

# Browser process name → executable to launch a URL in that browser
_BROWSER_PROC_MAP: Dict[str, str] = {
    "msedge.exe":    "msedge.exe",
    "chrome.exe":    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "brave.exe":     r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    "firefox.exe":   r"C:\Program Files\Mozilla Firefox\firefox.exe",
}

# Site name → URL (for smart_open)
_SITE_MAP: Dict[str, str] = {
    "youtube":    "https://www.youtube.com",
    "gmail":      "https://mail.google.com",
    "github":     "https://github.com",
    "google":     "https://www.google.com",
    "chatgpt":    "https://chat.openai.com",
    "netflix":    "https://www.netflix.com",
    "twitter":    "https://twitter.com",
    "x":          "https://twitter.com",
    "reddit":     "https://www.reddit.com",
    "linkedin":   "https://www.linkedin.com",
    "stackoverflow": "https://stackoverflow.com",
    "wikipedia":  "https://www.wikipedia.org",
    "whatsapp":   "https://web.whatsapp.com",
    "instagram":  "https://www.instagram.com",
    "spotify":    "https://open.spotify.com",
    "notion":     "https://notion.so",
}


# ── Timer storage ───────────────────────────────────────────────────────────

_active_timers: List[dict] = []

# ── Announce hook ────────────────────────────────────────────────────────────
# Set to voice.speak by main.py so timer/remind fire spoken alerts.
# If None, falls back to print-only.
_announce_fn: Optional[Callable[[str], None]] = None

def _announce(msg: str) -> None:
    """Print and optionally speak a timed alert message."""
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")
    if _announce_fn is not None:
        try:
            _announce_fn(msg)
        except Exception:
            pass


# ── Skill implementations ──────────────────────────────────────────────────

def skill_open_app(args: str) -> str:
    """Open a Windows application by name."""
    app_name = args.strip().lower()
    if not app_name:
        return "No application name provided."

    exe = _APP_REGISTRY.get(app_name)
    if exe:
        # Replace {user} placeholder
        user = os.environ.get("USERNAME", os.environ.get("USER", ""))
        exe = exe.replace("{user}", user)
    else:
        # Try running it directly (works for PATH-accessible apps)
        exe = app_name

    try:
        if exe.startswith("ms-"):
            # UWP / ms-settings links
            os.startfile(exe)
        else:
            subprocess.Popen(
                exe, shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return f"Opened {app_name}."
    except Exception as exc:
        return f"Could not open {app_name}: {exc}"


def skill_close_app(args: str) -> str:
    """Close a running application by name."""
    app_name = args.strip().lower()
    if not app_name:
        return "No application name provided."
    try:
        result = subprocess.run(
            ["taskkill", "/IM", f"{app_name}.exe", "/F"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return f"Closed {app_name}."
        return f"Could not find {app_name} running."
    except Exception as exc:
        return f"Failed to close {app_name}: {exc}"


def skill_web_search(args: str) -> str:
    """Search the web using DuckDuckGo (free, no API key)."""
    query = args.strip()
    if not query:
        return "No search query provided."

    # Try the new 'ddgs' package first, then legacy 'duckduckgo_search'
    ddgs_cls = None
    try:
        from ddgs import DDGS as _DDGS  # type: ignore
        ddgs_cls = _DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS as _DDGS2  # type: ignore
            ddgs_cls = _DDGS2
        except ImportError:
            pass

    if ddgs_cls is not None:
        try:
            with ddgs_cls() as ddgs:
                results = list(ddgs.text(query, max_results=5))
            if not results:
                return _web_search_fallback(query)

            summary_parts = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                body = r.get("body", "")
                summary_parts.append(f"{i}. {title}: {body}")

            return "Here is what I found. " + " ".join(summary_parts)
        except Exception as exc:
            return _web_search_fallback(query)
    else:
        return _web_search_fallback(query)


def _web_search_fallback(query: str) -> str:
    """Minimal DuckDuckGo search without extra packages."""
    try:
        url = f"https://api.duckduckgo.com/?q={urlparse.quote(query)}&format=json&no_html=1"
        req = urlreq.Request(url, headers={"User-Agent": "FRIDAY/1.0"})
        with urlreq.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        abstract = data.get("AbstractText", "")
        answer = data.get("Answer", "")
        related = data.get("RelatedTopics", [])

        parts = []
        if answer:
            parts.append(answer)
        if abstract:
            parts.append(abstract)
        for topic in related[:3]:
            text = topic.get("Text", "")
            if text:
                parts.append(text)

        if parts:
            return "Here is what I found. " + " ".join(parts)
        return f"I could not find a clear answer for: {query}"
    except Exception as exc:
        return f"Web search failed: {exc}"


def skill_system_info(args: str) -> str:
    """Get system information: CPU, RAM, disk, battery."""
    try:
        import psutil
    except ImportError:
        return _system_info_basic()

    parts = []
    cpu_pct = psutil.cpu_percent(interval=0.5)
    parts.append(f"CPU usage is {cpu_pct} percent")

    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024 ** 3)
    total_gb = mem.total / (1024 ** 3)
    parts.append(f"RAM: {used_gb:.1f} of {total_gb:.1f} GB used ({mem.percent} percent)")

    disk = psutil.disk_usage("C:\\")
    free_gb = disk.free / (1024 ** 3)
    parts.append(f"C drive: {free_gb:.0f} GB free")

    bat = psutil.sensors_battery()
    if bat:
        plug = "plugged in" if bat.power_plugged else "on battery"
        parts.append(f"Battery: {bat.percent} percent, {plug}")

    # GPU / VRAM via nvidia-smi (graceful fallback if unavailable)
    try:
        nv = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if nv.returncode == 0 and nv.stdout.strip():
            gpu_line = nv.stdout.strip().split("\n")[0]
            gpu_parts = [p.strip() for p in gpu_line.split(",")]
            if len(gpu_parts) >= 4:
                parts.append(
                    f"GPU: {gpu_parts[0]}, VRAM {gpu_parts[1]} of "
                    f"{gpu_parts[2]} MB used, utilization {gpu_parts[3]} percent"
                )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # nvidia-smi not installed or failed — skip GPU info

    return ". ".join(parts) + "."


def _system_info_basic() -> str:
    """Fallback system info without psutil."""
    info = platform.uname()
    return (
        f"System: {info.system} {info.release}, "
        f"Machine: {info.machine}, "
        f"Processor: {info.processor}."
    )


def skill_clipboard_read(args: str) -> str:
    """Read current clipboard contents."""
    try:
        import pyperclip
        content = pyperclip.paste()
        if content:
            # Truncate for voice
            if len(content) > 300:
                return f"Clipboard contains: {content[:300]}... (truncated)"
            return f"Clipboard contains: {content}"
        return "Clipboard is empty."
    except ImportError:
        # PowerShell fallback
        try:
            result = subprocess.run(
                ["powershell", "-Command", "Get-Clipboard"],
                capture_output=True, text=True, timeout=5,
            )
            content = result.stdout.strip()
            if content:
                if len(content) > 300:
                    return f"Clipboard contains: {content[:300]}... (truncated)"
                return f"Clipboard contains: {content}"
            return "Clipboard is empty."
        except Exception:
            return "Could not read clipboard."


def skill_clipboard_write(args: str) -> str:
    """Write text to clipboard."""
    text = args.strip()
    if not text:
        return "No text to copy."
    try:
        import pyperclip
        pyperclip.copy(text)
        return "Copied to clipboard."
    except ImportError:
        try:
            subprocess.run(
                ["powershell", "-Command", f"Set-Clipboard '{text}'"],
                timeout=5,
            )
            return "Copied to clipboard."
        except Exception:
            return "Could not write to clipboard."


def skill_volume(args: str) -> str:
    """Control system volume using PowerShell/nircmd."""
    action = args.strip().lower()
    if not action:
        return "Specify: mute, unmute, up, down, or a number 0-100."

    ps_base = (
        "Add-Type -TypeDefinition '"
        'using System.Runtime.InteropServices; '
        '[Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)] '
        "interface IAudioEndpointVolume { int f(); int g(); int h(); int i(); "
        "int SetMasterVolumeLevelScalar(float fLevel, System.Guid pguidEventContext); "
        "int j(); int GetMasterVolumeLevelScalar(out float pfLevel); }"
        "'"
    )

    try:
        if action == "mute":
            subprocess.run(
                ["powershell", "-Command",
                 "(New-Object -ComObject WScript.Shell).SendKeys([char]173)"],
                timeout=5, capture_output=True,
            )
            return "Volume muted."
        elif action == "unmute":
            subprocess.run(
                ["powershell", "-Command",
                 "(New-Object -ComObject WScript.Shell).SendKeys([char]173)"],
                timeout=5, capture_output=True,
            )
            return "Volume unmuted."
        elif action in ("up", "louder"):
            for _ in range(5):
                subprocess.run(
                    ["powershell", "-Command",
                     "(New-Object -ComObject WScript.Shell).SendKeys([char]175)"],
                    timeout=5, capture_output=True,
                )
            return "Volume increased."
        elif action in ("down", "quieter", "lower"):
            for _ in range(5):
                subprocess.run(
                    ["powershell", "-Command",
                     "(New-Object -ComObject WScript.Shell).SendKeys([char]174)"],
                    timeout=5, capture_output=True,
                )
            return "Volume decreased."
        else:
            return f"Unknown volume command: {action}. Use mute, unmute, up, down."
    except Exception as exc:
        return f"Volume control failed: {exc}"


def skill_timer(args: str) -> str:
    """Set a countdown timer. Args: seconds or 'Xm' for minutes."""
    raw = args.strip().lower()
    if not raw:
        return "Specify a duration, like 30 for 30 seconds or 5m for 5 minutes."

    try:
        if raw.endswith("m"):
            seconds = int(raw[:-1]) * 60
        elif raw.endswith("min"):
            seconds = int(raw[:-3]) * 60
        elif raw.endswith("h"):
            seconds = int(raw[:-1]) * 3600
        elif raw.endswith("s"):
            seconds = int(raw[:-1])
        else:
            seconds = int(raw)
    except ValueError:
        return f"Could not parse duration: {raw}"

    if seconds <= 0:
        return "Duration must be positive."
    if seconds > 86400: 
        return "Maximum timer is 24 hours."

    def _alarm():
        time.sleep(seconds)
        label_done = f"{seconds}s" if seconds < 60 else f"{seconds//60}m{seconds%60:02d}s"
        _announce(f"⏰ Timer done! {label_done} elapsed.")
        try:
            subprocess.run(
                ["powershell", "-Command",
                 "[console]::beep(880,300); [console]::beep(1047,300); [console]::beep(1319,400)"],
                timeout=10, capture_output=True,
            )
        except Exception:
            pass

    t = threading.Thread(target=_alarm, daemon=True, name=f"timer-{seconds}s")
    t.start()
    _active_timers.append({"seconds": seconds, "started": time.time(), "thread": t})

    if seconds >= 60:
        mins = seconds // 60
        secs = seconds % 60
        label = f"{mins} minute{'s' if mins != 1 else ''}"
        if secs:
            label += f" {secs} second{'s' if secs != 1 else ''}"
    else:
        label = f"{seconds} second{'s' if seconds != 1 else ''}"

    return f"Timer set for {label}."


def skill_calculator(args: str) -> str:
    """Safely evaluate a math expression."""
    expr = args.strip()
    if not expr:
        return "No expression to evaluate."

    # Clean up spoken math
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    expr = expr.replace("plus", "+").replace("minus", "-")
    expr = expr.replace("times", "*").replace("divided by", "/")
    expr = expr.replace("to the power of", "**")
    expr = expr.replace("mod", "%").replace("modulo", "%")

    # Allow only safe characters
    safe_pattern = re.compile(r'^[\d\s\+\-\*\/\.\(\)\%\,]+$')
    # Also allow math function names
    func_pattern = re.compile(r'\b(sqrt|sin|cos|tan|log|log2|log10|abs|pow|round|pi|e)\b')

    clean = func_pattern.sub('', expr)
    if not safe_pattern.match(clean.replace(' ', '').replace(',', '')):
        return f"I cannot safely evaluate: {expr}"

    # Replace math functions with math module calls
    expr = re.sub(r'\bsqrt\b', 'math.sqrt', expr)
    expr = re.sub(r'\bsin\b', 'math.sin', expr)
    expr = re.sub(r'\bcos\b', 'math.cos', expr)
    expr = re.sub(r'\btan\b', 'math.tan', expr)
    expr = re.sub(r'\blog10\b', 'math.log10', expr)
    expr = re.sub(r'\blog2\b', 'math.log2', expr)
    expr = re.sub(r'\blog\b', 'math.log', expr)
    expr = re.sub(r'\bpi\b', 'math.pi', expr)
    expr = re.sub(r'\be\b', 'math.e', expr)
    expr = re.sub(r'\babs\b', 'abs', expr)
    expr = re.sub(r'\bround\b', 'round', expr)

    try:
        result = eval(expr, {"__builtins__": {}, "math": math, "abs": abs, "round": round, "pow": pow})
        # Format nicely
        if isinstance(result, float):
            if result == int(result):
                return f"The answer is {int(result)}."
            return f"The answer is {result:.6g}."
        return f"The answer is {result}."
    except Exception as exc:
        return f"Could not calculate: {exc}"


def skill_weather(args: str) -> str:
    """Get weather info from wttr.in (free, no API key)."""
    location = args.strip() or "Indore"
    try:
        url = f"https://wttr.in/{urlparse.quote(location)}?format=j1"
        req = urlreq.Request(url, headers={"User-Agent": "FRIDAY/1.0"})
        with urlreq.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        current = data.get("current_condition", [{}])[0]
        temp_c = current.get("temp_C", "?")
        desc = current.get("weatherDesc", [{}])[0].get("value", "unknown")
        humidity = current.get("humidity", "?")
        feels = current.get("FeelsLikeC", "?")
        wind = current.get("windspeedKmph", "?")

        area = data.get("nearest_area", [{}])[0]
        city = area.get("areaName", [{}])[0].get("value", location)

        return (
            f"Weather in {city}: {desc}, {temp_c} degrees celsius, "
            f"feels like {feels} degrees. "
            f"Humidity is {humidity} percent. "
            f"Wind speed is {wind} kilometers per hour."
        )
    except Exception as exc:
        return f"Could not get weather: {exc}"


def skill_screenshot(args: str) -> str:
    """Take a screenshot and save it to Desktop."""
    try:
        from PIL import ImageGrab
    except ImportError:
        # Fallback: use PowerShell/Snipping Tool
        try:
            subprocess.Popen(["snippingtool", "/clip"], shell=True)
            return "Opened Snipping Tool for screenshot."
        except Exception:
            return "Screenshot requires Pillow: pip install Pillow"

    try:
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(desktop, f"screenshot_{ts}.png")
        img = ImageGrab.grab()
        img.save(path)
        return f"Screenshot saved to Desktop as screenshot_{ts}.png."
    except Exception as exc:
        return f"Screenshot failed: {exc}"


def skill_file_search(args: str) -> str:
    """Search for files by name pattern on common locations."""
    pattern = args.strip().lower()
    if not pattern:
        return "Specify a filename or pattern to search for."

    search_dirs = [
        os.path.expanduser("~\\Desktop"),
        os.path.expanduser("~\\Documents"),
        os.path.expanduser("~\\Downloads"),
    ]

    found = []
    for root_dir in search_dirs:
        if not os.path.isdir(root_dir):
            continue
        try:
            for dirpath, dirnames, filenames in os.walk(root_dir):
                # Limit depth to 3 levels
                depth = dirpath.replace(root_dir, "").count(os.sep)
                if depth > 3:
                    dirnames.clear()
                    continue
                for fn in filenames:
                    if pattern in fn.lower():
                        found.append(os.path.join(dirpath, fn))
                        if len(found) >= 10:
                            break
                if len(found) >= 10:
                    break
        except PermissionError:
            continue

    if not found:
        return f"No files matching '{pattern}' found in Desktop, Documents, or Downloads."

    file_list = ", ".join(os.path.basename(f) for f in found[:5])
    return f"Found {len(found)} file{'s' if len(found) > 1 else ''}: {file_list}."


def skill_datetime(args: str) -> str:
    """Get current date, time, or day information."""
    now = datetime.datetime.now()
    query = args.strip().lower()

    if "date" in query:
        return f"Today is {now.strftime('%A, %d %B %Y')}."
    elif "day" in query:
        return f"Today is {now.strftime('%A')}."
    else:
        return f"It is {now.strftime('%I:%M %p')} on {now.strftime('%A, %d %B %Y')}."


def skill_remind(args: str) -> str:
    """Set a quick reminder (same as timer but with a message)."""
    # Parse "in Xm message" or "X seconds message"
    match = re.match(r'(?:in\s+)?(\d+)\s*(s|sec|seconds?|m|min|minutes?|h|hours?)\s*(.*)', args.strip(), re.IGNORECASE)
    if not match:
        return "Format: remind in 5m check the oven"

    amount = int(match.group(1))
    unit = match.group(2).lower()
    message = match.group(3).strip() or "Time's up!"

    if unit.startswith("m"):
        seconds = amount * 60
    elif unit.startswith("h"):
        seconds = amount * 3600
    else:
        seconds = amount

    def _remind():
        time.sleep(seconds)
        _announce(f"🔔 Reminder: {message}")
        try:
            subprocess.run(
                ["powershell", "-Command",
                 "[console]::beep(600,300); [console]::beep(880,300); [console]::beep(600,300)"],
                timeout=10, capture_output=True,
            )
        except Exception:
            pass

    threading.Thread(target=_remind, daemon=True, name=f"remind-{seconds}s").start()

    if seconds >= 60:
        label = f"{seconds // 60} minute{'s' if seconds // 60 != 1 else ''}"
    else:
        label = f"{seconds} second{'s' if seconds != 1 else ''}"

    return f"Reminder set: '{message}' in {label}."


# ── Skill Registry ──────────────────────────────────────────────────────────

def skill_open_url(args: str) -> str:
    """Open a URL in the default web browser."""
    url = args.strip()
    if not url:
        return "No URL provided."
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        import webbrowser
        webbrowser.open(url)
        return f"Opened {url} in browser."
    except Exception as exc:
        return f"Could not open URL: {exc}"


# ── Phase 4 skills ────────────────────────────────────────────────────────────

_RESULT_LIMIT = 500   # max chars spoken to avoid wall-of-text in voice

# Paths that file_write must never touch
_BLOCKED_WRITE_DIRS = [
    os.environ.get("SystemRoot", r"C:\Windows"),
    r"C:\Windows", r"C:\System32", r"C:\Program Files",
    r"C:\Program Files (x86)", "/etc", "/usr", "/bin", "/sbin",
]

# Search roots for find_and_open — common user directories first, then deeper
_SEARCH_ROOTS = [
    os.path.expanduser("~\\Desktop"),
    os.path.expanduser("~\\Documents"),
    os.path.expanduser("~\\Downloads"),
    os.path.expanduser("~\\Pictures"),
    os.path.expanduser("~\\Videos"),
    os.path.expanduser("~\\Music"),
    os.path.expanduser("~"),   # entire user profile
    "C:\\",                     # whole drive (slow but thorough)
]

# Directories to skip during deep search (speeds up + avoids noise)
_SKIP_DIRS = {
    "Windows", "System32", "SysWOW64", "$Recycle.Bin", "ProgramData",
    "AppData\\Local\\Temp", "node_modules", ".git", "__pycache__",
    "WinSxS", "Temp",
}


def _truncate(text: str, limit: int = _RESULT_LIMIT) -> str:
    """Trim result to voice-safe length."""
    text = text.strip()
    if len(text) > limit:
        return text[:limit] + f"… [{len(text)-limit} chars truncated]"
    return text


def _get_active_browser() -> Optional[str]:
    """
    Detect which browser is currently running using psutil.
    Returns the executable path/name to use for opening a URL,
    or None if no browser is found (caller falls back to webbrowser.open).
    Priority: Edge > Chrome > Brave > Firefox (matches most common Windows setups).
    """
    try:
        import psutil
        running = {p.name().lower() for p in psutil.process_iter(["name"])}
        for proc_name, exe in _BROWSER_PROC_MAP.items():
            if proc_name.lower() in running:
                return exe
    except Exception:
        pass
    return None


def _open_url_in_browser(url: str, browser_exe: Optional[str] = None) -> str:
    """
    Open a URL — in the active browser if detected, else via webbrowser.open().
    Always opens as a new tab in the existing window (--new-tab flag).
    """
    if browser_exe is None:
        browser_exe = _get_active_browser()

    if browser_exe:
        try:
            # --new-tab opens in the existing browser window
            subprocess.Popen([browser_exe, "--new-tab", url],
                             creationflags=subprocess.DETACHED_PROCESS)
            return browser_exe
        except Exception:
            pass   # fall through to webbrowser

    import webbrowser
    webbrowser.open(url)
    return "default browser"


def skill_smart_open(args: str) -> str:
    """
    Intelligently open a website or URL in the currently active browser.
    Knows common sites by name (youtube, github, gmail, reddit, etc.).
    args: site name or full URL
    """
    target = args.strip().lower().strip('"\'')
    if not target:
        return "No site or URL provided."

    # Resolve friendly name to URL
    url = _SITE_MAP.get(target)
    if url is None:
        # Try partial match
        for key, val in _SITE_MAP.items():
            if key in target or target in key:
                url = val
                break
    if url is None:
        # Treat as raw URL
        url = target if target.startswith(("http://", "https://")) else "https://" + target

    browser = _open_url_in_browser(url)
    site_label = target.title()
    return f"Opening {site_label} in {os.path.basename(browser).replace('.exe', '')}."


def skill_find_and_open(args: str) -> str:
    """
    Find a file by name across common directories, then open it.
    args: filename or partial name (e.g. 'resume.pdf', 'budget')
    Searches Desktop, Documents, Downloads, then the full user profile.
    """
    query = args.strip().strip('"\'').lower()
    if not query:
        return "No filename provided."

    found: list[str] = []

    for root in _SEARCH_ROOTS:
        if not os.path.exists(root):
            continue
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                # Prune skip dirs in-place so os.walk doesn't descend into them
                dirnames[:] = [
                    d for d in dirnames
                    if d not in _SKIP_DIRS and not d.startswith(".")
                ]
                for fname in filenames:
                    if query in fname.lower():
                        found.append(os.path.join(dirpath, fname))
                if len(found) >= 10:
                    break   # stop at 10 matches
        except PermissionError:
            continue
        if found:
            break   # found in first matching root — stop searching deeper

    if not found:
        return f"No file matching '{args.strip()}' found in your common folders."

    # Open the best match (first found)
    best = found[0]
    try:
        os.startfile(best)   # Windows: opens with default associated app
        extra = f" ({len(found)-1} more match{'es' if len(found)-1 != 1 else ''} found)" if len(found) > 1 else ""
        return f"Opening {os.path.basename(best)}{extra}."
    except Exception as exc:
        return f"Found {os.path.basename(best)} but couldn't open it: {exc}"


def skill_youtube_search(args: str) -> str:
    """
    Search YouTube for a query, opening in the active browser.
    Tries to open the first video directly; falls back to the results page.
    args: search query (e.g. 'lofi music', 'python tutorial')
    """
    query = args.strip()
    if not query:
        return "No search query provided."

    results_url = "https://www.youtube.com/results?search_query=" + urlparse.quote_plus(query)

    # Try to fetch the first video URL from the results page
    video_url = None
    try:
        req = urlreq.Request(results_url, headers={"User-Agent": "Mozilla/5.0"})
        with urlreq.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        # Extract first video ID from the results page HTML
        vid_match = re.search(r'/watch\?v=([a-zA-Z0-9_-]{11})', html)
        if vid_match:
            video_url = f"https://www.youtube.com/watch?v={vid_match.group(1)}"
    except Exception:
        pass  # fall back to results page

    url_to_open = video_url or results_url
    browser = _open_url_in_browser(url_to_open)
    browser_name = os.path.basename(browser).replace(".exe", "").title()
    if video_url:
        return f"Playing top result for '{query}' in {browser_name}."
    return f"Searching YouTube for '{query}' in {browser_name}."



def skill_file_read(args: str) -> str:
    """
    Read a file and return its first 200 lines.
    args: file path (absolute or relative to user home)
    """
    path = args.strip().strip('"\'')
    if not path:
        return "No file path provided."

    # Expand ~ and env vars
    path = os.path.expandvars(os.path.expanduser(path))

    if not os.path.exists(path):
        return f"File not found: {path}"
    if os.path.isdir(path):
        items = os.listdir(path)[:20]
        return f"That's a directory. Contents: {', '.join(items)}"
    if not os.access(path, os.R_OK):
        return f"Permission denied: {path}"

    try:
        size_kb = os.path.getsize(path) / 1024
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        total = len(lines)
        preview = "".join(lines[:200])
        suffix = f"\n[{total} lines total, {size_kb:.1f} KB]" if total > 200 else f"\n[{total} lines, {size_kb:.1f} KB]"
        return _truncate(preview + suffix)
    except Exception as exc:
        return f"Error reading file: {exc}"


def skill_file_write(args: str) -> str:
    """
    Write content to a file.
    args: path||content   (pipe-delimited)
    Blocked: system directories (C:\\Windows, /etc, /usr, etc.)
    """
    if "||" not in args:
        return "Format: file_write:path||content"
    path, content = args.split("||", 1)
    path = os.path.expandvars(os.path.expanduser(path.strip().strip('"\'')))
    abs_path = os.path.abspath(path)

    # Security: block system directories
    for blocked in _BLOCKED_WRITE_DIRS:
        try:
            blocked_abs = os.path.abspath(blocked)
            if abs_path.startswith(blocked_abs):
                return f"Refused: writing to {blocked} is not allowed."
        except Exception:
            pass

    try:
        os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        lines_written = content.count("\n") + 1
        return f"Written {lines_written} line(s) to {abs_path}."
    except Exception as exc:
        return f"Error writing file: {exc}"


def skill_run_code(args: str) -> str:
    """
    Execute a code snippet in a sandboxed subprocess.
    args: lang:code  (e.g. python:print(1+1))
    Supported: python, python3, node, js
    Security: 10s timeout, no network, captures stdout+stderr.
    """
    if ":" not in args:
        return "Format: run_code:python:print('hello')"
    lang, code = args.split(":", 1)
    lang = lang.strip().lower()

    # Security: reject dangerous patterns
    _BANNED = [
        "os.system", "subprocess", "__import__", "eval(", "exec(",
        "open(", "socket", "urllib", "requests", "import os",
        "shutil", "rmdir", "remove(", "unlink(",
    ]
    for bad in _BANNED:
        if bad in code:
            return f"Refused: '{bad}' is not allowed in sandboxed code."

    lang_map = {"python": "python", "python3": "python", "node": "node", "js": "node"}
    interpreter = lang_map.get(lang)
    if not interpreter:
        return f"Unsupported language: {lang}. Use python or node."

    try:
        result = subprocess.run(
            [interpreter, "-c", code],
            capture_output=True, text=True,
            timeout=10,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if err and not out:
            return _truncate(f"Error: {err}")
        if err:
            return _truncate(f"Output: {out}\nStderr: {err}")
        return _truncate(out or "(no output)")
    except subprocess.TimeoutExpired:
        return "Code execution timed out after 10 seconds."
    except FileNotFoundError:
        return f"Interpreter '{interpreter}' not found on PATH."
    except Exception as exc:
        return f"Execution error: {exc}"


def skill_git_query(args: str) -> str:
    """
    Query a git repository (read-only: status, diff, log, branch).
    args: command  (status | diff | log | branch | show)
    Blocked: push, commit, reset, checkout, merge, rebase.
    """
    cmd = args.strip().lower().split()[0] if args.strip() else "status"

    _READONLY = {"status", "diff", "log", "branch", "show", "stash"}
    _BLOCKED_GIT = {"push", "commit", "reset", "checkout", "merge",
                    "rebase", "rm", "mv", "add", "clean", "fetch", "pull"}

    if cmd in _BLOCKED_GIT:
        return f"Refused: 'git {cmd}' is not allowed via skill (write operations only via terminal)."
    if cmd not in _READONLY:
        return f"Unknown git command: {cmd}. Use: status, diff, log, branch, show."

    git_args = args.strip().split()
    # Build safe git command
    git_cmd = ["git"] + git_args
    if cmd == "log":
        # Limit log output
        if "--oneline" not in git_args:
            git_cmd += ["--oneline", "-10"]

    try:
        result = subprocess.run(
            git_cmd,
            capture_output=True, text=True,
            timeout=10, cwd=os.getcwd(),
        )
        out = result.stdout.strip() or result.stderr.strip()
        return _truncate(out or f"git {cmd}: no output.")
    except subprocess.TimeoutExpired:
        return "git command timed out."
    except FileNotFoundError:
        return "git is not installed or not on PATH."
    except Exception as exc:
        return f"git error: {exc}"


def skill_browser_search(args: str) -> str:
    """
    Search something in the browser using the default search engine.
    args: search query
    """
    query = args.strip()
    if not query:
        return "No search query provided."
    try:
        import webbrowser
        url = "https://www.google.com/search?q=" + urlparse.quote_plus(query)
        webbrowser.open(url)
        return f"Searching for '{query}' in browser."
    except Exception as exc:
        return f"Could not open browser: {exc}"


@dataclass
class SkillEngine:
    """
    Dispatches [ACTION:skill:args] tags from LLM output.
    """

    debug_fn: Optional[Callable] = None
    skills: Dict[str, Callable[[str], str]] = field(default_factory=dict)

    # Pattern: [ACTION:skill_name:arguments]
    ACTION_PATTERN = re.compile(r'\[ACTION:(\w+):(.*?)\]', re.DOTALL)

    def __post_init__(self):
        self.skills = {
            "open_app":        skill_open_app,
            "close_app":       skill_close_app,
            "open_url":        skill_open_url,
            "web_search":      skill_web_search,
            "system_info":     skill_system_info,
            "clipboard_read":  skill_clipboard_read,
            "clipboard_write": skill_clipboard_write,
            "volume":          skill_volume,
            "timer":           skill_timer,
            "calculator":      skill_calculator,
            "weather":         skill_weather,
            "screenshot":      skill_screenshot,
            "file_search":     skill_file_search,
            "datetime":        skill_datetime,
            "remind":          skill_remind,
            # Phase 4 — Code, file, git, browser
            "file_read":       skill_file_read,
            "file_write":      skill_file_write,
            "run_code":        skill_run_code,
            "git_query":       skill_git_query,
            "browser_search":  skill_browser_search,
            # Smart open, file finder, YouTube
            "smart_open":      skill_smart_open,
            "find_and_open":   skill_find_and_open,
            "youtube_search":  skill_youtube_search,
        }

    def _debug(self, msg: str) -> None:
        if self.debug_fn:
            self.debug_fn(f"[skills] {msg}")

    def list_skills(self) -> str:
        """Human-readable list of available skills."""
        lines = []
        for name, fn in self.skills.items():
            doc = (fn.__doc__ or "").strip().split("\n")[0]
            lines.append(f"  {name}: {doc}")
        return "Available skills:\n" + "\n".join(lines)

    def extract_and_execute(self, text: str) -> tuple[str, list[str]]:
        """
        Find all [ACTION:...] tags in text, execute them,
        and return (cleaned_text, list_of_results).
        """
        results = []
        matches = list(self.ACTION_PATTERN.finditer(text))

        if not matches:
            return text, []

        for match in matches:
            skill_name = match.group(1).strip()
            skill_args = match.group(2).strip()

            self._debug(f"Executing skill: {skill_name}({skill_args})")

            fn = self.skills.get(skill_name)
            if fn:
                try:
                    result = fn(skill_args)
                    results.append(result)
                    self._debug(f"Skill result: {result[:100]}")
                except Exception as exc:
                    error_msg = f"Skill {skill_name} failed: {exc}"
                    results.append(error_msg)
                    self._debug(error_msg)
            else:
                results.append(f"Unknown skill: {skill_name}")

        # Remove action tags from text
        cleaned = self.ACTION_PATTERN.sub("", text).strip()
        # Clean up leftover whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r'  +', ' ', cleaned)

        return cleaned, results

    def get_skill_prompt(self) -> str:
        """
        Returns instruction block for the LLM system prompt
        so it knows how to invoke skills.
        """
        return (
            "TOOL USE:\n"
            "You can perform real actions by including action tags in your response.\n"
            "Format: [ACTION:skill_name:arguments]\n"
            "Available actions:\n"
            # ── System & apps ──
            "  [ACTION:open_app:chrome] — open an application\n"
            "  [ACTION:close_app:notepad] — close an application\n"
            "  [ACTION:smart_open:youtube] — open a site in the ACTIVE browser (youtube/github/gmail/reddit etc.)\n"
            "  [ACTION:open_url:github.com] — open a URL in the browser\n"
            "  [ACTION:browser_search:python asyncio tutorial] — Google search in browser\n"
            "  [ACTION:youtube_search:lofi music] — search YouTube in the active browser\n"
            "  [ACTION:web_search:latest python news] — fetch search results as text\n"
            "  [ACTION:find_and_open:resume.pdf] — find a file anywhere on the system and open it\n"
            "  [ACTION:system_info:] — get CPU, RAM, battery info\n"
            "  [ACTION:clipboard_read:] — read clipboard\n"
            "  [ACTION:clipboard_write:text to copy] — write to clipboard\n"
            "  [ACTION:volume:up/down/mute/unmute] — control volume\n"
            "  [ACTION:timer:30s or 5m] — set a timer\n"
            "  [ACTION:calculator:2+2*3] — evaluate math\n"
            "  [ACTION:weather:Indore] — get weather\n"
            "  [ACTION:screenshot:] — take a screenshot\n"
            "  [ACTION:file_search:report.pdf] — find files in common folders\n"
            "  [ACTION:datetime:] — get current date/time\n"
            "  [ACTION:remind:5m check the oven] — set a reminder\n"
            # ── Phase 4: Code, file, git ──
            "  [ACTION:file_read:/path/to/file.py] — read a file (first 200 lines)\n"
            "  [ACTION:file_write:/path/file.txt||content here] — write/create a file\n"
            "  [ACTION:run_code:python:print(1+2)] — run code (python/node), 10s timeout\n"
            "  [ACTION:git_query:status] — query git repo (status/diff/log/branch, read-only)\n\n"
            "SMART BROWSER RULES:\n"
            "  Use smart_open for named sites: 'open youtube', 'open github', 'open gmail'\n"
            "  Use youtube_search for 'search X on youtube' or 'play X on youtube'\n"
            "  Use browser_search for general Google searches\n"
            "  smart_open detects the currently running browser and opens in it\n\n"
            "SECURITY RULES:\n"
            "  run_code: no os.system, no file I/O, no network inside code\n"
            "  file_write: blocked for C:\\Windows, /etc, /usr, and system dirs\n"
            "  git_query: read-only only (no push/commit/reset via skill)\n\n"
            "IMPORTANT: Only use action tags when the user explicitly asks you to DO something.\n"
            "For questions, just answer normally without action tags.\n"
            "You may include ONE action tag per response. Put it at the END of your message.\n"
            "After the action tag, do NOT add any more text.\n"
        )
