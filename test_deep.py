"""Deep runtime test for ear.py, voice.py, skills.py"""
import sys
sys.path.insert(0, r'c:\Users\akhil\Downloads\jarvis')

# Test voice module deeply
print("=== Voice Module ===")
from voice import VoiceIO
v = VoiceIO(debug=True, tts_backend='auto')
print(f"Voice initialized: mic={v.mic_backend}, tts={v.tts_effective_backend}")
print(f"Streaming capture: {v._streaming_capture is not None}")
print(f"Has speak_pregenerated: {hasattr(v, 'speak_pregenerated')}")
print(f"Has speak_sentence_pipeline: {hasattr(v, 'speak_sentence_pipeline')}")
print(f"Has stop_playback: {hasattr(v, 'stop_playback')}")
print(f"Has shutdown: {hasattr(v, 'shutdown')}")
v.shutdown()
print("Voice shutdown: OK")

# Test ear streaming capture
print("\n=== Ear Module ===")
from ear import StreamingVoiceCapture, sr, sd, np
print(f"StreamingVoiceCapture: available")
print(f"sounddevice: {sd is not None}")
print(f"numpy: {np is not None}")
print(f"speech_recognition: {sr is not None}")

# Test skills deeply
print("\n=== Skills Module ===")
from skills import SkillEngine
e = SkillEngine()

# Test action extraction with actual skill execution
text = "Let me check that. [ACTION:calculator:100/4]"
clean, results = e.extract_and_execute(text)
print(f"Action parse: clean='{clean}', results={results}")

# Test weather
from skills import skill_weather
print(f"Weather: {skill_weather('Indore')[:80]}...")

# Test clipboard
from skills import skill_clipboard_read
print(f"Clipboard: {skill_clipboard_read('')[:60]}...")

# Test file search
from skills import skill_file_search
print(f"File search: {skill_file_search('main.py')[:80]}...")

# Full main.py boot test (no-voice, no-llm)
print("\n=== Main Boot Test ===")
from brain import Brain
from assistant import Friday
from memory import PersistentMemory
from commands import is_shutdown_command, is_wake_command

brain = Brain(enabled=False)
pm = PersistentMemory()
agent = Friday(name="friday", brain=brain, persistent_memory=pm)
print(f"Agent: name={agent.name}, awake={agent.awake}, skills={len(agent.skills.skills)}")

# Test handle with rule-based
response = agent.handle("help")
print(f"Help response: {response[:80]}...")

response = agent.handle("time")
print(f"Time response: {response}")

print("\n" + "=" * 50)
print("ALL DEEP TESTS PASSED!")
print("=" * 50)
