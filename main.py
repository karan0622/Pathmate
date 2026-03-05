"""
PathMate — Main Launcher (Cross-Platform)
==========================================
Works on macOS, Windows, and Linux — no OS-specific code.

Requirements:
    pip install pyttsx3 SpeechRecognition pyaudio
"""

import tkinter as tk
import pyttsx3
import speech_recognition as sr
import subprocess
import threading
import queue
import sys
import time
import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ENGINES       = os.path.join(BASE_DIR, "engines")

NOTICE_SCRIPT = os.path.join(ENGINES, "notice-reader.py")
NAV_SCRIPT    = os.path.join(ENGINES, "yolo.py")
BOOK_SCRIPT   = os.path.join(ENGINES, "bookreader.py")

# ─────────────────────────────────────────────
# THREAD-SAFE TTS
# pyttsx3 engine is created ONCE inside the
# worker thread and reused for every speak().
# This avoids the init-race and re-init bugs.
# ─────────────────────────────────────────────
_tts_queue = queue.Queue()
_tts_ready = threading.Event()

def _tts_worker():
    """Single persistent TTS thread — owns the engine for its lifetime."""
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
    _tts_ready.set()          # signal: engine is alive and ready

    while True:
        text = _tts_queue.get()
        if text is None:      # shutdown signal
            _tts_queue.task_done()
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[TTS Error] {e}")
        _tts_queue.task_done()

def start_tts():
    t = threading.Thread(target=_tts_worker, daemon=True)
    t.start()
    _tts_ready.wait(timeout=10)   # wait until engine is initialised (max 10s)
    print("[TTS] Engine ready.")

def speak(text: str):
    """Non-blocking: queue text for speech."""
    print(f"[TTS] {text}")
    # Drop stale queued items so nothing is backlogged
    while not _tts_queue.empty():
        try:
            _tts_queue.get_nowait()
            _tts_queue.task_done()
        except queue.Empty:
            break
    _tts_queue.put(text)

def stop_tts():
    _tts_queue.put(None)

# ─────────────────────────────────────────────
# LAUNCH ENGINE SCRIPTS
# ─────────────────────────────────────────────
def _launch(script_path: str):
    if not os.path.exists(script_path):
        speak(f"Error: script not found: {os.path.basename(script_path)}")
        print(f"[ERROR] Missing script: {script_path}")
        return
    subprocess.Popen([sys.executable, script_path])

def open_notice():
    _launch(NOTICE_SCRIPT)

def open_object_detection():
    _launch(NAV_SCRIPT)

def open_bookreader():
    _launch(BOOK_SCRIPT)

# ─────────────────────────────────────────────
# VOICE COMMAND LISTENER  (background thread)
# ─────────────────────────────────────────────
def listen_command():
    recognizer = sr.Recognizer()

    while True:
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                speak("Please say: notice, object detection, or book reading")

                try:
                    audio   = recognizer.listen(source)
                    command = recognizer.recognize_google(audio).lower()
                    print(f"[Voice] Heard: {command}")

                    if "notice" in command:
                        speak("Opening notices")
                        open_notice()
                        break

                    elif any(w in command for w in ("object", "detection", "object detection")):
                        speak("Opening object detection")
                        open_object_detection()
                        break

                    elif any(w in command for w in ("book", "reading", "book reading")):
                        speak("Opening book reader")
                        open_bookreader()
                        break

                    else:
                        speak("I did not understand. Please try again.")

                except sr.UnknownValueError:
                    speak("I could not hear you. Please speak again.")

                except sr.RequestError:
                    speak("Speech service is unavailable.")
                    break

        except OSError:
            speak("No microphone detected. Please use the buttons.")
            print("[Voice] No microphone available.")
            break

        time.sleep(0.3)

# ─────────────────────────────────────────────
# WELCOME  (runs after GUI is visible)
# ─────────────────────────────────────────────
def welcome():
    speak("Welcome to PathMate. "
          "Do you want to read notices, open object detection, or book reading?")
    threading.Thread(target=listen_command, daemon=True).start()

# ─────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────
def build_gui():
    root = tk.Tk()
    root.title("PathMate")
    root.geometry("600x500")
    root.configure(bg="black")
    root.resizable(False, False)

    root.lift()
    root.attributes("-topmost", True)
    root.after(600, lambda: root.attributes("-topmost", False))
    root.focus_force()

    tk.Label(
        root,
        text="Welcome to your PathMate",
        font=("Arial", 22, "bold"),
        fg="white",
        bg="black"
    ).pack(pady=40)

    tk.Label(
        root,
        text="Say or choose: Notice, Object Detection, or Book Reading",
        font=("Arial", 14),
        fg="white",
        bg="black"
    ).pack(pady=10)

    btn_style = dict(font=("Arial", 16), width=18, height=2)

    tk.Button(root, text="Notice",           command=open_notice,           **btn_style).pack(pady=10)
    tk.Button(root, text="Object Detection", command=open_object_detection, **btn_style).pack(pady=10)
    tk.Button(root, text="Book Reading",     command=open_bookreader,       **btn_style).pack(pady=10)

    # Fire welcome only after the engine is confirmed ready
    root.after(500, welcome)

    root.mainloop()
    stop_tts()

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    start_tts()   # blocks until pyttsx3 engine is ready
    build_gui()
