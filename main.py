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
YOLO_SCRIPT   = os.path.join(ENGINES, "yolo.py")
BOOK_SCRIPT   = os.path.join(ENGINES, "bookreader.py")
NAV_SCRIPT    = os.path.join(ENGINES, "navigation.py")

# ─────────────────────────────────────────────
# TRACK CHILD PROCESSES
# ─────────────────────────────────────────────
_child_processes = []

# ─────────────────────────────────────────────
# GLOBAL LISTENER FLAG
# stops main.py mic when any option is opened
# ─────────────────────────────────────────────
_listening = True

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
    _tts_ready.set()

    while True:
        text = _tts_queue.get()
        if text is None:
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
    _tts_ready.wait(timeout=10)
    print("[TTS] Engine ready.")

def speak(text: str):
    """Non-blocking: queue text for speech."""
    print(f"[TTS] {text}")
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
    process = subprocess.Popen([sys.executable, script_path])
    _child_processes.append(process)

# ─────────────────────────────────────────────
# LAUNCH HELPERS — stop listener before opening
# ─────────────────────────────────────────────
def launch_navigation():
    global _listening
    _listening = False
    open_navigation_window()
    _launch(NAV_SCRIPT)
    stop_tts()

def launch_notice():
    global _listening
    _listening = False
    open_notice_window()
    _launch(NOTICE_SCRIPT)
    stop_tts()

def launch_detection():
    global _listening
    _listening = False
    open_detection_window()
    _launch(YOLO_SCRIPT)
    stop_tts()

def launch_book():
    global _listening
    _listening = False
    open_book_window()
    _launch(BOOK_SCRIPT)
    stop_tts()

def restart_listener():
    global _listening
    _listening = True
    speak("What would you like to do next? You can say notice, object detection, book reading, or navigation.")
    threading.Thread(target=listen_command, daemon=True).start()
# ─────────────────────────────────────────────
# SHARED SCROLLABLE WINDOW BUILDER
# ─────────────────────────────────────────────
def make_window(title, heading, subtitle, accent="#00ff88"):
    win = tk.Toplevel()
    win.title(title)
    win.geometry("600x600")
    win.configure(bg="#0a0a0a")
    win.resizable(False, True)

    canvas = tk.Canvas(win, bg="#0a0a0a", highlightthickness=0)
    scrollbar = tk.Scrollbar(win, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas, bg="#0a0a0a")

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw", width=600)
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    win.bind("<MouseWheel>", on_mousewheel)

    tk.Frame(scroll_frame, bg=accent, height=3).pack(fill="x")

    tk.Label(
        scroll_frame,
        text="◈  P A T H M A T E",
        font=("Courier", 11, "bold"),
        fg=accent,
        bg="#0a0a0a"
    ).pack(pady=(25, 0))

    tk.Label(
        scroll_frame,
        text=heading,
        font=("Courier", 28, "bold"),
        fg="white",
        bg="#0a0a0a"
    ).pack()

    tk.Label(
        scroll_frame,
        text=subtitle,
        font=("Courier", 9),
        fg="#555555",
        bg="#0a0a0a"
    ).pack(pady=(4, 20))

    tk.Frame(scroll_frame, bg="#1a1a1a", height=1).pack(fill="x", padx=40)

    return scroll_frame, accent, win

def add_steps(frame, accent, steps):
    for num, text in steps:
        row = tk.Frame(frame, bg="#0a0a0a")
        row.pack(fill="x", padx=40, pady=3)
        tk.Label(
            row,
            text=num,
            font=("Courier", 9, "bold"),
            fg=accent,
            bg="#0a0a0a",
            width=3
        ).pack(side="left")
        tk.Label(
            row,
            text=text,
            font=("Courier", 10),
            fg="#cccccc",
            bg="#0a0a0a"
        ).pack(side="left", padx=8)

def add_close_button(frame, accent, win):
    tk.Frame(frame, bg="#1a1a1a", height=1).pack(fill="x", padx=40, pady=20)
    tk.Button(
        frame,
        text="✕  CLOSE",
        font=("Courier", 11, "bold"),
        fg="#0a0a0a",
        bg=accent,
        activebackground="#00cc66",
        width=20,
        height=1,
        bd=0,
        command=lambda: [win.destroy(), restart_listener()]
    ).pack(pady=10)

# ─────────────────────────────────────────────
# NAVIGATION WINDOW
# ─────────────────────────────────────────────
def open_navigation_window():
    frame, accent, win = make_window(
        "PathMate Navigation",
        "NAVIGATION",
        "Voice-guided campus navigation for everyone"
    )

    status_frame = tk.Frame(frame, bg="#111111", pady=16, padx=20)
    status_frame.pack(fill="x", padx=40, pady=20)
    tk.Label(status_frame, text="● LISTENING",
             font=("Courier", 11, "bold"), fg="#00ff88", bg="#111111").pack(side="left")
    tk.Label(status_frame, text="Navigation engine starting...",
             font=("Courier", 10), fg="#888888", bg="#111111").pack(side="right")

    tk.Label(frame, text="HOW TO USE",
             font=("Courier", 10, "bold"), fg=accent,
             bg="#0a0a0a").pack(anchor="w", padx=40, pady=(10, 6))

    add_steps(frame, accent, [
        ("01", "Say your current location when asked"),
        ("02", "Say your destination clearly"),
        ("03", "Follow the spoken directions"),
        ("04", "Say NEXT at each checkpoint to continue"),
        ("05", "Say SIGN to scan a nearby sign with camera"),
    ])

    tk.Frame(frame, bg="#1a1a1a", height=1).pack(fill="x", padx=40, pady=16)

    tk.Label(frame, text="KNOWN LOCATIONS  —  say these exactly",
             font=("Courier", 9, "bold"), fg="#555555",
             bg="#0a0a0a").pack(anchor="w", padx=40, pady=(0, 8))

    locations = ["ENTRANCE", "BLOCK A", "BLOCK B", "BLOCK E",
                 "B 201", "B 202", "LABS", "GROUND", "BOYS HOSTEL"]
    loc_frame = tk.Frame(frame, bg="#0a0a0a")
    loc_frame.pack(fill="x", padx=40, pady=(0, 10))
    for i, loc in enumerate(locations):
        tk.Label(loc_frame, text=loc, font=("Courier", 9),
                 fg="#444444", bg="#111111", padx=8, pady=4).grid(
                 row=i//3, column=i%3, padx=3, pady=2, sticky="w")

    add_close_button(frame, accent, win)

# ─────────────────────────────────────────────
# NOTICE WINDOW
# ─────────────────────────────────────────────
def open_notice_window():
    frame, accent, win = make_window(
        "PathMate Notice Reader",
        "NOTICE READER",
        "OCR-powered notice board reader",
        accent="#00aaff"
    )

    tk.Label(frame, text="HOW TO USE",
             font=("Courier", 10, "bold"), fg=accent,
             bg="#0a0a0a").pack(anchor="w", padx=40, pady=(20, 6))

    add_steps(frame, accent, [
        ("01", "Point your camera at a notice board"),
        ("02", "Hold steady for 2-3 seconds"),
        ("03", "App will read the notice aloud"),
        ("04", "Move to next notice and repeat"),
    ])

    status_frame = tk.Frame(frame, bg="#111111", pady=16, padx=20)
    status_frame.pack(fill="x", padx=40, pady=20)
    tk.Label(status_frame, text="● CAMERA READY",
             font=("Courier", 11, "bold"), fg=accent, bg="#111111").pack(side="left")
    tk.Label(status_frame, text="Notice reader starting...",
             font=("Courier", 10), fg="#888888", bg="#111111").pack(side="right")

    add_close_button(frame, accent, win)

# ─────────────────────────────────────────────
# OBJECT DETECTION WINDOW
# ─────────────────────────────────────────────
def open_detection_window():
    frame, accent, win = make_window(
        "PathMate Object Detection",
        "OBJECT DETECT",
        "Real-time obstacle and object detection",
        accent="#ff6600"
    )

    tk.Label(frame, text="HOW TO USE",
             font=("Courier", 10, "bold"), fg=accent,
             bg="#0a0a0a").pack(anchor="w", padx=40, pady=(20, 6))

    add_steps(frame, accent, [
        ("01", "Point your camera forward while walking"),
        ("02", "App detects people, doors, furniture"),
        ("03", "Listen for audio warnings about obstacles"),
        ("04", "Press Q to quit detection mode"),
    ])

    tk.Label(frame, text="DETECTS",
             font=("Courier", 9, "bold"), fg="#555555",
             bg="#0a0a0a").pack(anchor="w", padx=40, pady=(16, 8))

    categories = [
        ("●", "#00ff88", "Persons"),
        ("●", "#ff6600", "Furniture"),
        ("●", "#00aaff", "Vehicles"),
        ("●", "#ffff00", "Traffic Signs"),
        ("●", "#ff00ff", "Appliances"),
        ("●", "#00ffff", "Books"),
    ]

    cat_frame = tk.Frame(frame, bg="#0a0a0a")
    cat_frame.pack(fill="x", padx=40, pady=(0, 10))
    for i, (dot, color, label) in enumerate(categories):
        row = tk.Frame(cat_frame, bg="#0a0a0a")
        row.grid(row=i//2, column=i%2, sticky="w", padx=10, pady=2)
        tk.Label(row, text=dot, font=("Courier", 10),
                 fg=color, bg="#0a0a0a").pack(side="left")
        tk.Label(row, text=f"  {label}", font=("Courier", 9),
                 fg="#888888", bg="#0a0a0a").pack(side="left")

    add_close_button(frame, accent, win)

# ─────────────────────────────────────────────
# BOOK READING WINDOW
# ─────────────────────────────────────────────
def open_book_window():
    frame, accent, win = make_window(
        "PathMate Book Reader",
        "BOOK READER",
        "Point camera at any text to have it read aloud",
        accent="#cc44ff"
    )

    tk.Label(frame, text="HOW TO USE",
             font=("Courier", 10, "bold"), fg=accent,
             bg="#0a0a0a").pack(anchor="w", padx=40, pady=(20, 6))

    add_steps(frame, accent, [
        ("01", "Open a book or document"),
        ("02", "Point camera at the page clearly"),
        ("03", "Hold steady — app reads text aloud"),
        ("04", "Turn page and repeat"),
    ])

    status_frame = tk.Frame(frame, bg="#111111", pady=16, padx=20)
    status_frame.pack(fill="x", padx=40, pady=20)
    tk.Label(status_frame, text="● OCR READY",
             font=("Courier", 11, "bold"), fg=accent, bg="#111111").pack(side="left")
    tk.Label(status_frame, text="Book reader starting...",
             font=("Courier", 10), fg="#888888", bg="#111111").pack(side="right")

    add_close_button(frame, accent, win)

# ─────────────────────────────────────────────
# VOICE COMMAND LISTENER  (background thread)
# ─────────────────────────────────────────────
def listen_command():
    global _listening
    recognizer = sr.Recognizer()

    while _listening:  # stops when any option is opened
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                speak("Please say: notice, object detection, book reading, or navigation")

                try:
                    audio   = recognizer.listen(source)
                    command = recognizer.recognize_google(audio).lower()
                    print(f"[Voice] Heard: {command}")

                    if not _listening:  # check again after listening
                        break

                    if "notice" in command:
                        speak("Opening notices")
                        time.sleep(1)
                        launch_notice()
                        break

                    elif any(w in command for w in ("object", "detection", "object detection")):
                        speak("Opening object detection")
                        time.sleep(1)
                        launch_detection()
                        break

                    elif any(w in command for w in ("book", "reading", "book reading")):
                        speak("Opening book reader")
                        time.sleep(1)
                        launch_book()
                        break

                    elif any(w in command for w in ("navigation", "navigate", "directions", "nav", "guide", "path")):
                        speak("Opening navigation")
                        time.sleep(1)
                        launch_navigation()
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
# WELCOME
# ─────────────────────────────────────────────
def welcome():
    speak("Welcome to PathMate. "
          "Do you want to read notices, open object detection, book reading, or navigation?")
    threading.Thread(target=listen_command, daemon=True).start()

# ─────────────────────────────────────────────
# SHUTDOWN — kills all child processes cleanly
# ─────────────────────────────────────────────
def on_close(root):
    global _listening
    _listening = False
    stop_tts()
    for process in _child_processes:
        try:
            process.terminate()
        except:
            pass
    root.destroy()
    sys.exit(0)

# ─────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────
def build_gui():
    root = tk.Tk()
    root.title("PathMate")
    root.geometry("600x600")
    root.configure(bg="#0a0a0a")
    root.resizable(False, False)

    root.lift()
    root.attributes("-topmost", True)
    root.after(600, lambda: root.attributes("-topmost", False))
    root.focus_force()

    tk.Frame(root, bg="#00ff88", height=3).pack(fill="x")

    tk.Label(
        root,
        text="◈  P A T H M A T E",
        font=("Courier", 11, "bold"),
        fg="#00ff88",
        bg="#0a0a0a"
    ).pack(pady=(25, 0))

    tk.Label(
        root,
        text="CAMPUS ASSISTANT",
        font=("Courier", 30, "bold"),
        fg="white",
        bg="#0a0a0a"
    ).pack()

    tk.Label(
        root,
        text="Accessible navigation for visually impaired students",
        font=("Courier", 9),
        fg="#555555",
        bg="#0a0a0a"
    ).pack(pady=(4, 20))

    tk.Frame(root, bg="#1a1a1a", height=1).pack(fill="x", padx=40)

    tk.Label(
        root,
        text="SAY OR CHOOSE AN OPTION",
        font=("Courier", 10, "bold"),
        fg="#00ff88",
        bg="#0a0a0a"
    ).pack(pady=(20, 12))

    tk.Button(
        root,
        text="◈  NAVIGATION",
        font=("Courier", 13, "bold"),
        width=28, height=2, bd=0,
        fg="#0a0a0a",
        bg="#00ff88",
        activebackground="#00cc66",
        activeforeground="#0a0a0a",
        cursor="hand2",
        command=launch_navigation
    ).pack(pady=6)

    outline = dict(
        font=("Courier", 12),
        width=28, height=2, bd=0,
        fg="#cccccc",
        bg="#111111",
        activebackground="#1a1a1a",
        activeforeground="white",
        cursor="hand2"
    )

    tk.Button(root, text="▸  NOTICE READER",
              command=launch_notice, **outline).pack(pady=6)

    tk.Button(root, text="▸  OBJECT DETECTION",
              command=launch_detection, **outline).pack(pady=6)

    tk.Button(root, text="▸  BOOK READING",
              command=launch_book, **outline).pack(pady=6)

    tk.Frame(root, bg="#1a1a1a", height=1).pack(fill="x", padx=40, pady=20)

    tk.Label(
        root,
        text="● SYSTEM READY",
        font=("Courier", 9, "bold"),
        fg="#00ff88",
        bg="#0a0a0a"
    ).pack()

    tk.Label(
        root,
        text="Listening for voice commands...",
        font=("Courier", 9),
        fg="#333333",
        bg="#0a0a0a"
    ).pack(pady=(4, 0))

    root.protocol("WM_DELETE_WINDOW", lambda: on_close(root))
    root.after(500, welcome)
    root.mainloop()
    stop_tts()

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    start_tts()
    build_gui()
