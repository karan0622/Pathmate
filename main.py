import tkinter as tk
import pyttsx3
import speech_recognition as sr
import subprocess
import threading
import sys
import time

# ---------- Text to Speech ----------
engine = pyttsx3.init()
engine.setProperty('rate', 160)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ---------- Open Files ----------
def open_notice():
    subprocess.Popen([sys.executable, "engines/notice-reader.py"])

def open_navigation():
    subprocess.Popen([sys.executable, "engines/navigation.py"])

def open_bookreader():
    subprocess.Popen([sys.executable, "/Users/karansingh22/Documents/pathmate/Pathmate/engines/bookreader.py"])

# ---------- Voice Command (REPEATING) ----------
def listen_command():
    recognizer = sr.Recognizer()

    while True:  # 🔁 repeat until valid command
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            speak("Please say notice, navigation, or book reading")

            try:
                audio = recognizer.listen(source)  # no timeout
                command = recognizer.recognize_google(audio).lower()
                print("User said:", command)

                if "notice" in command:
                    speak("Opening notices")
                    open_notice()
                    break

                elif "navigation" in command:
                    speak("Opening navigation")
                    open_navigation()
                    break

                elif "book" in command or "reading" in command or "book reading" in command:
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

            time.sleep(0.5)

# ---------- GUI ----------
root = tk.Tk()
root.title("PathMate")
root.geometry("600x500")
root.configure(bg="black")

# 🔑 FORCE WINDOW TO FRONT (macOS FIX)
root.attributes("-topmost", True)
root.after(600, lambda: root.attributes("-topmost", False))
root.focus_force()

# ---------- Heading ----------
heading = tk.Label(
    root,
    text="Welcome to your PathMate",
    font=("Arial", 22, "bold"),
    fg="white",
    bg="black"
)
heading.pack(pady=40)

# ---------- Instruction ----------
instruction = tk.Label(
    root,
    text="Say or choose: Notice, Navigation, or Book Reading",
    font=("Arial", 14),
    fg="white",
    bg="black"
)
instruction.pack(pady=10)

# ---------- Buttons ----------
btn_notice = tk.Button(
    root,
    text="Notice",
    font=("Arial", 16),
    width=15,
    height=2,
    command=open_notice
)
btn_notice.pack(pady=10)

btn_navigation = tk.Button(
    root,
    text="Navigation",
    font=("Arial", 16),
    width=15,
    height=2,
    command=open_navigation
)
btn_navigation.pack(pady=10)

btn_bookreader = tk.Button(
    root,
    text="Book Reading",
    font=("Arial", 16),
    width=15,
    height=2,
    command=open_bookreader
)
btn_bookreader.pack(pady=10)

# ---------- Speak on Open ----------
def welcome():
    speak("Welcome to your PathMate. Do you want to read notices, want navigation, or book reading?")
    threading.Thread(target=listen_command, daemon=True).start()

root.after(1000, welcome)

root.mainloop()
