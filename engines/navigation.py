import speech_recognition as sr
from pathfinder import find_path, generate_directions
from campus_map import CAMPUS_MAP
import subprocess
import cv2
import pytesseract
import time

ALIASES = {
    "BLOCK A": "BLOCK-A",
    "BLOCK B": "BLOCK-B-DOOR-1",
    "BLOCK E": "BLOCK-E",
    "ENTRANCE": "ENTRANCE",
    "GROUND": "GROUND",
    "BOYS HOSTEL": "BOYS-HOSTEL",
    "B 202": "B-202",
    "B 201": "B-201",
    "LABS": "BLOCK-B-LABS",
}

def speak(text):
    subprocess.run(['powershell', '-Command', 
                   f'Add-Type -AssemblyName System.Speech; '
                   f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                   f'$s.Speak("{text}")'])

def listen_location(prompt): # prompt is just the question joh hum user se puchna chahte hai like where do you want to go?
    recognizer = sr.Recognizer() # create a recognizer object. think of it like turning on a microphone device - before we can listen to anything we need to set it up first 
    speak(prompt) # speaks the questions out loud to the user before listening
    time.sleep(0.5) # wait for powershell speech to fully finish before opening mic

    with sr.Microphone() as source: # use the microphone and when done, close it automatically
        recognizer.adjust_for_ambient_noise(source, duration=0.5) # listen to the background noise for 0.5 seconds to filter it out
        try: 
            audio = recognizer.listen(source, timeout=7, phrase_time_limit=6) # Actually records what the user says and stores it in audio
            said = recognizer.recognize_google(audio).upper() # Sends the recorded audio to Google and gets back text. Then converts to uppercase.
            said = ALIASES.get(said, said)
            print("User said:", said)
            return said  # Sends the text back to whoever called this function.
        except sr.UnknownValueError:
            speak("Sorry I could not hear you. Please try again.")
            return None
        except sr.RequestError:
            speak("Speech service unavailable")
            return None
        except sr.WaitTimeoutError:
            speak("I did not hear anything. Please try again.")
            return None

# listens for simple short commands like NEXT or SIGN
def listen_command(prompt):
    recognizer = sr.Recognizer()
    speak(prompt)
    time.sleep(0.5) # wait for powershell speech to fully finish before opening mic

    with sr.Microphone() as source: # use the microphone and when done, close it automatically
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=7, phrase_time_limit=4)
            said = recognizer.recognize_google(audio).upper()
            print("Command heard:", said)
            return said
        except:
            return None

# opens camera for 3 seconds and reads any sign using OCR
def read_sign_from_camera():
    speak("Point your camera at the sign. Reading in 3 seconds.")
    time.sleep(3)

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        speak("Camera could not capture image.")
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray).upper().strip()
    print("OCR read:", text)
    return text

# checks if user reached the expected location via voice or camera
def confirm_checkpoint(expected_location):
    speak(f"You should now be at {expected_location}. Say next to continue or say sign to scan a sign.")

    command = listen_command("Listening for your response.")

    if command is None:
        return True  # assume correct and move on

    if "SIGN" in command:
        ocr_text = read_sign_from_camera()
        if ocr_text and expected_location.replace("-", " ") in ocr_text:
            speak(f"Confirmed! You are at {expected_location}.")
            return True
        else:
            speak(f"Sign does not match. You should be at {expected_location}. Please check your location.")
            return False

    if "NEXT" in command:
        return True

    return True  # default move forward

# guides user step by step waiting for confirmation at each checkpoint
def guided_navigation(directions, path):
    for i, (step, next_location) in enumerate(directions[:-1]):
        speak(step)
        confirmed = False

        while not confirmed:
            confirmed = confirm_checkpoint(next_location)
            if not confirmed:
                speak(f"Let's try again. {step}")

    # speak final arrival message
    final_step, final_location = directions[-1]
    speak(final_step)

def start_navigation():
    speak("Welcome to PathMate navigation. Let's get started.")

    while True: # ishliye likha jishe jab tak ushe shi location nhi milti voh aaage na bade
        location = listen_location("Where are you right now? Please say your location.")

        if location is None: # mic couldn't hear, already told user
            continue          # just try again, don't say anything extra
        elif location in CAMPUS_MAP:
            speak(f"Got it. You are at {location}")
            break
        else:
            speak("I could not find that location. Please try again.")

    while True:
        destination = listen_location("Where do you want to go?")
        
        if destination is None:
            continue
        elif destination in CAMPUS_MAP:
            speak(f"Got it. Your destination is {destination}")
            break
        else:
            speak("I could not find that destination. Please try again.")

    # outside both while loops now
    path = find_path(location, destination)
    
    if path:
        directions = generate_directions(path)
        speak(f"Found a route with {len(directions) - 1} steps. Let's begin.")
        guided_navigation(directions, path)  # step by step with checkpoints
    else:
        speak("Sorry, I could not find a route to your destination.")

if __name__ == "__main__":
    start_navigation()