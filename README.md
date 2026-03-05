Readme · MD
Copy

# PathMate 🧭
### A Voice-First Campus Navigation Assistant for Visually Impaired Students

PathMate helps visually impaired students navigate their campus independently using live camera, voice commands, obstacle detection, and turn-by-turn spoken directions.

---

## 💡 How It Works

1. **Point your camera** at any room sign
2. App reads the sign and says *"You are at Block B, Room 202"*
3. **Press the button and say** your destination — *"Take me to the labs"*
4. App speaks step-by-step directions
5. **While walking**, the camera warns you about obstacles — *"Person ahead, Chair on your right"*

---

## 🗂️ Project Structure

```
pathmate/
    main.py                  # App launcher, GUI, voice input
    engines/
        campus_map.py        # Campus layout as a graph (nodes + connections)
        pathfinder.py        # BFS pathfinding algorithm
        navigation.py        # Live camera, YOLO detection, OCR, TTS
        notice-reader.py     # Reads notice board images aloud
```

---

## ⚙️ Features

- 🎙️ **Voice commands** — press a button and speak your destination
- 📷 **Live camera** — real-time room sign reading via OCR
- 🧠 **Smart pathfinding** — BFS algorithm finds shortest route on campus map
- 🔊 **Turn-by-turn directions** — fully spoken, no screen needed
- 🚧 **Obstacle detection** — YOLO detects people, chairs, doors while walking
- 📋 **Notice reader** — reads notice boards aloud

---

## 🛠️ Requirements

```
Python 3.8+
opencv-python
pytesseract
ultralytics
pyttsx3
speechrecognition
Pillow
numpy
```

Install all at once:
```bash
pip install opencv-python pytesseract ultralytics pyttsx3 SpeechRecognition Pillow numpy
```

Also install Tesseract OCR engine:
- **Mac:** `brew install tesseract`
- **Windows:** Download from https://github.com/UB-Mannheim/tesseract/wiki
- **Linux:** `sudo apt install tesseract-ocr`

---

## 🚀 How to Run

```bash
python main.py
```

---

## 🗺️ Updating the Campus Map

All campus locations are defined in `engines/campus_map.py`.

To add a new room or location, just add it to the `CAMPUS_MAP` dictionary:

```python
"B-203": [("BLOCK-B-CORRIDOR-2", 5)],
```

Each entry follows this format:
```python
"LOCATION-NAME": [("CONNECTED-LOCATION", distance_in_steps)]
```

---

## 👨‍💻 Built With

- [YOLOv8](https://github.com/ultralytics/ultralytics) — Object detection
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — Text reading
- [pyttsx3](https://pyttsx3.readthedocs.io/) — Text to speech
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) — Voice input
- [OpenCV](https://opencv.org/) — Camera and image processing
- [Tkinter](https://docs.python.org/3/library/tkinter.html) — GUI

---

## 🎯 Project Status

- [x] Voice launcher (main.py)
- [x] Notice reader (OCR on static image)
- [x] Live camera with YOLO + OCR
- [x] Campus map definition
- [ ] BFS Pathfinding
- [ ] Turn-by-turn direction generator
- [ ] Unified GUI with voice-first navigation
- [ ] Auto-detect current location from OCR → auto-fill navigation

---

## 🤝 Contributing

This project is built for accessibility. If you'd like to contribute or adapt it for your own campus, just update `campus_map.py` with your building layout.

---

## 📄 License

MIT License
