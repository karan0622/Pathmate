"""
PathMate — YOLO Object Detection + Live OCR Text Reader
========================================================
Detects: Vehicles, Furniture, Traffic Signs, Appliances,
         Persons, Plants, Books  +  Reads any text in frame

Requirements:
    pip install ultralytics opencv-python pytesseract numpy
    brew install tesseract          (Mac)
    brew install tesseract-lang     (optional extra languages)

Controls:
    Q      →  Quit
    S      →  Toggle speech on/off
    O      →  Toggle OCR mode on/off
    D      →  Toggle object detection on/off
    + / -  →  Confidence threshold up / down
"""

import cv2
import pytesseract
import numpy as np
import threading
import queue
import re
import time
import subprocess
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.4
SPEAK_COOLDOWN       = 3       # seconds before same object is re-announced
OCR_INTERVAL_FRAMES  = 25      # run OCR every N frames
OCR_STABILITY        = 3       # times a line must appear before speaking it
OCR_COOLDOWN         = 8.0     # seconds before same text is re-announced
CAMERA_INDEX         = 0
SHOW_FPS             = True

# ── Mac Tesseract path — uncomment if tesseract not found ──
# pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# ─────────────────────────────────────────────
# CATEGORY DEFINITIONS  (COCO classes)
# ─────────────────────────────────────────────
VEHICLES = [
    "car", "motorcycle", "bus", "truck", "bicycle",
    "boat", "train", "airplane"
]
FURNITURE = [
    "chair", "couch", "bed", "dining table", "toilet",
    "desk", "bookshelf", "cabinet"
]
TRAFFIC_SIGNS         = ["stop sign", "traffic light"]
ELECTRICAL_APPLIANCES = [
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "refrigerator", "hair drier",
    "clock", "fan"
]
PERSONS = ["person"]
PLANTS  = ["potted plant", "vase"]
BOOKS   = ["book"]

CATEGORY_COLORS = {
    "vehicle":      (0, 255, 0),
    "furniture":    (255, 165, 0),
    "traffic_sign": (0, 0, 255),
    "appliance":    (0, 255, 255),
    "person":       (255, 255, 0),
    "plant":        (0, 180, 0),
    "book":         (180, 105, 255),
    "other":        (200, 200, 200),
}

# ─────────────────────────────────────────────
# SPEECH ENGINE  (Mac `say` — no pyttsx3 crash)
# ─────────────────────────────────────────────
_speech_queue = queue.Queue(maxsize=2)
_speech_stop  = threading.Event()

def _speak_mac(text: str):
    try:
        subprocess.run(
            ["say", "-r", "200", "-v", "Samantha", text],
            check=True, timeout=15
        )
    except subprocess.TimeoutExpired:
        print("[TTS] Timeout — skipping")
    except Exception as e:
        print(f"[TTS Error] {e}")

def _speech_worker():
    while not _speech_stop.is_set():
        try:
            text = _speech_queue.get(timeout=0.5)
            if text is None:
                break
            if _speech_queue.empty():
                _speak_mac(text)
            else:
                print(f"[TTS] Skipped stale: {text[:40]}")
            _speech_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[TTS Worker Error] {e}")

threading.Thread(target=_speech_worker, daemon=True).start()

def speak(text: str):
    try:
        _speech_queue.put_nowait(text)
    except queue.Full:
        try:
            _speech_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            _speech_queue.put_nowait(text)
        except queue.Full:
            pass

def stop_speech():
    _speech_stop.set()
    try:
        _speech_queue.put_nowait(None)
    except queue.Full:
        pass

# ─────────────────────────────────────────────
# OBJECT DETECTION HELPERS
# ─────────────────────────────────────────────
_obj_spoken: dict = {}

def get_category(label: str) -> str:
    l = label.lower()
    if l in VEHICLES:              return "vehicle"
    if l in FURNITURE:             return "furniture"
    if l in TRAFFIC_SIGNS:         return "traffic_sign"
    if l in ELECTRICAL_APPLIANCES: return "appliance"
    if l in PERSONS:               return "person"
    if l in PLANTS:                return "plant"
    if l in BOOKS:                 return "book"
    return "other"

def get_color(category: str):
    return CATEGORY_COLORS.get(category, CATEGORY_COLORS["other"])

def obj_should_speak(label: str) -> bool:
    now = time.time()
    if label not in _obj_spoken or (now - _obj_spoken[label]) > SPEAK_COOLDOWN:
        _obj_spoken[label] = now
        return True
    return False

def draw_detection(frame, box, label, confidence, category):
    x1, y1, x2, y2 = map(int, box)
    color = get_color(category)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    txt = f"{label} {confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, txt, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, f"[{category.upper()}]", (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

def draw_legend(frame):
    items = [
        ("Vehicle",      "vehicle"),
        ("Furniture",    "furniture"),
        ("Traffic Sign", "traffic_sign"),
        ("Appliance",    "appliance"),
        ("Person",       "person"),
        ("Plant",        "plant"),
        ("Book",         "book"),
    ]
    x, y = 10, 30
    cv2.rectangle(frame, (5, 5), (185, len(items) * 22 + 15), (30, 30, 30), -1)
    for name, cat in items:
        color = get_color(cat)
        cv2.rectangle(frame, (x, y - 12), (x + 16, y + 2), color, -1)
        cv2.putText(frame, name, (x + 22, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 22

# ─────────────────────────────────────────────
# OCR HELPERS
# ─────────────────────────────────────────────
TESS_CONFIG = r'--oem 3 --psm 6 -l eng'

def ocr_preprocess(frame):
    resized   = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray      = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    denoised  = cv2.fastNlMeansDenoising(gray, h=10)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel    = np.ones((1, 1), np.uint8)
    return cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

def is_real_word(word: str) -> bool:
    if len(word) < 2:
        return False
    if not re.search(r'[a-zA-Z]', word):
        return False
    non_alnum = sum(1 for c in word if not c.isalnum())
    if non_alnum / len(word) > 0.4:
        return False
    letters = re.sub(r'[^a-zA-Z]', '', word)
    if len(letters) >= 3 and sum(1 for c in letters.lower() if c in 'aeiou') == 0:
        return False
    if len(set(word.lower())) <= 2 and len(word) > 3:
        return False
    return True

def clean_line(line: str) -> str:
    return ' '.join(line.encode('ascii', 'ignore').decode().split())

def is_real_line(line: str) -> bool:
    words = line.split()
    if not words:
        return False
    real = [w for w in words if is_real_word(w)]
    if len(real) / len(words) < 0.5:
        return False
    if len(real) == 1 and len(real[0]) < 4:
        return False
    return True

_ocr_lines: list = []
_ocr_lock         = threading.Lock()
_ocr_busy         = False

def run_ocr_thread(frame):
    global _ocr_busy
    try:
        processed = ocr_preprocess(frame)
        data = pytesseract.image_to_data(
            processed, config=TESS_CONFIG,
            output_type=pytesseract.Output.DICT
        )
        lines = {}
        for i in range(len(data["text"])):
            txt  = data["text"][i].strip()
            conf = int(data["conf"][i])
            if txt and conf > 55 and is_real_word(txt):
                lid = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                lines[lid] = lines.get(lid, "") + (" " if lid in lines else "") + txt

        valid = []
        for line in lines.values():
            cleaned = clean_line(line)
            if cleaned and is_real_line(cleaned):
                valid.append(cleaned)

        with _ocr_lock:
            _ocr_lines.clear()
            _ocr_lines.extend(valid)
    except Exception as e:
        print(f"[OCR Error] {e}")
    finally:
        _ocr_busy = False

class SpeechTracker:
    def __init__(self, cooldown=8.0, stability=3):
        self.spoken     = {}
        self.seen_count = {}
        self.cooldown   = cooldown
        self.stability  = stability

    def should_speak(self, lines: list) -> list:
        now      = time.time()
        to_speak = []
        for line in lines:
            self.seen_count[line] = self.seen_count.get(line, 0) + 1
            if self.seen_count[line] < self.stability:
                continue
            if now - self.spoken.get(line, 0) > self.cooldown:
                to_speak.append(line)
                self.spoken[line] = now
        visible = set(lines)
        for l in [k for k in self.seen_count if k not in visible]:
            del self.seen_count[l]
            self.spoken.pop(l, None)
        return to_speak

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  PathMate — Object Detection + OCR Text Reader")
    print("=" * 60)
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")
    print("Model loaded!\n")
    print("Controls:")
    print("  Q      →  Quit")
    print("  S      →  Toggle speech")
    print("  O      →  Toggle OCR")
    print("  D      →  Toggle object detection")
    print("  + / -  →  Confidence up / down")
    print("=" * 60)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    speech_enabled    = True
    detection_enabled = True
    ocr_enabled       = True
    confidence        = CONFIDENCE_THRESHOLD
    frame_count       = 0
    fps_counter       = 0
    fps_start         = time.time()
    fps_display       = 0
    ocr_tracker       = SpeechTracker(cooldown=OCR_COOLDOWN, stability=OCR_STABILITY)

    global _ocr_busy

    speak("PathMate started")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed.")
                break

            frame_count += 1

            # ── Object Detection ──────────────────────────────
            if detection_enabled:
                results = model(frame, conf=confidence, verbose=False)[0]
                for det in results.boxes:
                    label    = model.names[int(det.cls)]
                    conf_val = float(det.conf)
                    box      = det.xyxy[0].tolist()
                    category = get_category(label)
                    if category == "other":
                        continue
                    draw_detection(frame, box, label, conf_val, category)
                    if speech_enabled and obj_should_speak(label):
                        speak(f"{label} detected")

            # ── OCR (every N frames, non-blocking) ────────────
            if ocr_enabled and frame_count % OCR_INTERVAL_FRAMES == 0 and not _ocr_busy:
                _ocr_busy = True
                threading.Thread(
                    target=run_ocr_thread,
                    args=(frame.copy(),),
                    daemon=True
                ).start()

            # ── Draw OCR results ──────────────────────────────
            if ocr_enabled:
                with _ocr_lock:
                    current_lines = list(_ocr_lines)

                txt_x = frame.shape[1] - 420
                txt_y = 30
                if current_lines:
                    panel_h = len(current_lines) * 28 + 10
                    cv2.rectangle(frame,
                                  (txt_x - 5, 5),
                                  (frame.shape[1] - 5, panel_h),
                                  (20, 20, 20), -1)
                    for line in current_lines:
                        cv2.putText(frame, line[:45], (txt_x, txt_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2)
                        txt_y += 28

                if frame_count % OCR_INTERVAL_FRAMES == 0 and speech_enabled and current_lines:
                    to_speak = ocr_tracker.should_speak(current_lines)
                    if to_speak:
                        speak(". ".join(to_speak))

            # ── FPS ───────────────────────────────────────────
            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_start   = time.time()

            if SHOW_FPS:
                cv2.putText(frame, f"FPS: {fps_display}",
                            (frame.shape[1] - 110, frame.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # ── Status bar ────────────────────────────────────
            det_str = "DET:ON"  if detection_enabled else "DET:OFF"
            ocr_str = "OCR:ON"  if ocr_enabled       else "OCR:OFF"
            spk_str = "SPK:ON"  if speech_enabled     else "SPK:OFF"
            ocr_run = " [scanning...]" if _ocr_busy   else ""
            cv2.putText(frame,
                        f"{det_str}  {ocr_str}  {spk_str}  Conf:{confidence:.0%}{ocr_run}",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if detection_enabled:
                draw_legend(frame)

            cv2.imshow("PathMate | Q=Quit  D=Detect  O=OCR  S=Speech", frame)

            # ── Key controls ──────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                speech_enabled = not speech_enabled
                status = "ON" if speech_enabled else "OFF"
                print(f"Speech {status}")
                speak(f"Speech {status}")
            elif key == ord('d'):
                detection_enabled = not detection_enabled
                status = "ON" if detection_enabled else "OFF"
                print(f"Detection {status}")
                speak(f"Detection {status}")
            elif key == ord('o'):
                ocr_enabled = not ocr_enabled
                status = "ON" if ocr_enabled else "OFF"
                print(f"OCR {status}")
                speak(f"Text reading {status}")
            elif key in (ord('+'), ord('=')):
                confidence = min(0.95, confidence + 0.05)
                print(f"Confidence: {confidence:.0%}")
            elif key == ord('-'):
                confidence = max(0.05, confidence - 0.05)
                print(f"Confidence: {confidence:.0%}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        stop_speech()
        print("PathMate stopped.")

if __name__ == "__main__":
    main()