"""
PathMate — YOLO Object Detector (Cross-Platform)
==================================================
Cross-platform TTS support:
  • macOS   → built-in `say` command (Samantha voice)
  • Windows → pyttsx3 (SAPI5)  |  pip install pyttsx3
  • Linux   → espeak (if installed)  OR  pyttsx3 fallback

All other features from v2:
  • YOLOv8s model (small) — better accuracy than nano
  • Object counting per category shown on screen
  • Distance estimation (near / medium / far) by box size
  • Proximity alert — speaks "WARNING: person very close" etc.
  • Smoother detections — tracks objects across frames (no flicker)
  • Danger priority — persons/vehicles/traffic signs spoken first
  • Non-max suppression tuned (less duplicate boxes)
  • Cleaner bounding box UI with rounded corners + drop shadow

Requirements:
    pip install ultralytics opencv-python pyttsx3

Controls:
    Q      →  Quit
    S      →  Toggle speech on/off
    D      →  Toggle detection on/off
    C      →  Toggle object count overlay
    P      →  Toggle proximity alerts
    + / -  →  Confidence threshold up / down
    M      →  Cycle model  (nano → small → medium)
"""

import cv2
import threading
import queue
import time
import subprocess
import platform
import sys
from collections import defaultdict
from ultralytics import YOLO

# ─────────────────────────────────────────────
# PLATFORM DETECTION
# ─────────────────────────────────────────────
OS = platform.system()   # "Darwin", "Windows", "Linux"

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.45
SPEAK_COOLDOWN       = 4       # seconds before same object is re-announced
PROXIMITY_COOLDOWN   = 3       # seconds between proximity warnings
CAMERA_INDEX         = 0
SHOW_FPS             = True
DEFAULT_MODEL        = "yolov8s.pt"
MODELS               = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
MODEL_LABELS         = ["Nano", "Small", "Medium"]

# Box area thresholds (fraction of frame area) for distance estimation
NEAR_THRESHOLD   = 0.15   # box > 15% of frame → NEAR
MEDIUM_THRESHOLD = 0.05   # box > 5%  of frame → MEDIUM
# below 5% → FAR

# ─────────────────────────────────────────────
# CATEGORY DEFINITIONS
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

SPEAK_PRIORITY = {
    "person":       5,
    "vehicle":      4,
    "traffic_sign": 3,
    "appliance":    2,
    "furniture":    1,
    "plant":        0,
    "book":         0,
}

CATEGORY_COLORS = {
    "vehicle":      (0, 220, 0),
    "furniture":    (255, 165, 0),
    "traffic_sign": (0, 60, 255),
    "appliance":    (0, 220, 220),
    "person":       (0, 200, 255),
    "plant":        (0, 160, 0),
    "book":         (180, 105, 255),
    "other":        (200, 200, 200),
}

# ─────────────────────────────────────────────
# CROSS-PLATFORM SPEECH ENGINE
# ─────────────────────────────────────────────

# Try to import pyttsx3 for Windows/Linux fallback
_pyttsx3_engine = None
_pyttsx3_lock   = threading.Lock()

def _init_pyttsx3():
    global _pyttsx3_engine
    try:
        import pyttsx3
        _pyttsx3_engine = pyttsx3.init()
        _pyttsx3_engine.setProperty("rate", 210)
        print("[TTS] pyttsx3 initialised successfully.")
    except Exception as e:
        print(f"[TTS] pyttsx3 init failed: {e}")
        _pyttsx3_engine = None


def _speak_macos(text: str):
    """Use macOS built-in `say` command."""
    try:
        subprocess.run(
            ["say", "-r", "210", "-v", "Samantha", text],
            check=True, timeout=15
        )
    except subprocess.TimeoutExpired:
        print("[TTS] Timeout — skipping")
    except Exception as e:
        print(f"[TTS Error] {e}")


def _speak_linux(text: str):
    """Try espeak on Linux; fall back to pyttsx3."""
    try:
        subprocess.run(
            ["espeak", "-s", "210", text],
            check=True, timeout=15
        )
    except FileNotFoundError:
        # espeak not installed — use pyttsx3
        _speak_pyttsx3(text)
    except subprocess.TimeoutExpired:
        print("[TTS] Timeout — skipping")
    except Exception as e:
        print(f"[TTS Error] {e}")


def _speak_pyttsx3(text: str):
    """Windows (and Linux fallback) via pyttsx3."""
    global _pyttsx3_engine
    if _pyttsx3_engine is None:
        print("[TTS] pyttsx3 not available — skipping speech.")
        return
    try:
        with _pyttsx3_lock:
            _pyttsx3_engine.say(text)
            _pyttsx3_engine.runAndWait()
    except Exception as e:
        print(f"[TTS Error] {e}")


def _speak(text: str):
    """Dispatch to the correct TTS backend for the current OS."""
    if OS == "Darwin":
        _speak_macos(text)
    elif OS == "Windows":
        _speak_pyttsx3(text)
    else:   # Linux / other
        _speak_linux(text)


# ── Speech worker thread ──────────────────────
_speech_queue = queue.Queue(maxsize=3)
_speech_stop  = threading.Event()


def _speech_worker():
    while not _speech_stop.is_set():
        try:
            text = _speech_queue.get(timeout=0.5)
            if text is None:
                break
            if _speech_queue.empty():
                _speak(text)
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
# CATEGORY + DISTANCE HELPERS
# ─────────────────────────────────────────────
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


def get_distance(box, frame_h, frame_w):
    x1, y1, x2, y2 = box
    box_area   = (x2 - x1) * (y2 - y1)
    frame_area = frame_h * frame_w
    ratio      = box_area / frame_area
    if ratio >= NEAR_THRESHOLD:
        return "NEAR", (0, 0, 255)
    elif ratio >= MEDIUM_THRESHOLD:
        return "MED",  (0, 165, 255)
    else:
        return "FAR",  (0, 220, 0)

# ─────────────────────────────────────────────
# OBJECT TRACKER
# ─────────────────────────────────────────────
class ObjectTracker:
    def __init__(self, cooldown=4.0, confirm_frames=2):
        self.spoken      = {}
        self.seen_frames = defaultdict(int)
        self.confirm     = confirm_frames
        self.cooldown    = cooldown
        self.prox_spoken = {}

    def update(self, detected_labels: set):
        for lbl in detected_labels:
            self.seen_frames[lbl] += 1
        gone = [l for l in self.seen_frames if l not in detected_labels]
        for l in gone:
            self.seen_frames[l] = 0

    def should_speak(self, label: str) -> bool:
        if self.seen_frames[label] < self.confirm:
            return False
        now = time.time()
        if now - self.spoken.get(label, 0) > self.cooldown:
            self.spoken[label] = now
            return True
        return False

    def should_warn_proximity(self, label: str) -> bool:
        now = time.time()
        if now - self.prox_spoken.get(label, 0) > PROXIMITY_COOLDOWN:
            self.prox_spoken[label] = now
            return True
        return False


tracker = ObjectTracker(cooldown=SPEAK_COOLDOWN, confirm_frames=2)

# ─────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, thickness=2, r=10):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90,  0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0,   0, 90, color, thickness)


def draw_detection(frame, box, label, confidence, category, dist_label, dist_color):
    x1, y1, x2, y2 = map(int, box)
    color = get_color(category)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1 + 3, y1 + 3), (x2 + 3, y2 + 3), (0, 0, 0), 2)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    draw_rounded_rect(frame, (x1, y1), (x2, y2), color, thickness=2)

    txt = f"{label}  {confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, txt, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    (dw, dh), _ = cv2.getTextSize(dist_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(frame, (x2 - dw - 10, y2 - dh - 8), (x2, y2), dist_color, -1)
    cv2.putText(frame, dist_label, (x2 - dw - 5, y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


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
    cv2.rectangle(frame, (5, 5), (190, len(items) * 22 + 15), (20, 20, 20), -1)
    for name, cat in items:
        color = get_color(cat)
        cv2.rectangle(frame, (x, y - 12), (x + 16, y + 2), color, -1)
        cv2.putText(frame, name, (x + 22, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
        y += 22


def draw_counts(frame, counts: dict):
    active = {k: v for k, v in counts.items() if v > 0}
    if not active:
        return
    x = frame.shape[1] - 180
    y = 30
    cv2.rectangle(frame, (x - 5, 5), (frame.shape[1] - 5, len(active) * 22 + 15),
                  (20, 20, 20), -1)
    for cat, cnt in sorted(active.items(), key=lambda i: -i[1]):
        color = get_color(cat)
        cv2.putText(frame, f"{cat[:10]}: {cnt}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        y += 22

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  PathMate — Cross-Platform YOLO Object Detector")
    print(f"  Platform: {OS}")
    print("=" * 60)

    # Initialise pyttsx3 for Windows (and Linux fallback)
    if OS in ("Windows", "Linux"):
        _init_pyttsx3()

    model_idx = MODELS.index(DEFAULT_MODEL)
    print(f"Loading model: {MODELS[model_idx]} ({MODEL_LABELS[model_idx]})...")
    model = YOLO(MODELS[model_idx])
    print("Model loaded!\n")
    print("Controls:")
    print("  Q      →  Quit")
    print("  S      →  Toggle speech")
    print("  D      →  Toggle detection")
    print("  C      →  Toggle count overlay")
    print("  P      →  Toggle proximity alerts")
    print("  M      →  Cycle model (Nano/Small/Medium)")
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
    counts_enabled    = True
    proximity_enabled = True
    confidence        = CONFIDENCE_THRESHOLD
    fps_counter       = 0
    fps_start         = time.time()
    fps_display       = 0
    reload_model      = False

    speak("Object detection started")

    try:
        while True:
            # ── Reload model if user switched ─────────────────
            if reload_model:
                print(f"Loading {MODELS[model_idx]} ({MODEL_LABELS[model_idx]})...")
                speak(f"Switching to {MODEL_LABELS[model_idx]} model")
                model = YOLO(MODELS[model_idx])
                reload_model = False
                print("Model switched.")

            ret, frame = cap.read()
            if not ret:
                print("Camera read failed.")
                break

            fh, fw = frame.shape[:2]
            counts = defaultdict(int)
            detected_labels = set()

            # ── Run YOLO ──────────────────────────────────────
            if detection_enabled:
                results = model(
                    frame,
                    conf=confidence,
                    iou=0.45,
                    verbose=False
                )[0]

                detections = []
                for det in results.boxes:
                    lbl = model.names[int(det.cls)]
                    cat = get_category(lbl)
                    if cat == "other":
                        continue
                    detections.append((det, lbl, cat))

                detections.sort(
                    key=lambda x: SPEAK_PRIORITY.get(x[2], 0), reverse=True
                )

                for det, label, category in detections:
                    conf_val = float(det.conf)
                    box      = det.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, box)

                    dist_label, dist_color = get_distance(
                        (x1, y1, x2, y2), fh, fw
                    )

                    draw_detection(frame, box, label, conf_val,
                                   category, dist_label, dist_color)

                    counts[category] += 1
                    detected_labels.add(label)

                tracker.update(detected_labels)

                for det, label, category in detections:
                    box    = det.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, box)
                    dist_label, _ = get_distance((x1, y1, x2, y2), fh, fw)

                    if speech_enabled and tracker.should_speak(label):
                        speak(f"{label} detected")

                    if (proximity_enabled and speech_enabled
                            and dist_label == "NEAR"
                            and category in ("person", "vehicle", "traffic_sign")):
                        if tracker.should_warn_proximity(label):
                            speak(f"Warning! {label} very close")

            # ── Overlays ──────────────────────────────────────
            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_start   = time.time()

            if SHOW_FPS:
                cv2.putText(frame, f"FPS: {fps_display}",
                            (fw - 115, fh - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if detection_enabled:
                draw_legend(frame)
                if counts_enabled:
                    draw_counts(frame, counts)

            flags = []
            flags.append("DET:ON"  if detection_enabled else "DET:OFF")
            flags.append("CNT:ON"  if counts_enabled    else "CNT:OFF")
            flags.append("PROX:ON" if proximity_enabled else "PROX:OFF")
            flags.append("SPK:ON"  if speech_enabled    else "SPK:OFF")
            flags.append(f"Conf:{confidence:.0%}")
            flags.append(f"Model:{MODEL_LABELS[model_idx]}")
            flags.append(f"OS:{OS}")
            cv2.putText(frame, "  ".join(flags),
                        (10, fh - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

            cv2.imshow(
                "PathMate YOLO | Q=Quit  S=Speech  D=Det  C=Count  P=Prox  M=Model",
                frame
            )

            # ── Keys ──────────────────────────────────────────
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
            elif key == ord('c'):
                counts_enabled = not counts_enabled
                print(f"Count overlay {'ON' if counts_enabled else 'OFF'}")
            elif key == ord('p'):
                proximity_enabled = not proximity_enabled
                status = "ON" if proximity_enabled else "OFF"
                print(f"Proximity alerts {status}")
                speak(f"Proximity alerts {status}")
            elif key == ord('m'):
                model_idx    = (model_idx + 1) % len(MODELS)
                reload_model = True
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
        print("Detection stopped.")


if __name__ == "__main__":
    main()
