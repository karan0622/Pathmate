"""
Campus Navigator — Live Camera + Voice GPS Navigation
======================================================
Live camera feed with real-time turn-by-turn voice navigation.
No YOLO, no object detection — pure navigation.

HOW TO USE:
    Press N  →  Speaks all destinations, listens for your voice reply
    Press X  →  Stop navigation
    Press R  →  Repeat current direction
    Press Q  →  Quit

GPS MODES (set GPS_MODE below):
    "iphone"    →  iPhone hotspot + GPS2IP app  ← RECOMMENDED
    "mac"       →  macOS CoreLocation (needs Xcode tools)
    "sim"       →  Simulated walk for testing
    "nmea_tcp"  →  Generic NMEA over TCP
    "serial"    →  USB/Bluetooth GPS dongle
    "gpsd"      →  Linux gpsd daemon

iPhone setup (2 min):
    1. Install "GPS2IP Lite" (free) on iPhone from App Store
    2. Open GPS2IP → tap START
    3. Turn on iPhone Personal Hotspot
    4. Connect Mac WiFi to iPhone hotspot
    5. Run this script — connects automatically to 172.20.10.1:11123

Requirements:
    pip install opencv-python pyttsx3 SpeechRecognition pyaudio
"""

import math
import cv2
import threading
import queue
import time
import subprocess
import platform
import socket
import re
import tempfile
import os

# ─────────────────────────────────────────────────────────────────────────────
# GPS CONFIGURATION  — change GPS_MODE to match your setup
# ─────────────────────────────────────────────────────────────────────────────
GPS_MODE      = "iphone"       # "iphone" | "mac" | "sim" | "nmea_tcp" | "serial" | "gpsd"

IPHONE_HOST   = "172.20.10.1"  # always this when Mac is on iPhone hotspot
IPHONE_PORT   = 11123          # GPS2IP default port

NMEA_TCP_HOST = "192.168.1.100"
NMEA_TCP_PORT = 4352
SERIAL_PORT   = "/dev/ttyUSB0"
SERIAL_BAUD   = 9600
GPSD_HOST     = "127.0.0.1"
GPSD_PORT     = 2947

CAMERA_INDEX  = 0

# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION TUNING
# ─────────────────────────────────────────────────────────────────────────────
DIRECTION_COOLDOWN  = 8    # seconds between repeated voice directions
WAYPOINT_RADIUS_M   = 15   # metres — auto-advance to next waypoint
ARRIVAL_RADIUS_M    = 12   # metres — declare destination reached

# ─────────────────────────────────────────────────────────────────────────────
# CAMPUS LOCATIONS  (from your KML)
# ─────────────────────────────────────────────────────────────────────────────
LOCATIONS = {
    "Main Gate":       {"lat": 28.4633456, "lng": 77.4899615},
    "Gate 2":          {"lat": 28.4638313, "lng": 77.4907468},
    "Block A":         {"lat": 28.463182,  "lng": 77.4901102},
    "Block B":         {"lat": 28.4628895, "lng": 77.4903468},
    "Block D":         {"lat": 28.4628149, "lng": 77.491378 },
    "Block E Library": {"lat": 28.4625064, "lng": 77.490678 },
    "Canteen":         {"lat": 28.4617474, "lng": 77.4905871},
    "Boys Hostel":     {"lat": 28.4624505, "lng": 77.4908483},
    "Girls Hostel":    {"lat": 28.4629764, "lng": 77.4915966},
    "Ground":          {"lat": 28.4629669, "lng": 77.4903199},
    "Auditorium":      {"lat": 28.4626779, "lng": 77.4905291},
    "ATM":             {"lat": 28.4630812, "lng": 77.4896413},
    "Stationery":      {"lat": 28.4630316, "lng": 77.489526 },
    "Admin Office":    {"lat": 28.4628406, "lng": 77.4899551},
    "Bio Garden":      {"lat": 28.4623502, "lng": 77.4916228},
}

ALIASES = {
    "main gate":"Main Gate",       "gate":"Main Gate",
    "entrance":"Main Gate",        "gate 2":"Gate 2",
    "second gate":"Gate 2",        "back gate":"Gate 2",
    "block a":"Block A",           "a block":"Block A",
    "block b":"Block B",           "b block":"Block B",
    "block d":"Block D",           "d block":"Block D",
    "block e":"Block E Library",   "library":"Block E Library",
    "lib":"Block E Library",       "block e library":"Block E Library",
    "canteen":"Canteen",           "mess":"Canteen",
    "food":"Canteen",              "cafeteria":"Canteen",
    "boys hostel":"Boys Hostel",   "boys":"Boys Hostel",
    "gents":"Boys Hostel",         "girls hostel":"Girls Hostel",
    "girls":"Girls Hostel",        "ladies":"Girls Hostel",
    "ground":"Ground",             "sports":"Ground",
    "field":"Ground",              "playground":"Ground",
    "auditorium":"Auditorium",     "audi":"Auditorium",
    "hall":"Auditorium",           "atm":"ATM",
    "cash":"ATM",                  "bank":"ATM",
    "stationery":"Stationery",     "shop":"Stationery",
    "books":"Stationery",          "admin":"Admin Office",
    "office":"Admin Office",       "administration":"Admin Office",
    "bio garden":"Bio Garden",     "garden":"Bio Garden",
    "bio":"Bio Garden",
}

CAMPUS_EDGES = [
    ("Main Gate","Block A"),       ("Main Gate","ATM"),
    ("Main Gate","Gate 2"),        ("Main Gate","Admin Office"),
    ("Gate 2","Block A"),          ("Gate 2","Ground"),
    ("Block A","Block B"),         ("Block A","Ground"),
    ("Block A","Admin Office"),    ("Block B","Ground"),
    ("Block B","Auditorium"),      ("Block B","Block E Library"),
    ("Block B","Admin Office"),    ("Auditorium","Block E Library"),
    ("Auditorium","Ground"),       ("Block E Library","Boys Hostel"),
    ("Boys Hostel","Block D"),     ("Boys Hostel","Canteen"),
    ("Block D","Girls Hostel"),    ("Girls Hostel","Bio Garden"),
    ("Canteen","Bio Garden"),      ("Canteen","Block E Library"),
    ("ATM","Stationery"),          ("Stationery","Admin Office"),
]

OS = platform.system()

# ─────────────────────────────────────────────────────────────────────────────
# SPEECH ENGINE
# speak()          → async queue (used during navigation)
# _speak_blocking() → direct blocking call (used during destination menu only)
# _nav_selecting   → True while menu is open; silences async queue
# ─────────────────────────────────────────────────────────────────────────────
_pyttsx3_engine = None
_pyttsx3_lock   = threading.Lock()
_nav_selecting  = False

def _init_pyttsx3():
    global _pyttsx3_engine
    try:
        import pyttsx3
        _pyttsx3_engine = pyttsx3.init()
        _pyttsx3_engine.setProperty("rate", 210)
        print("[TTS] pyttsx3 ready.")
    except Exception as e:
        print(f"[TTS] pyttsx3 failed: {e}")

def _kill_tts():
    if OS == "Darwin":
        subprocess.run(["killall","say"],    capture_output=True)
    elif OS == "Linux":
        subprocess.run(["killall","espeak"], capture_output=True)

def _speak_blocking(text: str, rate: int = 210):
    global _pyttsx3_engine
    if OS == "Darwin":
        try:
            subprocess.run(["say","-r",str(rate),"-v","Samantha",text],
                           check=True, timeout=60)
        except Exception as e:
            print(f"[TTS] {e}")
    elif OS == "Windows":
        if _pyttsx3_engine is None: return
        try:
            with _pyttsx3_lock:
                _pyttsx3_engine.setProperty("rate", rate)
                _pyttsx3_engine.say(text)
                _pyttsx3_engine.runAndWait()
                _pyttsx3_engine.setProperty("rate", 210)
        except Exception as e:
            print(f"[TTS] {e}")
    else:  # Linux
        try:
            subprocess.run(["espeak","-s",str(rate),text], check=True, timeout=60)
        except FileNotFoundError:
            if _pyttsx3_engine:
                try:
                    with _pyttsx3_lock:
                        _pyttsx3_engine.setProperty("rate", rate)
                        _pyttsx3_engine.say(text)
                        _pyttsx3_engine.runAndWait()
                        _pyttsx3_engine.setProperty("rate", 210)
                except Exception as e:
                    print(f"[TTS] {e}")
        except Exception as e:
            print(f"[TTS] {e}")

_speech_queue = queue.Queue(maxsize=3)
_speech_stop  = threading.Event()

def _speech_worker():
    while not _speech_stop.is_set():
        try:
            text = _speech_queue.get(timeout=0.5)
            if text is None: break
            if _speech_queue.empty():
                _speak_blocking(text)
            else:
                print(f"[TTS] dropped stale: {text[:50]}")
            _speech_queue.task_done()
        except queue.Empty: continue
        except Exception as e: print(f"[TTS worker] {e}")

threading.Thread(target=_speech_worker, daemon=True).start()

def speak(text: str):
    if _nav_selecting: return
    try:
        _speech_queue.put_nowait(text)
    except queue.Full:
        try:    _speech_queue.get_nowait()
        except: pass
        try:    _speech_queue.put_nowait(text)
        except: pass

def _drain_and_kill():
    while not _speech_queue.empty():
        try: _speech_queue.get_nowait()
        except: break
    _kill_tts()
    time.sleep(0.2)

def stop_speech():
    _speech_stop.set()
    try: _speech_queue.put_nowait(None)
    except: pass

# ─────────────────────────────────────────────────────────────────────────────
# MICROPHONE INPUT
# ─────────────────────────────────────────────────────────────────────────────
def _listen_mic(timeout: int = 7, phrase_limit: int = 5) -> str:
    try:
        import speech_recognition as sr
    except ImportError:
        print("[MIC] pip install SpeechRecognition pyaudio")
        _speak_blocking("Voice input library not installed.", 210)
        return ""
    r = sr.Recognizer()
    r.pause_threshold  = 0.8
    r.energy_threshold = 300
    r.dynamic_energy_threshold = True
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.4)
            print("[MIC] Listening...")
            audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
        text = r.recognize_google(audio, language="en-IN")
        print(f"[MIC] Heard: '{text}'")
        return text.lower().strip()
    except Exception as e:
        print(f"[MIC] {type(e).__name__}: {e}")
        return ""

# ─────────────────────────────────────────────────────────────────────────────
# GEO MATH
# ─────────────────────────────────────────────────────────────────────────────
def haversine(lat1, lng1, lat2, lng2):
    R = 6371000
    dLat = math.radians(lat2 - lat1)
    dLng = math.radians(lng2 - lng1)
    a = (math.sin(dLat/2)**2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dLng/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def bearing(lat1, lng1, lat2, lng2):
    dLng = math.radians(lng2 - lng1)
    y = math.sin(dLng) * math.cos(math.radians(lat2))
    x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
         - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dLng))
    return (math.degrees(math.atan2(y, x)) + 360) % 360

def relative_direction(heading: float, target_bearing: float, dtarget: int, tname: str):
    """
    Build a fully natural spoken instruction like a real guide would say.

    heading        = direction user is currently walking  (degrees 0-360)
    target_bearing = compass bearing to the next waypoint (degrees 0-360)
    dtarget        = distance to next waypoint in metres
    tname          = name of next waypoint

    Returns (spoken_sentence, display_label, arrow_char)
    """
    import random
    diff = (target_bearing - heading + 360) % 360

    # ── Distance phrase ────────────────────────────────────────────────────────
    if   dtarget < 5:   dist = "right there"
    elif dtarget < 15:  dist = "just a few steps away"
    elif dtarget < 30:  dist = f"about {dtarget} steps ahead"
    elif dtarget < 60:  dist = f"around {dtarget} metres"
    elif dtarget < 120: dist = f"about {dtarget} metres"
    else:               dist = f"roughly {dtarget} metres"

    # ── Very close — no direction needed ──────────────────────────────────────
    if dtarget < 8:
        return f"{tname} is {dist}, you are almost there.", "Almost there", "↑"

    if dtarget < 20:
        return f"Keep going, {tname} is {dist}.", "Keep going", "↑"

    # ── Direction + natural sentence ──────────────────────────────────────────
    if diff < 15 or diff >= 345:
        label = "Straight ahead"
        arrow = "↑"
        phrases = [
            f"Keep going straight, {tname} is {dist}.",
            f"Go straight ahead for {dist} to {tname}.",
            f"Continue straight, {tname} is {dist} in front of you.",
        ]

    elif 15 <= diff < 45:
        label = "Slight right"
        arrow = "↗"
        phrases = [
            f"Bear slightly to your right, then walk {dist} to {tname}.",
            f"Drift a little to the right, {tname} is {dist}.",
            f"Keep slightly right, {tname} is {dist} ahead.",
        ]

    elif 45 <= diff < 100:
        label = "Turn right"
        arrow = "→"
        phrases = [
            f"Turn right and walk {dist} to {tname}.",
            f"Take a right here, then {dist} to {tname}.",
            f"Turn right now, {tname} is {dist} that way.",
        ]

    elif 100 <= diff < 135:
        label = "Sharp right"
        arrow = "↘"
        phrases = [
            f"Take a sharp right, {tname} is {dist} ahead.",
            f"Turn hard right and continue {dist} to {tname}.",
        ]

    elif 135 <= diff < 180:
        label = "Turn back right"
        arrow = "↓"
        phrases = [
            f"You have gone too far. Turn around to your right, {tname} is {dist} back.",
            f"Turn back to the right, {tname} is {dist} behind you.",
        ]

    elif 180 <= diff < 225:
        label = "Turn back left"
        arrow = "↓"
        phrases = [
            f"You have gone too far. Turn around to your left, {tname} is {dist} back.",
            f"Turn back to the left, {tname} is {dist} behind you.",
        ]

    elif 225 <= diff < 260:
        label = "Sharp left"
        arrow = "↙"
        phrases = [
            f"Take a sharp left, {tname} is {dist} ahead.",
            f"Turn hard left and continue {dist} to {tname}.",
        ]

    elif 260 <= diff < 315:
        label = "Turn left"
        arrow = "←"
        phrases = [
            f"Turn left and walk {dist} to {tname}.",
            f"Take a left here, then {dist} to {tname}.",
            f"Turn left now, {tname} is {dist} that way.",
        ]

    else:  # 315–345
        label = "Slight left"
        arrow = "↖"
        phrases = [
            f"Bear slightly to your left, then walk {dist} to {tname}.",
            f"Drift a little to the left, {tname} is {dist}.",
            f"Keep slightly left, {tname} is {dist} ahead.",
        ]

    spoken = random.choice(phrases)
    return spoken, label, arrow

# ─────────────────────────────────────────────────────────────────────────────
# CAMPUS GRAPH + DIJKSTRA
# ─────────────────────────────────────────────────────────────────────────────
def _build_graph():
    g = {n: {} for n in LOCATIONS}
    for a, b in CAMPUS_EDGES:
        d = round(haversine(LOCATIONS[a]["lat"], LOCATIONS[a]["lng"],
                            LOCATIONS[b]["lat"], LOCATIONS[b]["lng"]))
        g[a][b] = d
        g[b][a] = d
    return g

GRAPH = _build_graph()

def dijkstra(start, end):
    dist = {n: float("inf") for n in LOCATIONS}
    prev = {}
    dist[start] = 0
    unvisited = set(LOCATIONS)
    while unvisited:
        u = min(unvisited, key=lambda n: dist[n])
        if u == end or dist[u] == float("inf"): break
        unvisited.remove(u)
        for nb, w in GRAPH.get(u, {}).items():
            alt = dist[u] + w
            if alt < dist[nb]:
                dist[nb] = alt
                prev[nb] = u
    path, cur = [], end
    while cur:
        path.insert(0, cur)
        cur = prev.get(cur)
    if not path or path[0] != start: return None
    return {"path": path, "distance": dist[end]}

def nearest_node(lat, lng):
    return min(LOCATIONS,
               key=lambda n: haversine(lat, lng,
                                       LOCATIONS[n]["lat"], LOCATIONS[n]["lng"]))

def resolve_destination(text: str):
    t = text.lower().strip()
    if t in ALIASES: return ALIASES[t]
    for alias, name in ALIASES.items():
        if alias in t: return name
    for name in LOCATIONS:
        if name.lower() in t: return name
    # spoken number words → digits
    words = {"one":"1","two":"2","three":"3","four":"4","five":"5",
             "six":"6","seven":"7","eight":"8","nine":"9","ten":"10",
             "eleven":"11","twelve":"12","thirteen":"13","fourteen":"14","fifteen":"15"}
    for w, d in words.items():
        t = t.replace(w, d)
    m = re.search(r'\b(\d+)\b', t)
    if m:
        idx = int(m.group(1)) - 1
        names = list(LOCATIONS.keys())
        if 0 <= idx < len(names): return names[idx]
    return None

# ─────────────────────────────────────────────────────────────────────────────
# GPS READER
# ─────────────────────────────────────────────────────────────────────────────
class GPSReader:
    def __init__(self):
        self.lat = self.lng = self.accuracy = None
        self.valid = False
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def get(self):
        with self._lock:
            return self.lat, self.lng, self.accuracy, self.valid

    def stop(self): self._stop.set()

    # ── NMEA parser ───────────────────────────────────────────────────────────
    def _parse_nmea(self, sentence):
        try:
            s = sentence.strip()
            if not s.startswith("$"): return
            p = s.split(",")
            if p[0] in ("$GPRMC","$GNRMC"):
                if len(p) < 7 or p[2] != "A": return
                lat = self._n2d(p[3], p[4])
                lng = self._n2d(p[5], p[6])
                if lat is not None:
                    with self._lock:
                        self.lat, self.lng, self.valid = lat, lng, True
            elif p[0] in ("$GPGGA","$GNGGA"):
                if len(p) < 10 or p[6] == "0": return
                lat = self._n2d(p[2], p[3])
                lng = self._n2d(p[4], p[5])
                hdop = float(p[8]) if p[8] else 5.0
                if lat is not None:
                    with self._lock:
                        self.lat, self.lng = lat, lng
                        self.accuracy = round(hdop * 3, 1)
                        self.valid = True
        except: pass

    @staticmethod
    def _n2d(raw, direction):
        try:
            dot = raw.index(".")
            v = float(raw[:dot-2]) + float(raw[dot-2:]) / 60.0
            return -v if direction in ("S","W") else v
        except: return None

    def _run(self):
        if   GPS_MODE == "iphone":   self._iphone()
        elif GPS_MODE == "mac":      self._mac()
        elif GPS_MODE == "sim":      self._sim()
        elif GPS_MODE == "gpsd":     self._gpsd()
        elif GPS_MODE == "serial":   self._serial()
        elif GPS_MODE == "nmea_tcp": self._tcp(NMEA_TCP_HOST, NMEA_TCP_PORT)

    # ── iPhone hotspot ────────────────────────────────────────────────────────
    def _iphone(self):
        print(f"[GPS] Connecting to iPhone GPS at {IPHONE_HOST}:{IPHONE_PORT}")
        print(f"[GPS] Ensure: iPhone hotspot ON, GPS2IP app running, Mac on iPhone WiFi")
        delay = 3
        while not self._stop.is_set():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(8)
                    s.connect((IPHONE_HOST, IPHONE_PORT))
                    s.settimeout(5)
                    print("[GPS] iPhone GPS connected!")
                    delay = 3
                    buf = ""
                    while not self._stop.is_set():
                        try:
                            chunk = s.recv(1024).decode("ascii", errors="ignore")
                            if not chunk: break
                            buf += chunk
                            while "\n" in buf:
                                line, buf = buf.split("\n", 1)
                                self._parse_nmea(line)
                        except socket.timeout: continue
                        except: break
            except Exception as e:
                print(f"[GPS] iPhone: {e} — retry in {delay}s")
            if not self._stop.is_set():
                time.sleep(delay)
                delay = min(delay * 2, 30)

    # ── macOS CoreLocation ────────────────────────────────────────────────────
    def _mac(self):
        SWIFT = r"""
import CoreLocation
import Foundation
class D: NSObject, CLLocationManagerDelegate {
    let m = CLLocationManager()
    override init() {
        super.init(); m.delegate = self
        m.desiredAccuracy = kCLLocationAccuracyBest
        m.requestWhenInUseAuthorization(); m.startUpdatingLocation()
    }
    func locationManager(_ m: CLLocationManager, didUpdateLocations l: [CLLocation]) {
        let c = l.last!
        print("\(c.coordinate.latitude),\(c.coordinate.longitude),\(c.horizontalAccuracy)")
        fflush(stdout)
    }
    func locationManager(_ m: CLLocationManager, didFailWithError e: Error) {
        print("ERR:\(e.localizedDescription)"); fflush(stdout)
    }
}
let d = D(); RunLoop.main.run()
"""
        tmp = tempfile.NamedTemporaryFile(suffix=".swift", delete=False, mode="w")
        tmp.write(SWIFT); tmp.close()
        print("[GPS] Starting CoreLocation via Swift — allow location if prompted.")
        proc = None
        try:
            proc = subprocess.Popen(["swift", tmp.name],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, bufsize=1)
            for line in proc.stdout:
                if self._stop.is_set(): break
                line = line.strip()
                if not line or line.startswith("ERR"): continue
                parts = line.split(",")
                if len(parts) >= 3:
                    try:
                        lat, lng, acc = float(parts[0]), float(parts[1]), float(parts[2])
                        with self._lock:
                            self.lat, self.lng, self.accuracy, self.valid = lat, lng, round(acc,1), True
                    except: pass
        except FileNotFoundError:
            print("[GPS] Swift not found. Run: xcode-select --install")
            print("[GPS] Falling back to simulation.")
            self._sim()
        except Exception as e:
            print(f"[GPS] Mac: {e}")
        finally:
            if proc:
                try: proc.terminate()
                except: pass
            try: os.unlink(tmp.name)
            except: pass

    # ── Simulation ────────────────────────────────────────────────────────────
    def _sim(self):
        pts = [(28.4633456,77.4899615),(28.463182,77.4901102),(28.4628895,77.4903468),
               (28.4626779,77.4905291),(28.4625064,77.490678),(28.4624505,77.4908483),
               (28.4617474,77.4905871)]
        idx = 0; lat, lng = pts[0]
        print("[GPS] Simulation mode.")
        while not self._stop.is_set():
            tlat, tlng = pts[idx]
            dlat, dlng = tlat - lat, tlng - lng
            d = math.sqrt(dlat**2 + dlng**2)
            if d < 0.00002: idx = (idx + 1) % len(pts)
            else:
                s = min(0.000008, d); lat += dlat/d*s; lng += dlng/d*s
            with self._lock:
                self.lat, self.lng, self.accuracy, self.valid = lat, lng, 3.0, True
            time.sleep(1)

    # ── gpsd ──────────────────────────────────────────────────────────────────
    def _gpsd(self):
        try:
            import gpsd
            gpsd.connect(host=GPSD_HOST, port=GPSD_PORT)
            print("[GPS] gpsd connected.")
            while not self._stop.is_set():
                try:
                    pkt = gpsd.get_current()
                    if pkt.mode >= 2:
                        with self._lock:
                            self.lat, self.lng = pkt.lat, pkt.lon
                            self.accuracy = getattr(pkt,"error",{}).get("x",5.0)
                            self.valid = True
                except: pass
                time.sleep(1)
        except Exception as e: print(f"[GPS] gpsd: {e}")

    # ── Serial ────────────────────────────────────────────────────────────────
    def _serial(self):
        try:
            import serial
            with serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1) as ser:
                print(f"[GPS] Serial {SERIAL_PORT}")
                while not self._stop.is_set():
                    self._parse_nmea(ser.readline().decode("ascii", errors="ignore"))
        except Exception as e: print(f"[GPS] serial: {e}")

    # ── Generic TCP ───────────────────────────────────────────────────────────
    def _tcp(self, host, port):
        print(f"[GPS] NMEA TCP {host}:{port}")
        while not self._stop.is_set():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(5); s.connect((host, port))
                    buf = ""
                    while not self._stop.is_set():
                        buf += s.recv(1024).decode("ascii", errors="ignore")
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            self._parse_nmea(line)
            except Exception as e:
                print(f"[GPS] TCP: {e}"); time.sleep(3)

# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATOR
# ─────────────────────────────────────────────────────────────────────────────
class Navigator:
    def __init__(self, gps: GPSReader):
        self.gps   = gps
        self.active = False
        self.destination = None
        self.route_path  = []
        self.route_coords = []
        self.step_idx    = 0
        self.total_dist  = 0
        self.last_dir_time = 0
        self.announced_arrival = False
        self._lock = threading.Lock()
        # Heading tracking — last two GPS positions to work out which way user faces
        self._prev_lat  = None
        self._prev_lng  = None
        self._heading   = None   # degrees 0-360, updated as user moves
        self._last_instruction = ""  # avoid repeating identical instruction

    def start(self, dest_name: str):
        lat, lng, _, valid = self.gps.get()
        if not valid:
            speak("GPS not ready. Please wait.")
            return False
        result = dijkstra(nearest_node(lat, lng), dest_name)
        if not result:
            speak(f"Cannot find route to {dest_name}.")
            return False
        with self._lock:
            self.destination   = dest_name
            self.route_path    = result["path"]
            self.route_coords  = [(LOCATIONS[n]["lat"], LOCATIONS[n]["lng"])
                                  for n in result["path"]]
            self.step_idx      = 1
            self.total_dist    = result["distance"]
            self.active        = True
            self.announced_arrival = False
            self.last_dir_time = 0
        mins = max(1, round(self.total_dist / 80))
        via  = " then ".join(result["path"][1:-1]) if len(result["path"]) > 2 else "direct"
        speak(f"Navigating to {dest_name}. "
              f"{self.total_dist} metres, about {mins} minutes. Via {via}.")
        return True

    def stop(self):
        with self._lock:
            self.active = False
            self.destination = None
        speak("Navigation stopped.")

    def _update_heading(self, lat, lng):
        """
        Compute walking heading from last two GPS positions.
        Only updates if we moved more than 3m (avoids GPS jitter noise).
        Returns current heading in degrees, or None if not enough movement yet.
        """
        if self._prev_lat is not None:
            moved = haversine(self._prev_lat, self._prev_lng, lat, lng)
            if moved > 3.0:   # at least 3m movement to trust the heading
                self._heading = bearing(self._prev_lat, self._prev_lng, lat, lng)
                self._prev_lat, self._prev_lng = lat, lng
        else:
            self._prev_lat, self._prev_lng = lat, lng
        return self._heading

    def update(self):
        """Returns OSD dict or None. Also fires voice directions."""
        with self._lock:
            if not self.active: return None
            lat, lng, _, valid = self.gps.get()
            if not valid or lat is None:
                return {"line1": "Waiting for GPS...", "line2": "", "arrow": "?", "arrived": False}

            # Update our heading estimate from movement
            heading = self._update_heading(lat, lng)

            dest = self.destination
            dl   = LOCATIONS[dest]
            dist_dest = haversine(lat, lng, dl["lat"], dl["lng"])

            # Arrived at final destination
            if dist_dest < ARRIVAL_RADIUS_M and not self.announced_arrival:
                self.announced_arrival = True
                self.active = False
                import random
                arrival_phrases = [
                    f"You have arrived at {dest}! You made it.",
                    f"Here you are! This is {dest}.",
                    f"Destination reached. Welcome to {dest}.",
                ]
                speak(random.choice(arrival_phrases))
                return {"line1": f"ARRIVED at {dest}!", "line2": "", "arrow": "✓", "arrived": True}

            # Advance past waypoints already walked through
            while self.step_idx < len(self.route_path):
                wp_lat, wp_lng = self.route_coords[self.step_idx]
                if haversine(lat, lng, wp_lat, wp_lng) < WAYPOINT_RADIUS_M:
                    wname = self.route_path[self.step_idx]
                    import random
                    passing_phrases = [
                        f"You are now passing {wname}. Keep going.",
                        f"Good, {wname} is right here. Continue ahead.",
                        f"You have reached {wname}. Keep walking.",
                    ]
                    speak(random.choice(passing_phrases))
                    self.step_idx += 1
                    self.last_dir_time = 0
                    self._last_instruction = ""   # force re-announce after waypoint
                else:
                    break

            if self.step_idx >= len(self.route_path):
                return {"line1": f"Approaching {dest}", "line2": f"{round(dist_dest)}m away",
                        "arrow": "↑", "arrived": False}

            tname      = self.route_path[self.step_idx]
            tlat, tlng = self.route_coords[self.step_idx]
            dtarget    = round(haversine(lat, lng, tlat, tlng))
            abs_bearing = bearing(lat, lng, tlat, tlng)

            # ── Build instruction ──────────────────────────────────────────────
            if heading is not None:
                # Know which way user faces → full natural relative instruction
                spoken, instr, arrow = relative_direction(heading, abs_bearing, dtarget, tname)
            else:
                # Not enough movement yet to know heading — ask user to walk a few steps
                arrow  = "?"
                instr  = "Walk a few steps..."
                spoken = f"Walk a few steps forward so I can figure out which way you are facing, then I will guide you to {tname}."

            # ── Periodic voice announcement ───────────────────────────────────
            now = time.time()
            instruction_changed = (spoken != self._last_instruction)
            time_elapsed = (now - self.last_dir_time) > DIRECTION_COOLDOWN

            if time_elapsed or instruction_changed:
                self.last_dir_time = now
                self._last_instruction = spoken
                speak(spoken)

            step_num    = self.step_idx
            total_steps = len(self.route_path) - 1
            heading_str = f"heading {round(heading)}°" if heading is not None else "heading unknown"
            return {
                "arrow":  arrow,
                "line1":  f"{arrow}  {instr}  →  {tname}",
                "line2":  f"{dtarget}m away  |  step {step_num}/{total_steps}  |  {heading_str}",
                "arrived": False,
            }

    def repeat_direction(self):
        with self._lock:
            if not self.active:
                speak("No active navigation.")
                return
            self.last_dir_time = 0

# ─────────────────────────────────────────────────────────────────────────────
# DESTINATION SELECTION  — fully voice driven, zero overlap
# ─────────────────────────────────────────────────────────────────────────────
def ask_destination(navigator: Navigator):
    global _nav_selecting
    if _nav_selecting: return

    def _run():
        global _nav_selecting
        _nav_selecting = True
        _drain_and_kill()

        names = list(LOCATIONS.keys())

        # One fast sentence — all destinations
        sentence = ("Choose your destination. "
                    + ", ".join(f"{i} {n}" for i, n in enumerate(names, 1))
                    + ". Say the number or name now.")

        print("\n" + "="*50)
        for i, n in enumerate(names, 1):
            print(f"  {i:2}. {n}")
        print("="*50)

        _speak_blocking(sentence, rate=300)

        if not _nav_selecting:
            _nav_selecting = False
            return

        _speak_blocking("Listening.", rate=220)
        heard = _listen_mic(timeout=7, phrase_limit=5)
        _nav_selecting = False

        if not heard:
            _speak_blocking("Did not catch that. Press N to try again.", 210)
            return

        dest = resolve_destination(heard)
        if not dest:
            _speak_blocking(f"Could not find {heard}. Press N to try again.", 210)
            return

        print(f"[NAV] '{heard}' → {dest}")
        navigator.start(dest)

    threading.Thread(target=_run, daemon=True, name="NavSelect").start()

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA OVERLAY DRAWING
# ─────────────────────────────────────────────────────────────────────────────
# Colours
C_BG      = (10, 10, 10)
C_BORDER  = (0, 212, 255)
C_TEXT    = (0, 255, 200)
C_MUTED   = (100, 100, 100)
C_GREEN   = (0, 255, 100)
C_ORANGE  = (0, 165, 255)
C_RED     = (0, 60, 255)

def _text_size(text, scale, thickness=2):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return w, h

def draw_nav_banner(frame, osd, gps_valid, gps_acc):
    fh, fw = frame.shape[:2]

    # ── GPS status pill (top right) ───────────────────────────────────────────
    gps_color = C_GREEN if gps_valid else C_RED
    gps_label = ("GPS OK" if gps_valid else "GPS WAIT") + (f" ±{gps_acc}m" if gps_acc else "")
    tw, th = _text_size(gps_label, 0.5, 1)
    cv2.rectangle(frame, (fw - tw - 22, 6), (fw - 6, 28), C_BG, -1)
    cv2.rectangle(frame, (fw - tw - 22, 6), (fw - 6, 28), gps_color, 1)
    cv2.circle(frame, (fw - tw - 12, 17), 5, gps_color, -1)
    cv2.putText(frame, gps_label, (fw - tw - 4, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, gps_color, 1)

    # ── Listening banner ──────────────────────────────────────────────────────
    if _nav_selecting:
        cv2.rectangle(frame, (0, 40), (fw, 120), C_BG, -1)
        cv2.rectangle(frame, (0, 40), (fw, 120), C_BORDER, 2)
        lines = ["🎤  LISTENING FOR DESTINATION", "Say a number or place name"]
        scales = [0.8, 0.55]
        colors = [C_BORDER, C_MUTED]
        y = 72
        for line, sc, col in zip(lines, scales, colors):
            tw, _ = _text_size(line, sc)
            cv2.putText(frame, line, ((fw - tw)//2, y),
                        cv2.FONT_HERSHEY_SIMPLEX, sc, col, 2)
            y += 32
        return

    # ── Navigation OSD ────────────────────────────────────────────────────────
    if osd:
        arrived = osd.get("arrived", False)
        border_col = C_GREEN if arrived else C_BORDER

        # Arrow box (left side)
        arrow = osd.get("arrow", "?")
        cv2.rectangle(frame, (0, 40), (100, 130), C_BG, -1)
        cv2.rectangle(frame, (0, 40), (100, 130), border_col, 2)
        aw, _ = _text_size(arrow, 2.5, 3)
        cv2.putText(frame, arrow, ((100 - aw)//2, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, border_col, 3)

        # Text box (rest of width)
        cv2.rectangle(frame, (102, 40), (fw, 130), C_BG, -1)
        cv2.rectangle(frame, (102, 40), (fw, 130), border_col, 2)
        line1 = osd.get("line1", "")
        line2 = osd.get("line2", "")
        cv2.putText(frame, line1, (112, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, C_TEXT, 2)
        cv2.putText(frame, line2, (112, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_MUTED, 1)
    else:
        # Idle hint
        cv2.rectangle(frame, (0, 40), (fw, 80), C_BG, -1)
        msg = "Press  N  to navigate by voice"
        tw, _ = _text_size(msg, 0.65)
        cv2.putText(frame, msg, ((fw - tw)//2, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_MUTED, 1)

    # ── Bottom status strip ───────────────────────────────────────────────────
    cv2.rectangle(frame, (0, fh - 26), (fw, fh), C_BG, -1)
    status = f"GPS:{GPS_MODE.upper()}  |  N=Navigate  X=Stop  R=Repeat  Q=Quit"
    cv2.putText(frame, status, (8, fh - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_MUTED, 1)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Campus Navigator — Voice GPS + Live Camera")
    print(f"  Platform: {OS}  |  GPS Mode: {GPS_MODE}")
    print("=" * 60)
    print("  pip install opencv-python pyttsx3")
    print("  pip install SpeechRecognition pyaudio")
    print("=" * 60)
    print("  N = navigate (voice)   X = stop   R = repeat   Q = quit")
    print("=" * 60)

    if OS in ("Windows", "Linux"):
        _init_pyttsx3()

    gps       = GPSReader()
    navigator = Navigator(gps)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    # FPS tracking
    fps_counter = 0
    fps_start   = time.time()
    fps_display = 0

    speak("Campus Navigator ready. Press N to navigate.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera error.")
                break

            fh, fw = frame.shape[:2]

            # FPS
            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_start   = time.time()

            # Navigation update + draw overlay
            osd = None if _nav_selecting else navigator.update()
            lat, lng, acc, valid = gps.get()
            draw_nav_banner(frame, osd, valid, acc)

            # FPS counter (top left)
            cv2.putText(frame, f"FPS {fps_display}", (8, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_MUTED, 1)

            cv2.imshow("Campus Navigator  |  N=Navigate  X=Stop  R=Repeat  Q=Quit", frame)

            key = cv2.waitKey(1) & 0xFF
            if   key == ord('q'): break
            elif key == ord('n'): ask_destination(navigator)
            elif key == ord('x'): navigator.stop()
            elif key == ord('r'): navigator.repeat_direction()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        gps.stop()
        stop_speech()
        print("Navigator stopped.")


if __name__ == "__main__":
    main()