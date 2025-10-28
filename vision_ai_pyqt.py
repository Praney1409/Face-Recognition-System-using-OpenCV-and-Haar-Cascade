"""
Vision AI - PyQt6 desktop app
Features:
- Camera preview
- Wake-word ("hey vision") listening loop (background)
- Accepts voice commands after wake word:
    add face, train model, start recognition, what's the time, hello, stop, exit
- Add face capture, training (LBPH), recognition
- Low-light enhancement via LAB equalization
"""

import sys
import os
import cv2
import time
import threading
import queue
import numpy as np
import re
from datetime import datetime
from difflib import SequenceMatcher

# PyQt6 imports
from PyQt6 import QtCore, QtGui, QtWidgets

# voice
import pyttsx3
import speech_recognition as sr

# Paths
DATASET_DIR = "dataset"
TRAINER_FILE = "trainer.yml"
LABELS_FILE = "labels.npy"
os.makedirs(DATASET_DIR, exist_ok=True)

# Face detection / recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# require opencv-contrib-python
recognizer = cv2.face.LBPHFaceRecognizer_create()

### Utility: robust fuzzy match for wake and commands
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def matches_any(phrase, variants, threshold=0.6):
    phrase = phrase.lower()
    for v in variants:
        if similar(phrase, v.lower()) >= threshold or v.lower() in phrase:
            return True
    return False

### Image enhancement for low-light
def enhance_image(frame):
    try:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced
    except Exception:
        return frame

### TTS helper (non-blocking)
class TTS:
    def __init__(self):
        self.engine = pyttsx3.init()
        # choose a voice if multiple are available
        voices = self.engine.getProperty("voices")
        if len(voices) > 0:
            try:
                self.engine.setProperty("voice", voices[0].id)
            except Exception:
                pass
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            text = self.q.get()
            if text is None:
                break
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception:
                pass

    def speak(self, text):
        self.q.put(text)

    def stop(self):
        self.q.put(None)

tts = TTS()

### Audio recognizer wrapper (blocking recognize)
def recognize_from_mic(timeout=5, phrase_time_limit=5):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        # adjust for ambient noise briefly
        r.adjust_for_ambient_noise(source, duration=0.8)
        try:
            audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except Exception:
            return ""
    try:
        # try english india first, fallback to hindi
        text = r.recognize_google(audio, language="en-IN")
        return text.lower()
    except sr.UnknownValueError:
        try:
            text = r.recognize_google(audio, language="hi-IN")
            return text.lower()
        except Exception:
            return ""
    except Exception:
        return ""

### Face operations
def capture_faces_for(name, max_count=30, min_size=(60, 60)):
    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    count = 0
    start = time.time()
    while count < max_count and (time.time() - start) < 60:
        ret, frame = cam.read()
        if not ret:
            break
        frame = enhance_image(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size)
        for (x, y, w, h) in faces:
            face_img = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            count += 1
            cv2.imwrite(os.path.join(person_dir, f"{name}_{count}.jpg"), face_img)
        # small sleep to avoid hammering
        time.sleep(0.1)
    cam.release()
    return count

def train_model():
    faces = []
    labels = []
    label_map = {}
    idx = 0
    for person in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_dir):
            continue
        for f in os.listdir(person_dir):
            path = os.path.join(person_dir, f)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (200, 200))
            faces.append(img)
            labels.append(idx)
        label_map[idx] = person
        idx += 1
    if len(faces) == 0:
        return False, "No faces to train."
    recognizer.train(faces, np.array(labels))
    recognizer.save(TRAINER_FILE)
    np.save(LABELS_FILE, label_map)
    return True, f"Trained on {len(faces)} images, {len(label_map)} persons."

def load_labels():
    if os.path.exists(LABELS_FILE):
        return np.load(LABELS_FILE, allow_pickle=True).item()
    return {}

def recognize_once(timeout_seconds=30, min_size=(60,60), confidence_thresh=80, on_detect=None):
    # simple single-run recognizer that calls on_detect(name, confidence, frame)
    if not os.path.exists(TRAINER_FILE):
        return "no_model"
    recognizer.read(TRAINER_FILE)
    labels = load_labels()
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    start = time.time()
    while time.time() - start < timeout_seconds:
        ret, frame = cam.read()
        if not ret:
            break
        frame = enhance_image(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size)
        for (x, y, w, h) in faces:
            face_img = cv2.resize(gray[y:y+h, x:x+w], (200,200))
            try:
                label, conf = recognizer.predict(face_img)
            except Exception:
                label, conf = None, 999
            name = labels.get(label, "Unknown") if label is not None else "Unknown"
            if on_detect:
                on_detect(name, conf, frame.copy())
            # return after first detection (so UI can update)
            cam.release()
            return name, conf
        time.sleep(0.02)
    cam.release()
    return None, None

### PyQt6 GUI classes
class VideoThread(QtCore.QThread):
    change_pixmap = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._run_flag = False

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        self._run_flag = True
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                cv_img = enhance_image(cv_img)
                self.change_pixmap.emit(cv_img)
            else:
                time.sleep(0.05)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class VisionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision AI — Desktop Assistant")
        self.setGeometry(120, 80, 1000, 700)

        # central widget
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout()
        central.setLayout(layout)

        # top: camera preview
        self.camera_label = QtWidgets.QLabel()
        self.camera_label.setFixedSize(640, 480)
        self.camera_label.setStyleSheet("background-color: #222;")
        layout.addWidget(self.camera_label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # info label
        self.info_label = QtWidgets.QLabel("Status: Initializing...")
        layout.addWidget(self.info_label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # buttons row
        btn_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(btn_layout)

        self.btn_listen = QtWidgets.QPushButton("Start Assistant (Listen)")
        self.btn_listen.clicked.connect(self.toggle_listening)
        btn_layout.addWidget(self.btn_listen)

        self.btn_add = QtWidgets.QPushButton("Add Face (Voice)")
        self.btn_add.clicked.connect(self.add_face_prompt)
        btn_layout.addWidget(self.btn_add)

        self.btn_train = QtWidgets.QPushButton("Train Model")
        self.btn_train.clicked.connect(self.train_action)
        btn_layout.addWidget(self.btn_train)

        self.btn_recognize = QtWidgets.QPushButton("Start Recognition")
        self.btn_recognize.clicked.connect(self.start_recognition)
        btn_layout.addWidget(self.btn_recognize)

        self.btn_stop = QtWidgets.QPushButton("Stop All")
        self.btn_stop.clicked.connect(self.stop_all)
        btn_layout.addWidget(self.btn_stop)

        # bottom: logs
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(120)
        layout.addWidget(self.log_text)

        # video thread
        self.vthread = VideoThread()
        self.vthread.change_pixmap.connect(self.update_image)
        self.vthread.start()

        # assistant thread flag
        self.assistant_running = False
        self.assistant_thread = None

        # recognition thread
        self.recognition_thread = None

        # speak intro
        tts.speak("Vision activated. Hello. Say Hey Vision to wake me up.")

        self.wake_variants = [
            "hey vision", "hi vision", "hello vision", "vision", "ok vision",
            "hey vijon", "hey vishan", "hey visan"
        ]
        self.command_variants = {
            "add": ["add face", "register", "add person", "capture face", "naya chehra", "naye chehre"],
            "train": ["train", "train model", "start training", "retrain"],
            "recognize": ["recognize", "start recognition", "detect faces", "identify"],
            "time": ["time", "what time", "tell time", "whats the time"],
            "hello": ["hello", "hi", "hey"],
            "stop": ["stop", "exit", "quit", "shutdown"]
        }

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")
        self.info_label.setText(f"Status: {msg}")

    def closeEvent(self, event):
        # stop threads
        self.stop_all()
        self.vthread.stop()
        tts.stop()
        super().closeEvent(event)

    ### GUI image update
    @QtCore.pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Convert cv image (BGR) to Qt pixmap and show."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qt_img).scaled(self.camera_label.width(), self.camera_label.height(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.camera_label.setPixmap(pix)

    ### Assistant control
    def toggle_listening(self):
        if not self.assistant_running:
            self.start_assistant()
        else:
            self.stop_assistant()

    def start_assistant(self):
        self.assistant_running = True
        self.btn_listen.setText("Stop Assistant (Listening)")
        self.assistant_thread = threading.Thread(target=self.assistant_loop, daemon=True)
        self.assistant_thread.start()
        self.log("Assistant started and listening for wake word.")

    def stop_assistant(self):
        self.assistant_running = False
        self.btn_listen.setText("Start Assistant (Listen)")
        self.log("Assistant stopped listening.")

    def assistant_loop(self):
        """Long-running background loop: listens for wake word, then a command."""
        # we speak once that it's active
        tts.speak("Voice assistant active. Say Hey Vision to wake me.")
        while self.assistant_running:
            # Listen for a short phrase (wake)
            text = recognize_from_mic(timeout=6, phrase_time_limit=4)
            if not text:
                continue
            txt = re.sub(r'[^a-z\s]', '', text.lower())
            self.log(f"Heard: {text}")
            # check wake word via fuzzy match
            if matches_any(txt, self.wake_variants, threshold=0.55):
                tts.speak("Yes, I am listening. What do you want me to do?")
                self.log("Wake word detected. Awaiting command...")
                # listen for command
                cmd_text = recognize_from_mic(timeout=6, phrase_time_limit=6)
                if not cmd_text:
                    tts.speak("Sorry, I didn't catch that.")
                    continue
                self.log(f"Command: {cmd_text}")
                cmd_norm = re.sub(r'[^a-z\s]', '', cmd_text.lower())
                # map to actions with fuzzy matching
                if matches_any(cmd_norm, self.command_variants["add"], threshold=0.5):
                    tts.speak("Please say the name for registration.")
                    self.log("Command -> Add face")
                    name_heard = recognize_from_mic(timeout=6, phrase_time_limit=4)
                    if name_heard:
                        name_norm = re.sub(r'[^a-z\s]', '', name_heard).strip().replace(" ", "_")
                        tts.speak(f"Registering {name_norm}. Please face the camera.")
                        self.log(f"Capturing faces for {name_norm} ...")
                        count = capture_faces_for(name_norm)
                        self.log(f"Captured {count} images for {name_norm}")
                        tts.speak(f"Captured {count} images for {name_norm}. You should train the model now.")
                    else:
                        tts.speak("Name not heard. Cancelled.")
                elif matches_any(cmd_norm, self.command_variants["train"], threshold=0.5):
                    self.log("Command -> Train model")
                    tts.speak("Training model now. This may take a little while.")
                    ok, msg = train_model()
                    if ok:
                        self.log(msg)
                        tts.speak("Training completed.")
                    else:
                        self.log(msg)
                        tts.speak("Training failed: no data found.")
                elif matches_any(cmd_norm, self.command_variants["recognize"], threshold=0.5):
                    self.log("Command -> Recognize faces")
                    tts.speak("Starting recognition for 30 seconds.")
                    # run recognition for a while and report detections
                    def on_detect(name, conf, frame):
                        self.log(f"Detected: {name} (conf {int(conf)})")
                        tts.speak(f"I see {name}")
                    res_name, res_conf = recognize_once(timeout_seconds=30, on_detect=on_detect)
                    if res_name is None:
                        tts.speak("No faces recognized.")
                    elif res_name == "no_model":
                        tts.speak("No trained model found. Please train first.")
                elif matches_any(cmd_norm, self.command_variants["time"], threshold=0.5):
                    now = datetime.now().strftime("%I:%M %p")
                    tts.speak(f"The time is {now}")
                elif matches_any(cmd_norm, self.command_variants["hello"], threshold=0.5):
                    tts.speak("Hello, I am Vision. How can I help?")
                elif matches_any(cmd_norm, self.command_variants["stop"], threshold=0.5):
                    tts.speak("Okay. Stopping assistant.")
                    self.stop_assistant()
                    break
                else:
                    tts.speak("Sorry, I didn't understand that command.")
            # small delay to avoid CPU hog
            time.sleep(0.2)

    ### GUI Button Actions
    def add_face_prompt(self):
        # prompt a simple dialog to enter name
        name, ok = QtWidgets.QInputDialog.getText(self, "Add Face", "Enter name:")
        if ok and name.strip():
            name_norm = re.sub(r'[^a-zA-Z0-9_ ]', '', name).strip().replace(" ", "_")
            tts.speak(f"Capturing faces for {name_norm}. Please look at the camera.")
            self.log(f"Capturing faces for {name_norm} ...")
            count = capture_faces_for(name_norm)
            self.log(f"Captured {count} images for {name_norm}")
            tts.speak(f"Captured {count} images for {name_norm}. You can train the model now.")

    def train_action(self):
        tts.speak("Training model now.")
        self.log("Training model...")
        ok, msg = train_model()
        if ok:
            self.log(msg)
            tts.speak("Training completed successfully.")
        else:
            self.log(msg)
            tts.speak("Training failed or no data found.")

    def start_recognition(self):
        # a UI-triggered recognize (reports first detection)
        self.log("Starting recognition (UI)...")
        tts.speak("Starting recognition for 30 seconds.")
        def on_detect(name, conf, frame):
            self.log(f"Detected: {name} (conf {int(conf)})")
            tts.speak(f"I see {name}")
        res_name, res_conf = recognize_once(timeout_seconds=30, on_detect=on_detect)
        if res_name is None:
            self.log("No face detected in time.")
            tts.speak("No faces recognized.")
        elif res_name == "no_model":
            self.log("No trained model found.")
            tts.speak("No trained model found. Please train first.")

    def stop_all(self):
        self.stop_assistant()
        self.log("Stopping recognition & assistant.")
        tts.speak("Stopping all activities.")
        # no persistent recognition thread to kill here (recognize_once runs blocking)

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = VisionApp()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
