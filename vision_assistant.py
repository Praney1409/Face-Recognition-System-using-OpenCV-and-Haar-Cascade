import speech_recognition as sr
import pyttsx3
import os
import time
import re

# ---------------- TTS SETUP ----------------
engine = pyttsx3.init()
engine.setProperty("rate", 180)
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[1].id)  # female voice

def speak(text):
    print(f"\nVISION 🔊: {text}")
    engine.say(text)
    engine.runAndWait()

# ---------------- VOICE RECOGNIZER ----------------
recognizer = sr.Recognizer()

# Wake word variations
wake_words = [
    "hey vision", "hi vision", "hello vision",
    "hey vijon", "hey vishan", "hey visan", "hey visun",
    "hey vijan", "hey wishon", "hey wishan",
    "vision", "vijan", "visan", "vishon",
    "hey vision system", "ok vision", "hey virson"
]

# Command variants
recognize_cmds = [
    "recognize", "rekognize", "face recognition", "start recognition",
    "detect face", "detect faces", "open face detection",
    "start camera", "recognition start", "recognize face", "chehra pehchano",
    "mukh pehchano", "identify faces", "recognize person", "pehchaan karo"
]

add_person_cmds = [
    "add person", "new person", "add new person", "capture faces",
    "capture new face", "add someone", "add face", "save new person",
    "naya aadmi jodo", "naya chehra add karo", "add user", "register person"
]

train_cmds = [
    "train", "train model", "start training", "retrain", "model train",
    "train data", "model training", "train the model", "training start",
    "model sikhana", "model banana", "model update"
]

view_dataset_cmds = [
    "view dataset", "open dataset", "show dataset", "see dataset",
    "dataset dikhao", "show data", "open data", "view data",
    "dataset display", "display data", "show saved persons",
    "open people list", "show faces", "saved faces", "open records"
]

exit_cmds = [
    "stop", "exit", "close", "terminate", "bye", "shutdown",
    "band karo", "niklo", "quit", "stop vision"
]

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def listen(timeout=5):
    """Listen and return recognized text"""
    with sr.Microphone() as source:
        print("🎧 Listening...")
        audio = recognizer.listen(source, phrase_time_limit=timeout)
    try:
        command = recognizer.recognize_google(audio, language="en-IN").lower()
        print(f"🗣️ You said: {command}")
        return normalize_text(command)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        speak("Network error. Try again later.")
        return ""

def wait_for_wake():
    """Wait until wake word detected"""
    while True:
        cmd = listen()
        for w in wake_words:
            if w in cmd:
                return True

def process_command(command):
    if not command:
        speak("Sorry, I didn't catch that.")
        return

    # Recognize Faces
    if any(word in command for word in recognize_cmds):
        speak("Starting face recognition.")
        os.system("streamlit run face_recognition_app.py")

    # Add Person
    elif any(word in command for word in add_person_cmds):
        speak("Opening new person capture window.")
        os.system("streamlit run face_recognition_app.py")

    # Train Model
    elif any(word in command for word in train_cmds):
        speak("Training the model now.")
        os.system("streamlit run face_recognition_app.py")

    # View Dataset
    elif any(word in command for word in view_dataset_cmds):
        speak("Showing dataset information.")
        os.system("streamlit run face_recognition_app.py")

    # Exit
    elif any(word in command for word in exit_cmds):
        speak("Goodbye! Vision going to sleep.")
        return "exit"

    else:
        speak("Sorry, I didn’t understand that command.")

def main():
    speak("Vision is online. Say 'Hey Vision' to wake me up.")
    while True:
        print("\n🎧 Waiting for wake word...")
        if wait_for_wake():
            speak("Yes, I'm here. What should I do?")
            command = listen(timeout=6)
            result = process_command(command)
            if result == "exit":
                break
            speak("Ready for your next command. Say 'Hey Vision' anytime.")
            time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        speak("Vision shutting down. Goodbye!")
