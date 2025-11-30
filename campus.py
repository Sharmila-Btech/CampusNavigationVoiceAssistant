import cv2
import threading
import time
import speech_recognition as sr
from gtts import gTTS
import os
import noisereduce as nr
import numpy as np
import tempfile
import soundfile as sf
import json

# ----------------------------
# Global Configuration
# ----------------------------
current_language = 'ta'
human_detected = True

# ----------------------------
# Speak Function using gTTS + mpg321
# ----------------------------
def speak(text, lang=None):
    lang = lang if lang else current_language
    print(f"Assistant [{lang}]: {text}")
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save("voice.mp3")
    os.system("mpg321 voice.mp3 > /dev/null 2>&1")  # use mpg321 instead of playsound
    os.remove("voice.mp3")

# ----------------------------
# Load Campus Locations from JSON
# ----------------------------
with open("data/campus_locations.json", "r", encoding="utf-8") as f:
    campus_locations = json.load(f)

# ----------------------------
# Voice Command Function (with noise reduction)
# ----------------------------
def takeCommand():
    r = sr.Recognizer()
    r.energy_threshold = 300  # Lower threshold for softer voices
    try:
        with sr.Microphone() as source:
            print("Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=1)
            print("Listening...")
            r.pause_threshold = 1
            audio = r.listen(source, timeout=5, phrase_time_limit=5)

        # Extract audio data
        raw_data = audio.get_raw_data()
        sample_rate = audio.sample_rate
        audio_data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)

        # Normalize
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val * 0.9

        # Boost
        audio_data = np.clip(audio_data * 3.0, -1.0, 1.0)
        audio_data = (audio_data * 32767).astype(np.int16)

        # Noise reduction
        reduced_audio = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=1.0)

        # Save cleaned audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            sf.write(tmp_wav.name, reduced_audio, sample_rate)
            tmp_path = tmp_wav.name

        # Recognize speech
        with sr.AudioFile(tmp_path) as source:
            audio_clean = r.record(source)
            query = r.recognize_google(audio_clean, language='en-IN')
            print(f"User said: {query}")
            return query.lower()

    except Exception as e:
        print("Error:", e)
        speak("Sorry, I didn't understand.")
        return "none"

# ----------------------------
# Greet User
# ----------------------------
def username():
    if current_language == 'en':
        speak("Welcome guest. How can I help you?")
    else:
        speak("Vanakkam guest! Enna help venum?")

# ----------------------------
# Command Handler
# ----------------------------
def handle_commands():
    global current_language
    while True:
        global human_detected
        if not human_detected:
            speak("User illa. Program close panren.")
            break

        query = takeCommand()
        if query == "none":
            continue

        if 'exit' in query or 'quit' in query:
            speak("Goodbye!" if current_language == 'en' else "Poyitu varen!")
            break
        elif 'tamil' in query:
            current_language = 'ta'
            speak("Tamil mode la irukkom")
        elif 'english' in query:
            current_language = 'en'
            speak("English mode activated")
        else:
            found = False
            for location, details in campus_locations.items():
                if location.lower() in query:
                    speak(details[current_language])
                    found = True
                    break
            if not found:
                fallback = {
                    'en': "I can help you find departments, classrooms, or facilities. Please ask clearly.",
                    'ta': "Naan departments, classrooms, illa facilities pathi solluven. Neenga konjam clear ah ketkunga."
                }
                speak(fallback[current_language])

# ----------------------------
# Background Human Monitoring
# ----------------------------
def monitor_human_presence():
    global human_detected
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    last_seen = time.time()

    print("Monitoring user presence...")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        close_face_detected = False
        for (x, y, w, h) in faces:
            if w >= 250 and h >= 250:
                close_face_detected = True
                last_seen = time.time()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                break

        if not close_face_detected and time.time() - last_seen > 3:
            human_detected = False
            break

        cv2.imshow("User Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            human_detected = False
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------
# Initial Face Detection with Box
# ----------------------------
def detect_and_start():
    print("Scanning for human presence with face alignment...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    box_size = 300
    box_x = (frame_width - box_size) // 2
    box_y = (frame_height - box_size) // 2

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        cv2.rectangle(frame, (box_x, box_y), (box_x + box_size, box_y + box_size), (0, 255, 255), 2)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if (
                x > box_x and y > box_y and
                x + w < box_x + box_size and
                y + h < box_y + box_size and
                w >= 250 and h >= 250
            ):
                print("Face aligned correctly inside the box and is near. Starting assistant...")
                cv2.imshow("Face Detection", frame)
                cv2.waitKey(1000)
                cap.release()
                cv2.destroyAllWindows()

                monitor_thread = threading.Thread(target=monitor_human_presence)
                monitor_thread.daemon = True
                monitor_thread.start()

                username()
                handle_commands()
                return

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == '__main__':
    detect_and_start()
