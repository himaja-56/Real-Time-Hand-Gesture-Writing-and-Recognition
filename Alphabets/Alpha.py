import cv2
import mediapipe as mp
import numpy as np
import time
import os
import tensorflow as tf
import random
import pyttsx3
import threading
import queue

# ================= SAFE TEXT TO SPEECH =================
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

speech_queue = queue.Queue()

def speech_worker():
    while True:
        word = speech_queue.get()
        if word is None:
            break
        engine.say(word)
        engine.runAndWait()
        speech_queue.task_done()

speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def speak_word(word):
    # Clear old pending speech (speak latest only)
    while not speech_queue.empty():
        try:
            speech_queue.get_nowait()
        except:
            break
    speech_queue.put(word)

# ================= LOAD ALPHABET MODEL =================
MODEL_PATH = "alphabet_model.h5"

if not os.path.exists(MODEL_PATH):
    print("❌ Model not found")
    exit()

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Alphabet CNN loaded")

# ================= ANIMALS DATASET =================
DATASET_ROOT = "Animals10"  # Adjust path if needed
print("✅ Animals10 dataset connected")

# ================= MediaPipe =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# ================= STATE =================
cap = cv2.VideoCapture(0)

canvas_vis = None
canvas_ai = None

PAUSE_TIME = 1.0
CONF_THRESHOLD = 0.60
TOP_BAR_H = 80

text_sequence = []
last_write_time = None
writing_now = False

prev_x, prev_y = None, None
display_image = None

buttons = ["CLEAR", "BACK", "DETECT", "QUIT"]

# ================= HELPERS =================
def get_fingers(lms):
    tips = [4, 8, 12, 16, 20]
    fingers = []
    fingers.append(lms.landmark[4].x < lms.landmark[3].x)
    for i in range(1, 5):
        fingers.append(lms.landmark[tips[i]].y <
                       lms.landmark[tips[i]-2].y)
    return fingers

def center_for_ai(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    if np.sum(thresh) < 300:
        return None

    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    crop = thresh[y:y+h, x:x+w]

    size = max(w, h) + 40
    square = np.zeros((size, size), dtype=np.uint8)
    square[(size-h)//2:(size-h)//2+h,
           (size-w)//2:(size-w)//2+w] = crop

    resized = cv2.resize(square, (28, 28),
                         interpolation=cv2.INTER_AREA)

    return (resized / 255.0).reshape(1, 28, 28, 1)

def idx_to_char(idx):
    return chr(ord('A') + idx)

# ================= LOAD ANIMAL IMAGE =================
def load_animal_image(word):
    global display_image

    word = word.lower().strip()
    class_folder = os.path.join(DATASET_ROOT, word)

    if not os.path.isdir(class_folder):
        display_image = None
        return False

    images = os.listdir(class_folder)
    if len(images) == 0:
        display_image = None
        return False

    img_name = random.choice(images)
    img_path = os.path.join(class_folder, img_name)

    img = cv2.imread(img_path)
    if img is None:
        display_image = None
        return False

    # Resize proportionally (no distortion)
    h, w, _ = img.shape
    scale = 300 / max(h, w)
    img = cv2.resize(img, (int(w*scale), int(h*scale)))

    display_image = img
    return True

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas_vis is None:
        canvas_vis = np.zeros_like(frame)
        canvas_ai = np.zeros_like(frame)

    # -------- Toolbar --------
    cv2.rectangle(frame, (0, 0), (w, TOP_BAR_H),
                  (30, 30, 30), -1)

    btn_w = w // len(buttons)

    for i, name in enumerate(buttons):
        x1 = i * btn_w
        cv2.putText(frame, name,
                    (x1 + 20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,255), 2)

    # -------- Hand Detection --------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    writing_now = False

    if res.multi_hand_landmarks:
        lms = res.multi_hand_landmarks[0]
        fingers = get_fingers(lms)

        x = int(lms.landmark[8].x * w)
        y = int(lms.landmark[8].y * h)

        # Button Click (two fingers)
        if y < TOP_BAR_H and fingers[1] and fingers[2]:
            idx = x // btn_w
            action = buttons[idx]

            if action == "CLEAR":
                text_sequence.clear()
                canvas_vis[:] = 0
                canvas_ai[:] = 0
                display_image = None

            elif action == "BACK":
                if text_sequence:
                    text_sequence.pop()

            elif action == "DETECT":
                word = "".join(text_sequence).strip()
                if word:
                    success = load_animal_image(word)
                    if success:
                        speak_word(word)

            elif action == "QUIT":
                break

            time.sleep(0.4)

        # Writing Mode
        elif fingers[1] and not fingers[2]:
            writing_now = True

            if prev_x is not None:
                cv2.line(canvas_vis,
                         (prev_x, prev_y),
                         (x, y),
                         (255, 255, 255), 15)

                cv2.line(canvas_ai,
                         (prev_x, prev_y),
                         (x, y),
                         (255, 255, 255), 15)

            prev_x, prev_y = x, y
            last_write_time = time.time()

        else:
            prev_x, prev_y = None, None

        mp_draw.draw_landmarks(frame,
                               lms,
                               mp_hands.HAND_CONNECTIONS)

    # -------- Alphabet Prediction --------
    if last_write_time and not writing_now:
        if time.time() - last_write_time > PAUSE_TIME:

            processed = center_for_ai(canvas_ai)

            if processed is not None:
                pred = model.predict(processed, verbose=0)[0]
                idx = np.argmax(pred)
                confidence = pred[idx]

                if confidence > CONF_THRESHOLD:
                    text_sequence.append(idx_to_char(idx))

            last_write_time = None
            canvas_vis[:] = 0
            canvas_ai[:] = 0

    # -------- Bottom Panel --------
    cv2.rectangle(frame, (0, h-120), (w, h),
                  (20, 20, 20), -1)

    word = "".join(text_sequence)

    cv2.putText(frame,
                word if word else "--",
                (50, h-40),
                cv2.FONT_HERSHEY_DUPLEX,
                2.2,
                (0, 255, 0),
                4)

    # -------- Show Animal Image --------
    if display_image is not None:
        img_h, img_w, _ = display_image.shape
        start_x = w - img_w - 20
        start_y = 100

        if start_y + img_h < h:
            frame[start_y:start_y+img_h, start_x:start_x+img_w] = display_image

    # -------- Overlay Pen Strokes --------
    gray = cv2.cvtColor(canvas_vis, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    frame[mask > 0] = canvas_vis[mask > 0]

    cv2.imshow("Air Writing - Animals Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
speech_queue.put(None)
cap.release()
cv2.destroyAllWindows()