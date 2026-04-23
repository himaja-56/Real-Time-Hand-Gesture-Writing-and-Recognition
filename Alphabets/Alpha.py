import cv2
import mediapipe as mp
import numpy as np
import time
import os
import tensorflow as tf
import requests
import threading
import difflib
from dotenv import load_dotenv

# ================= LOAD ENV =================
load_dotenv()

# ================= LOAD MODEL =================
MODEL_PATH = "alphabet_model.h5"

if not os.path.exists(MODEL_PATH):
    print("❌ Model not found")
    exit()

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ CNN Model Loaded")

# ================= IMAGE GENERATION =================
generated_image = None
is_generating = False

def generate_image(prompt):
    global generated_image, is_generating

    if is_generating:
        return

    is_generating = True
    print("🚀 Generating image for:", prompt)

    try:
        url = f"https://image.pollinations.ai/prompt/{prompt}"
        response = requests.get(url, timeout=20)

        image_bytes = response.content
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is not None:
            generated_image = cv2.resize(img, (256, 256))
            print("✅ Image generated")

    except Exception as e:
        print("❌ Error:", e)

    is_generating = False


# ================= AUTOCORRECT =================
def autocorrect_text(text):
    dictionary = ["cat", "dog", "car", "tree", "house", "person", "phone", "book", "sofa"]

    words = text.lower().split()
    corrected = []

    for w in words:
        match = difflib.get_close_matches(w, dictionary, n=1)
        corrected.append(match[0] if match else w)

    return " ".join(corrected)


# ================= MEDIAPIPE =================
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
prev_x, prev_y = None, None

# 👉 SPACE CONTROL
space_added = False
space_start_time = None

buttons = ["CLEAR", "BACK", "GENERATE", "QUIT"]

# ================= HELPERS =================
def get_fingers(lms):
    tips = [4, 8, 12, 16, 20]
    fingers = [lms.landmark[4].x < lms.landmark[3].x]
    for i in range(1, 5):
        fingers.append(lms.landmark[tips[i]].y < lms.landmark[tips[i]-2].y)
    return fingers

def center_for_ai(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

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

    resized = cv2.resize(square, (28, 28))
    return (resized / 255.0).reshape(1, 28, 28, 1)

def idx_to_char(idx):
    return chr(ord('A') + idx)

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

    # TOP BAR
    cv2.rectangle(frame, (0, 0), (w, TOP_BAR_H), (30,30,30), -1)
    btn_w = w // len(buttons)

    for i, name in enumerate(buttons):
        cv2.putText(frame, name, (i*btn_w+20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        lms = res.multi_hand_landmarks[0]
        fingers = get_fingers(lms)

        x = int(lms.landmark[8].x * w)
        y = int(lms.landmark[8].y * h)

        # ✋ FULL PALM = SPACE
        if all(fingers):
            if space_start_time is None:
                space_start_time = time.time()

            elif time.time() - space_start_time > 0.5 and not space_added:
                text_sequence.append(" ")
                print("Space added")
                space_added = True
        else:
            space_start_time = None
            space_added = False

        # BUTTON CLICK
        if y < TOP_BAR_H and fingers[1]:
            action = buttons[x // btn_w]
            print("👉 Button clicked:", action)

            if action == "CLEAR":
                text_sequence.clear()
                canvas_vis[:] = 0
                canvas_ai[:] = 0

            elif action == "BACK" and text_sequence:
                text_sequence.pop()

            elif action == "GENERATE":
                raw_text = "".join(text_sequence) if text_sequence else "HELLO"
                print("Raw:", raw_text)

                # ✅ Apply correction ONLY here
                corrected = autocorrect_text(raw_text)
                print("Corrected:", corrected)

                prompt = f"A realistic image of {corrected}"
                threading.Thread(target=generate_image, args=(prompt,)).start()

            elif action == "QUIT":
                break

            time.sleep(0.4)

        # DRAWING MODE
        elif fingers[1] and not fingers[2]:
            if prev_x is not None:
                cv2.line(canvas_vis, (prev_x, prev_y), (x, y), (255,255,255), 12)
                cv2.line(canvas_ai, (prev_x, prev_y), (x, y), (255,255,255), 12)

            prev_x, prev_y = x, y
            last_write_time = time.time()
        else:
            prev_x, prev_y = None, None

        mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

    # LETTER DETECTION
    if last_write_time and time.time() - last_write_time > PAUSE_TIME:
        processed = center_for_ai(canvas_ai)

        if processed is not None:
            pred = model.predict(processed, verbose=0)[0]
            idx = np.argmax(pred)

            if pred[idx] > CONF_THRESHOLD:
                text_sequence.append(idx_to_char(idx))

        last_write_time = None
        canvas_vis[:] = 0
        canvas_ai[:] = 0

    # DISPLAY RAW TEXT (NO CORRECTION)
    cv2.rectangle(frame, (0, h-120), (w, h), (20,20,20), -1)

    word = "".join(text_sequence)

    cv2.putText(frame, word if word else "--",
                (50, h-40), cv2.FONT_HERSHEY_DUPLEX,
                1.5, (0,255,0), 3)

    # DISPLAY IMAGE
    if generated_image is not None:
        frame[50:306, w-306:w-50] = generated_image

    # DRAW OVERLAY
    gray = cv2.cvtColor(canvas_vis, cv2.COLOR_BGR2GRAY)
    frame[gray > 0] = canvas_vis[gray > 0]

    cv2.imshow("AI Scene Generator", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
