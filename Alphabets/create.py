import cv2
import mediapipe as mp
import numpy as np
import time
import os

# ================= DATASET SETTINGS =================
DATASET_PATH = "air_dataset"
CURRENT_LABEL = None
SAVE_DELAY = 1.2

os.makedirs(DATASET_PATH, exist_ok=True)

# ================= MediaPipe =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

cap = cv2.VideoCapture(0)
canvas = None

prev_x, prev_y = None, None
last_write_time = None
writing_now = False

# ================= FINGER DETECTION =================
def get_fingers(lms):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    fingers.append(lms.landmark[4].x < lms.landmark[3].x)

    # Other fingers
    for i in range(1, 5):
        fingers.append(lms.landmark[tips[i]].y <
                       lms.landmark[tips[i]-2].y)
    return fingers

# ================= PREPROCESS =================
def process_for_saving(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    if np.sum(thresh) < 800:
        return None

    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    if w*h < 200:
        return None

    crop = thresh[y:y+h, x:x+w]

    size = max(w, h) + 35
    square = np.zeros((size, size), dtype=np.uint8)
    square[(size-h)//2:(size-h)//2+h,
           (size-w)//2:(size-w)//2+w] = crop

    resized = cv2.resize(square, (28, 28),
                         interpolation=cv2.INTER_AREA)

    return resized

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    writing_now = False

    if result.multi_hand_landmarks:
        lms = result.multi_hand_landmarks[0]
        fingers = get_fingers(lms)

        x = int(lms.landmark[8].x * w)
        y = int(lms.landmark[8].y * h)

        # ✌ TWO FINGERS → STOP WRITING
        if fingers[1] and fingers[2]:
            prev_x, prev_y = None, None

        # ☝ ONE FINGER → WRITE
        elif fingers[1] and not fingers[2]:
            writing_now = True

            if prev_x is not None:
                cv2.line(canvas,
                         (prev_x, prev_y),
                         (x, y),
                         (0,255,0), 10)  # 🔥 GREEN PEN

            prev_x, prev_y = x, y
            last_write_time = time.time()

        else:
            prev_x, prev_y = None, None

        mp_draw.draw_landmarks(frame,
                               lms,
                               mp_hands.HAND_CONNECTIONS)
    else:
        prev_x, prev_y = None, None

    # -------- SAVE AFTER PAUSE --------
    if last_write_time and not writing_now:
        if time.time() - last_write_time > SAVE_DELAY:
            if CURRENT_LABEL is not None:
                processed = process_for_saving(canvas)

                if processed is not None:
                    label_folder = os.path.join(DATASET_PATH,
                                                CURRENT_LABEL)
                    os.makedirs(label_folder, exist_ok=True)

                    count = len(os.listdir(label_folder))
                    save_path = os.path.join(label_folder,
                                             f"{CURRENT_LABEL}_{count}.png")

                    cv2.imwrite(save_path, processed)
                    print(f"Saved {save_path}")

                canvas[:] = 0
                last_write_time = None

    # -------- DISPLAY --------
    display = frame.copy()
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    display[mask > 0] = canvas[mask > 0]

    label_text = CURRENT_LABEL if CURRENT_LABEL else "None"
    cv2.putText(display,
                f"Collecting: {label_text}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.imshow("Air Dataset Collector", display)

    key = cv2.waitKey(1) & 0xFF

    # Press A-Z to choose label
    if 65 <= key <= 90:
        CURRENT_LABEL = chr(key)
        print(f"Now collecting: {CURRENT_LABEL}")

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()