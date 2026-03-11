import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow.keras import models, layers

# ================= CNN =================
def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

model = build_cnn()
model.load_weights("mnist_cnn_model.h5")
print("✅ CNN loaded")

# ================= MediaPipe =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# ================= State =================
cap = cv2.VideoCapture(0)

canvas_vis = None   # what user sees (colored)
canvas_ai  = None   # what CNN sees (white only)

TOP_BAR_H = 80
PAUSE_TIME = 1.0
CONF_THRESHOLD = 0.88

digit_sequence = []
digit_locked = False
last_write_time = None
writing_now = False

tools = [
    {"name": "PEN",    "val": (255, 255, 255)},
    {"name": "RED",    "val": (0, 0, 255)},
    {"name": "BLUE",   "val": (255, 0, 0)},
    {"name": "GREEN",  "val": (0, 255, 0)},
    {"name": "ERASER", "val": (0, 0, 0)},
    {"name": "CLEAR",  "val": None},
    {"name": "QUIT",   "val": None}
]

selected_color = (255, 255, 255)
current_tool = "PEN"
prev_x, prev_y = None, None

# ================= Helpers =================
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
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    if w * h < 500:
        return None

    crop = thresh[y:y+h, x:x+w]
    size = max(w, h) + 20

    square = np.zeros((size, size), dtype=np.uint8)
    square[(size-h)//2:(size-h)//2+h,
           (size-w)//2:(size-w)//2+w] = crop

    resized = cv2.resize(square, (20, 20))
    resized = np.pad(resized, ((4,4), (4,4)), "constant")

    return (resized / 255.0).reshape(1, 28, 28, 1)

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas_vis is None:
        canvas_vis = np.zeros_like(frame)
        canvas_ai  = np.zeros_like(frame)

    # ---------- TOOLBAR ----------
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, TOP_BAR_H), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

    btn_w = w // len(tools)
    for i, tool in enumerate(tools):
        x1 = i * btn_w
        x_center = x1 + btn_w // 2

        if tool["name"] == current_tool:
            cv2.rectangle(frame, (x1+5, 5),
                          (x1+btn_w-5, TOP_BAR_H-5),
                          (0,255,255), 2)
            col = (0,255,255)
        else:
            col = (255,255,255)

        cv2.putText(frame, tool["name"],
                    (x_center-30, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, col, 1)

        if tool["val"] is not None and tool["name"] not in ["CLEAR","QUIT"]:
            cv2.circle(frame, (x_center, 60), 8, tool["val"], -1)
            cv2.circle(frame, (x_center, 60), 9, (255,255,255), 1)

    # ---------- HAND TRACKING ----------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    writing_now = False

    if res.multi_hand_landmarks:
        lms = res.multi_hand_landmarks[0]
        fingers = get_fingers(lms)
        x, y = int(lms.landmark[8].x * w), int(lms.landmark[8].y * h)

        # TOOL SELECTION
        if y < TOP_BAR_H and fingers[1] and fingers[2]:
            idx = x // btn_w
            target = tools[idx]["name"]

            if target == "QUIT":
                break
            elif target == "CLEAR":
                canvas_vis[:] = 0
                canvas_ai[:] = 0
                digit_sequence.clear()
                digit_locked = False
            else:
                current_tool = target
                selected_color = tools[idx]["val"]

        # ERASER
        elif (all(not f for f in fingers)) or (current_tool == "ERASER" and fingers[1]):
            cv2.circle(canvas_vis, (x, y), 35, (0,0,0), -1)
            cv2.circle(canvas_ai,  (x, y), 35, (0,0,0), -1)
            prev_x, prev_y = None, None

        # WRITING
        elif fingers[1] and not fingers[2]:
            writing_now = True
            digit_locked = False

            if prev_x is not None and y > TOP_BAR_H:
                # visual canvas (color)
                cv2.line(canvas_vis, (prev_x, prev_y),
                         (x, y), selected_color, 7)
                # AI canvas (white only)
                cv2.line(canvas_ai, (prev_x, prev_y),
                         (x, y), (255,255,255), 7)

            prev_x, prev_y = x, y
            last_write_time = time.time()

        else:
            prev_x, prev_y = None, None

        mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

    # ---------- MULTI-DIGIT PREDICTION ----------
    if last_write_time and not writing_now:
        if time.time() - last_write_time > PAUSE_TIME and not digit_locked:
            processed = center_for_ai(canvas_ai)
            if processed is not None:
                pred = model.predict(processed, verbose=0)[0]
                conf = np.max(pred)
                digit = int(np.argmax(pred))

                if conf > CONF_THRESHOLD:
                    digit_sequence.append(digit)
                    print(f"Digit: {digit}, Confidence: {conf:.2f}")
                    digit_locked = True
                    canvas_vis[:] = 0
                    canvas_ai[:] = 0

    # ---------- DISPLAY ----------
    number_str = "".join(map(str, digit_sequence)) if digit_sequence else "--"
    cv2.putText(frame, f"NUMBER: {number_str}",
                (w - 380, h - 40),
                cv2.FONT_HERSHEY_TRIPLEX, 1.3, (0,255,0), 2)

    # Overlay visible canvas
    gray = cv2.cvtColor(canvas_vis, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    frame[mask > 0] = canvas_vis[mask > 0]

    cv2.imshow("Air Writing - Multi Digit Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
