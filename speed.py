import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# ---------------- CONFIG ----------------
PIXEL_TO_METER = 0.05     # calibration factor
FRAME_GAP = 8             # frames used for speed calculation
MIN_PIXEL_MOVE = 8        # ignore jitter
MAX_SPEED = 120           # km/h cap

# ---------------- LOAD MODEL ----------------
model = YOLO("../yolov8n.pt")

# ---------------- LOAD VIDEO ----------------
cap = cv2.VideoCapture("video-frames/traffic.mp4")
if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

# ---------------- STORAGE ----------------
track_history = defaultdict(list)
vehicle_speed = {}   # stores last stable speed per ID

# ---------------- PROCESS VIDEO ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video finished")
        break

    results = model.track(frame, persist=True, classes=[2, 3, 5, 7])

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, track_id in zip(boxes, ids):
            track_id = int(track_id)
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            track_history[track_id].append((cx, cy))

            # ---- SPEED CALCULATION (STABLE) ----
            if len(track_history[track_id]) >= FRAME_GAP:
                x_prev, y_prev = track_history[track_id][-FRAME_GAP]
                x_curr, y_curr = track_history[track_id][-1]

                pixel_dist = np.hypot(x_curr - x_prev, y_curr - y_prev)

                if pixel_dist >= MIN_PIXEL_MOVE:
                    time_sec = FRAME_GAP / fps
                    speed_kmph = (pixel_dist * PIXEL_TO_METER / time_sec) * 3.6
                    speed_kmph = min(speed_kmph, MAX_SPEED)
                    speed_kmph = round(speed_kmph, 2)

                    # store stable speed
                    vehicle_speed[track_id] = speed_kmph

                    # ---- PRINT TO TERMINAL ----
                    print(f"Vehicle ID {track_id} Speed: {speed_kmph} km/h")

            # ---- DRAW USING LAST STABLE SPEED ----
            speed_to_show = vehicle_speed.get(track_id, 0)

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"ID:{track_id} Speed:{speed_to_show} km/h",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    # ---- DISPLAY ----
    frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
    cv2.imshow("Vehicle Speed Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
