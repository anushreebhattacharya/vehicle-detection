import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# -----------------------------
# CONFIG
# -----------------------------
PIXEL_TO_METER = 0.05   # adjust after calibration
FPS = 30                # set your video FPS
SECONDS_INTERVAL = 1

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO("yolov8n.pt")  # lightweight model

# -----------------------------
# STORAGE
# -----------------------------
prev_positions = {}
prev_frame = {}
speed_records = defaultdict(list)

frame_count = 0

# -----------------------------
# SPEED FUNCTION
# -----------------------------
def calculate_speed(p1, p2, time_diff):
    pixel_dist = np.linalg.norm(np.array(p1) - np.array(p2))
    meter_dist = pixel_dist * PIXEL_TO_METER
    speed_mps = meter_dist / time_diff
    return speed_mps * 3.6  # km/h

# -----------------------------
# VIDEO
# -----------------------------
cap = cv2.VideoCapture("traffic.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # -----------------------------
    # YOLO TRACKING
    # -----------------------------
    results = model.track(frame, persist=True)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, obj_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            obj_id = int(obj_id)

            if obj_id in prev_positions:
                frame_diff = frame_count - prev_frame[obj_id]
                time_diff = frame_diff / FPS

                if time_diff >= SECONDS_INTERVAL:
                    speed = calculate_speed(prev_positions[obj_id], (cx, cy), time_diff)

                    speed_records[obj_id].append(speed)

                    print(f"Vehicle ID {obj_id} -> Speed: {speed:.2f} km/h")

                    prev_positions[obj_id] = (cx, cy)
                    prev_frame[obj_id] = frame_count
            else:
                prev_positions[obj_id] = (cx, cy)
                prev_frame[obj_id] = frame_count

            # Draw
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f"ID {obj_id}", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# -----------------------------
# FINAL OUTPUT
# -----------------------------
print("\n--- Average Speed per Vehicle ---")

for obj_id, speeds in speed_records.items():
    if speeds:
        avg_speed = sum(speeds) / len(speeds)
        print(f"Vehicle ID {obj_id} -> Average Speed: {avg_speed:.2f} km/h")