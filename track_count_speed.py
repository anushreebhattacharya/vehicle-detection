import cv2
import math
from ultralytics import YOLO

# ---------------- CONFIG ----------------
VIDEO_PATH = "video-frames/traffic.mp4"
MODEL_PATH = "../runs/detect/train3/weights/best.pt"

LINE_Y = 1000               # counting line
LINE_THICKNESS = 3
LINE_THRESHOLD = 25       # pixels, tolerance for counting
PIXEL_TO_METER = 0.05        # adjust after calibration
FPS = 30
DISPLAY_SCALE = 0.4       # scale up for larger display
# ---------------------------------------

# Load model
model = YOLO("./runs/detect/train3/weights/best.pt")
cap = cv2.VideoCapture("video-frames/traffic.mp4")
assert cap.isOpened() 

# Track history: id -> (cx, cy)
track_history = {}
counted_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # YOLOv8 tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.3,
        verbose=False
    )

    # Draw counting line
    cv2.line(frame, (0, LINE_Y), (w, LINE_Y), (0, 0, 255), LINE_THICKNESS)
    cv2.putText(frame, "Counting Line", (10, LINE_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    if results and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, tid in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            tid = int(tid)

            speed_kmh = 0.0

            if tid in track_history:
                px, py = track_history[tid]

                # SPEED calculation
                dist_pixels = math.hypot(cx - px, cy - py)
                dist_meters = dist_pixels * PIXEL_TO_METER
                speed_mps = dist_meters * FPS
                speed_kmh = speed_mps * 3.6

                # COUNT when crossing line (with threshold)
                if tid not in counted_ids and abs(cy - LINE_Y) <= LINE_THRESHOLD:
                    counted_ids.add(tid)

            track_history[tid] = (cx, cy)

            # DRAW bounding box + center
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(frame, (cx, cy), 4, (255,0,0), -1)

            # ID + speed text
            cv2.putText(
                frame,
                f"ID {tid} | {speed_kmh:.1f} km/h",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255,255,0), 2
            )

    # Display vehicle count
    cv2.putText(
        frame,
        f"Vehicle Count: {len(counted_ids)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (0,0,255), 3
    )

    # Resize for display
    display_frame = cv2.resize(frame, (int(w*DISPLAY_SCALE), int(h*DISPLAY_SCALE)))
    cv2.imshow("Vehicle Tracking", display_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

print("✅ Total Vehicles Counted:", len(counted_ids))  