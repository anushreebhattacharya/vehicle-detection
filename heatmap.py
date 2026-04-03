import cv2
import numpy as np

# ---------------- CONFIG ----------------
VIDEO_PATH = "video-frames/traffic.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Cannot open video file: {VIDEO_PATH}")
    exit()

heatmap = None
prev_gray = None
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if prev_gray is None:
        prev_gray = gray
        heatmap = np.zeros_like(gray, dtype=np.float32)
        continue

    # -------- MOTION DETECTION --------
    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Remove noise
    thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Accumulate motion areas
    heatmap += thresh.astype(np.float32)

    prev_gray = gray

    # -------- LIVE DISPLAY --------
    live_heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    live_heatmap = live_heatmap.astype(np.uint8)
    colored_live = cv2.applyColorMap(live_heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame, 0.6, colored_live, 0.4, 0)
    cv2.imshow("Live Vehicle Heatmap", cv2.resize(overlay, (800, 450)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# -------- FINAL HEATMAP --------
heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
heatmap = heatmap.astype(np.uint8)
colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

cv2.imshow("Final Traffic Heatmap", cv2.resize(colored_heatmap, (900, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"✅ Processed {frame_count} frames successfully.")
