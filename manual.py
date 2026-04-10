import cv2
import time
import torch
import threading
from ultralytics import YOLO
import loteria_bot_controller

CAMERA_INDEX = 0

JOG_STEPS  = [0.05, 0.1, 0.2, 0.5]
JOG_LABELS = ["fine 0.05", "medium 0.1", "coarse 0.2", "large 0.5"]
jog_idx    = 1

# Shared between camera thread and main display loop
latest_frame      = None
latest_detections = []  # list of (x1, y1, x2, y2, cx, cy, name)
frame_lock        = threading.Lock()
running           = True


def get_pos_string():
    pos = loteria_bot_controller.get_position()
    if pos:
        return f"X:{pos[0]:+.3f}  Y:{pos[1]:.3f}"
    return "pos unknown"


def camera_thread(cap, model):
    """Runs in background — grabs frames, runs YOLO, stores results."""
    global latest_frame, latest_detections, running

    skip = 0
    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        skip += 1
        detections = []

        # Run YOLO every 2 frames
        if skip % 2 == 0:
            results = model(frame, conf=0.45, imgsz=320, verbose=False)
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls  = int(box.cls[0].item())
                    cx   = (x1 + x2) // 2
                    cy   = (y1 + y2) // 2
                    name = loteria_bot_controller.CARD_NAMES.get(cls, None)
                    if cls in loteria_bot_controller.CARD_GRID and name:
                        detections.append((x1, y1, x2, y2, cx, cy, name))

        with frame_lock:
            latest_frame = frame.copy()
            if detections:  # only update if we got new results
                latest_detections = detections

        time.sleep(0.001)


def draw_detections(frame, detections):
    for (x1, y1, x2, y2, cx, cy, name) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
        cv2.putText(frame, name, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return frame


def draw_status(frame, status_msg):
    label = JOG_LABELS[jog_idx]
    cv2.putText(frame, f"Step: {label}  |  {status_msg}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame


def main():
    global jog_idx, running

    print("\n[app4] Manual CNC Control + Card Detection")
    print("[app4] WASD=move  SPACE=drop  O=origin  Z=zero  P=pos  +/-=step  Q=quit\n")

    model = YOLO("runs/detect/runs/detect/loteria_yolo/weights/best.pt", verbose=False)
    model.to(torch.device("cpu"))
    print("[app4] YOLO loaded.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[app4] ❌ Could not open camera {CAMERA_INDEX}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    for _ in range(5):
        cap.read()

    # Start background camera + detection thread
    t = threading.Thread(target=camera_thread, args=(cap, model), daemon=True)
    t.start()

    status_msg = f"Ready — {get_pos_string()}"
    cv2.namedWindow("app4")
    print(f"[app4] {status_msg}")

    while running:
        # Grab latest frame and detections from background thread
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            frame      = latest_frame.copy()
            detections = list(latest_detections)

        frame = draw_detections(frame, detections)
        frame = draw_status(frame, status_msg)
        cv2.imshow("app4", frame)

        key = cv2.waitKey(10) & 0xFF

        if key == 255:
            continue

        step = JOG_STEPS[jog_idx]

        if key == ord('w') or key == ord('W'):
            loteria_bot_controller.jog('Y', -step)
            status_msg = f"↑ W  {get_pos_string()}"
            print(f"[app4] {status_msg}")

        elif key == ord('s') or key == ord('S'):
            loteria_bot_controller.jog('Y', +step)
            status_msg = f"↓ S  {get_pos_string()}"
            print(f"[app4] {status_msg}")

        elif key == ord('a') or key == ord('A'):
            loteria_bot_controller.jog('X', -step)
            status_msg = f"← A  {get_pos_string()}"
            print(f"[app4] {status_msg}")

        elif key == ord('d') or key == ord('D'):
            loteria_bot_controller.jog('X', +step)
            status_msg = f"→ D  {get_pos_string()}"
            print(f"[app4] {status_msg}")

        elif key == ord(' '):
            print("[app4] 👇 Dropping bean...")
            loteria_bot_controller.send_gcode("M3 S90")
            time.sleep(0.5)
            loteria_bot_controller.send_gcode("M3 S0")
            status_msg = f"Bean dropped!  {get_pos_string()}"
            print(f"[app4] {status_msg}")

        elif key == ord('o') or key == ord('O'):
            print("[app4] 🅿️  Going to origin...")
            loteria_bot_controller.send_gcode("G90")
            loteria_bot_controller.send_gcode(f"G1 Y{loteria_bot_controller.ORIGIN_Y} F{loteria_bot_controller.SPEED_Y}")
            loteria_bot_controller.send_gcode(f"G1 X{loteria_bot_controller.ORIGIN_X} F{loteria_bot_controller.SPEED_X}")
            status_msg = f"At origin  {get_pos_string()}"
            print(f"[app4] {status_msg}")

        elif key == ord('z') or key == ord('Z'):
            loteria_bot_controller.send_gcode("G10 L20 P1 X0 Y0 Z0")
            status_msg = "Zeroed — (0,0) set here"
            print(f"[app4] ✅ {status_msg}")

        elif key == ord('p') or key == ord('P'):
            status_msg = get_pos_string()
            print(f"[app4] 📍 {status_msg}")

        elif key == ord('+') or key == ord('='):
            jog_idx = min(jog_idx + 1, len(JOG_STEPS) - 1)
            status_msg = f"Step → {JOG_LABELS[jog_idx]}"
            print(f"[app4] {status_msg}")

        elif key == ord('-'):
            jog_idx = max(jog_idx - 1, 0)
            status_msg = f"Step → {JOG_LABELS[jog_idx]}"
            print(f"[app4] {status_msg}")

        elif key == ord('q') or key == ord('Q') or key == 27:
            print("[app4] Quitting...")
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()
    t.join(timeout=1.0)
    print("[app4] Done.")


if __name__ == "__main__":
    main()