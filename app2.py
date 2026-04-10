from ultralytics import YOLO
import cv2
import time
import torch
import threading
import queue
import loteria_bot_controller

# ==========================================
# CONFIGURATION
# ==========================================
TEST_MODE_SINGLE_CAMERA = False

camera_class_ids = {}
tablas = [
    "tabla 1", "tabla 2", "tabla 3", "tabla 4", "tabla 5",
    "tabla 6", "tabla 7", "tabla 8", "tabla 9", "tabla 10",
]

green_cards  = set()        # Cards fully handled (bean dropped)
queued_cards = set()        # Cards waiting in the drop queue
drop_queue   = queue.Queue()  # (cls, x_mid, y_mid)
queue_lock   = threading.Lock()


# ==============================================================================
# LOGGING HELPER
# ==============================================================================
def log(tag, msg):
    ts = time.strftime('%H:%M:%S')
    print(f"[{ts}] [{tag}] {msg}")


# ==============================================================================
# DROP WORKER — one thread, one drop at a time
# ==============================================================================
def drop_worker():
    log("WORKER", "Drop worker ready and waiting for matches...")
    while True:
        cls, x_mid, y_mid = drop_queue.get()

        log("WORKER", f"▶ Starting drop for class {cls}")

        try:
            # Pass class ID — controller looks up exact grid position directly
            # pixel coords passed as fallback (not used if class is in grid)
            loteria_bot_controller.drop_bean(cls, pixel_x=x_mid, pixel_y=y_mid)
            log("WORKER", f"✅ Drop complete for class {cls}. Robot parked.")
        except Exception as e:
            log("WORKER", f"❌ Drop FAILED for class {cls}: {e}")

        with queue_lock:
            green_cards.add(cls)
            queued_cards.discard(cls)

        log("WORKER", f"Cards done: {len(green_cards)} | Remaining in queue: {drop_queue.qsize()}")
        drop_queue.task_done()


# ==============================================================================
# VISION — classify detections and queue new matches
# ==============================================================================
def coordinate_objects(results, frame, shared_classes=None):
    detected_classes = set()
    total_cards = 16

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x_mid = int((x1 + x2) / 2)
                y_mid = int((y1 + y2) / 2)
                cls   = int(box.cls[0].item())

                detected_classes.add(cls)

                # Skip tabla overlay classes
                if cls not in tablas:
                    with queue_lock:
                        already_done   = cls in green_cards
                        already_queued = cls in queued_cards
                        is_match       = (shared_classes and cls in shared_classes) or TEST_MODE_SINGLE_CAMERA

                    if already_done:
                        color = (0, 255, 0)     # Green  — bean dropped

                    elif already_queued:
                        color = (0, 255, 255)   # Yellow — queued

                    elif is_match:
                        color = (0, 165, 255)   # Orange — new match

                        # Check if this class is in the known grid before queuing
                        if cls in loteria_bot_controller.CARD_GRID:
                            with queue_lock:
                                queued_cards.add(cls)
                            drop_queue.put((cls, x_mid, y_mid))
                            row, col = loteria_bot_controller.CARD_GRID[cls]
                            log("VISION", f"🎯 MATCH! Class {cls} @ grid ({row},{col}) — queue: {drop_queue.qsize()}")
                        else:
                            log("VISION", f"⚠️  Class {cls} matched but not in grid map — skipping.")

                    else:
                        color = (0, 0, 255)     # Red    — one camera only

                    cv2.circle(frame, (x_mid, y_mid), 20, color, -1)

                    # Show class name from grid if available
                    label = f"cls:{cls}"
                    if cls in loteria_bot_controller.CARD_GRID:
                        r, c = loteria_bot_controller.CARD_GRID[cls]
                        label = f"{cls} ({r},{c})"

                    cv2.putText(frame, label, (x_mid + 10, y_mid),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    if len(green_cards) >= total_cards:
        cv2.putText(frame, "LOTERIA!", (80, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)

    return detected_classes


# ==============================================================================
# MAIN
# ==============================================================================
def testing_middle_dot():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    log("SETUP", f"Using device: {device}")

    model = YOLO("runs/detect/runs/detect/loteria_yolo/weights/best.pt", verbose=False)
    model.to(device)
    log("SETUP", "YOLO model loaded.")

    threading.Thread(target=drop_worker, daemon=True).start()
    log("SETUP", "Drop worker thread started.")

    cap0 = cv2.VideoCapture(8)
    cap1 = cv2.VideoCapture(10)
    log("SETUP", f"Camera 8  open: {cap0.isOpened()}")
    log("SETUP", f"Camera 10 open: {cap1.isOpened()}")

    for cap in [cap0, cap1]:
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 10)

    frames      = {}
    frames_lock = threading.Lock()
    running     = True
    skip_frames = 2

    def capture_process(cap, camera_id):
        nonlocal running
        frame_count          = 0
        consecutive_failures = 0
        MAX_FAILURES         = 10

        while running and cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    log("CAM", f"❌ Camera {camera_id} lost. Stopping.")
                    break
                time.sleep(0.1)
                continue

            consecutive_failures = 0
            frame_count += 1

            if frame_count % skip_frames != 0:
                continue

            try:
                results = model(frame, conf=0.50, imgsz=320, verbose=False)
                annotated_frame = results[0].plot()

                other_id = 1 - camera_id
                with frames_lock:
                    shared_classes = camera_class_ids.get(other_id, set())

                detected_classes = coordinate_objects(results, annotated_frame, shared_classes)

                with frames_lock:
                    frames[camera_id]           = annotated_frame.copy()
                    camera_class_ids[camera_id] = detected_classes

            except Exception as e:
                log("CAM", f"Camera {camera_id} error: {e}")

            time.sleep(0.001)

    threads = []
    if cap0.isOpened():
        threads.append(threading.Thread(target=capture_process, args=(cap0, 0), daemon=True))
    if cap1.isOpened():
        threads.append(threading.Thread(target=capture_process, args=(cap1, 1), daemon=True))

    for t in threads:
        t.start()

    log("SETUP", "Camera threads started. Press 'r' to reset, 'q' to quit.")

    try:
        last_frames = {}

        while running:
            with frames_lock:
                frames_to_show = frames.copy()
                for k in list(frames_to_show):
                    frames.pop(k, None)

            last_frames.update(frames_to_show)

            for camera_id, frame in last_frames.items():
                with queue_lock:
                    done   = len(green_cards)
                    queued = drop_queue.qsize()

                status = f"Done: {done}/16 | Queued: {queued}"
                cv2.putText(frame, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow(f"Camera {camera_id}", frame)

            key = cv2.waitKey(10) & 0xFF

            if key == ord("r"):
                with queue_lock:
                    green_cards.clear()
                    queued_cards.clear()
                while not drop_queue.empty():
                    try:
                        drop_queue.get_nowait()
                        drop_queue.task_done()
                    except queue.Empty:
                        break
                with frames_lock:
                    camera_class_ids.clear()
                log("RESET", "♻️  Game reset — all cards cleared.")

            elif key == ord("q"):
                running = False
                log("MAIN", "Quit requested.")
                break

            if device.type == "cuda":
                torch.cuda.empty_cache()
            time.sleep(0.01)

    except KeyboardInterrupt:
        log("MAIN", "Keyboard interrupt received.")
    finally:
        running = False
        log("MAIN", "Shutting down...")

        if hasattr(loteria_bot_controller, 'emergency_stop'):
            loteria_bot_controller.emergency_stop()

        for t in threads:
            t.join(timeout=1.0)

        cap0.release()
        cap1.release()
        cv2.destroyAllWindows()

        if device.type == "cuda":
            torch.cuda.empty_cache()

        log("MAIN", "Cleanup complete.")


if __name__ == "__main__":
    testing_middle_dot()