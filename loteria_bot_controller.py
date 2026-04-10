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

class_color = {}
camera_class_ids = {}
tablas = [
    "tabla 1", "tabla 2", "tabla 3", "tabla 4", "tabla 5",
    "tabla 6", "tabla 7", "tabla 8", "tabla 9", "tabla 10",
]

green_cards = set()       # Cards fully handled (bean dropped)
queued_cards = set()      # Cards currently waiting in the drop queue
drop_queue = queue.Queue()  # Thread-safe queue of (cls, x_mid, y_mid)
queue_lock = threading.Lock()


# ==============================================================================
# DROP WORKER — runs in its own thread, processes one drop at a time
# ==============================================================================
def drop_worker():
    while True:
        cls, x_mid, y_mid = drop_queue.get()  # Blocks until something is in the queue

        print(f"\n=======================================================")
        print(f"🟢 DROP WORKER: Handling class {cls} at ({x_mid}, {y_mid})")
        print(f"=======================================================\n")

        try:
            loteria_bot_controller.drop_bean(x_mid, y_mid)
        except Exception as e:
            print(f"[DROP WORKER] Error during drop: {e}")

        # Mark as fully done after the robot physically returns
        with queue_lock:
            green_cards.add(cls)
            queued_cards.discard(cls)

        print(f"[DROP WORKER] ✅ Class {cls} complete. Ready for next card.")
        drop_queue.task_done()


# ==============================================================================
# VISION — draw dots and queue new matches
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
                cls = int(box.cls[0].item())

                detected_classes.add(cls)

                if cls not in tablas:
                    with queue_lock:
                        already_done = cls in green_cards
                        already_queued = cls in queued_cards
                        is_match = (shared_classes and cls in shared_classes) or TEST_MODE_SINGLE_CAMERA

                    if already_done:
                        color = (0, 255, 0)  # Green — already handled
                    elif already_queued:
                        color = (0, 255, 255)  # Yellow — in queue, robot on its way
                    elif is_match:
                        color = (0, 165, 255)  # Orange — just matched, queuing now
                        with queue_lock:
                            queued_cards.add(cls)
                        drop_queue.put((cls, x_mid, y_mid))
                        print(f"[VISION] Queued class {cls} at ({x_mid}, {y_mid}). Queue size: {drop_queue.qsize()}")
                    else:
                        color = (0, 0, 255)  # Red — seen by only one camera

                    cv2.circle(frame, (x_mid, y_mid), 20, color, -1)
                    cv2.putText(
                        frame,
                        f"({x_mid},{y_mid})",
                        (x_mid + 10, y_mid),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

    if len(green_cards) >= total_cards:
        cv2.putText(frame, "LOTERIA", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)

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
    print(f"Using device: {device}")

    model = YOLO("runs/detect/runs/detect/loteria_yolo/weights/best.pt")
    model.to(device)

    # Start the drop worker thread (one at a time, sequential drops)
    worker_thread = threading.Thread(target=drop_worker, daemon=True)
    worker_thread.start()
    print("[SETUP] Drop worker thread started.")

    cap0 = cv2.VideoCapture(8)
    cap1 = cv2.VideoCapture(10)

    print(f"Camera 1 (Index 8) open: {cap0.isOpened()}")
    print(f"Camera 2 (Index 10) open: {cap1.isOpened()}")

    for cap in [cap0, cap1]:
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 10)

    frames = {}
    frames_lock = threading.Lock()
    running = True
    skip_frames = 2

    def capture_process(cap, camera_id):
        nonlocal running
        frame_count = 0
        consecutive_failures = 0
        MAX_FAILURES = 10

        while running and cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                consecutive_failures += 1
                print(f"Camera {camera_id}: Read failed ({consecutive_failures}/{MAX_FAILURES})")
                if consecutive_failures >= MAX_FAILURES:
                    print(f"Lost connection to Camera {camera_id}")
                    break
                time.sleep(0.1)
                continue

            consecutive_failures = 0
            frame_count += 1

            if frame_count % skip_frames != 0:
                continue

            try:
                results = model(frame, conf=0.50, imgsz=320)
                annotated_frame = results[0].plot()

                other_camera_id = 1 - camera_id
                with frames_lock:
                    shared_classes = camera_class_ids.get(other_camera_id, set())

                detected_classes = coordinate_objects(results, annotated_frame, shared_classes)

                with frames_lock:
                    frames[camera_id] = annotated_frame.copy()
                    camera_class_ids[camera_id] = detected_classes

            except Exception as e:
                print(f"Error processing frame from camera {camera_id}: {e}")

            time.sleep(0.001)

    threads = []
    if cap0.isOpened():
        threads.append(threading.Thread(target=capture_process, args=(cap0, 0)))
    if cap1.isOpened():
        threads.append(threading.Thread(target=capture_process, args=(cap1, 1)))

    for t in threads:
        t.daemon = True
        t.start()

    try:
        last_frames = {}

        while running:
            with frames_lock:
                frames_to_show = frames.copy()
                for k in list(frames_to_show.keys()):
                    frames.pop(k, None)

            for camera_id, frame in frames_to_show.items():
                last_frames[camera_id] = frame

            for camera_id, frame in last_frames.items():
                # Show queue status on screen
                status = f"Done:{len(green_cards)} | Queued:{drop_queue.qsize()}"
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow(f"Camera {camera_id}", frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord("r"):
                with queue_lock:
                    green_cards.clear()
                    queued_cards.clear()
                # Drain the queue
                while not drop_queue.empty():
                    try:
                        drop_queue.get_nowait()
                        drop_queue.task_done()
                    except queue.Empty:
                        break
                with frames_lock:
                    camera_class_ids.clear()
                print("GAME RESET - All cards cleared")
            elif key == ord("q"):
                running = False
                print("Quitting...")
                break

            if device.type == "cuda":
                torch.cuda.empty_cache()
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        running = False
        print("Cleaning up...")

        if hasattr(loteria_bot_controller, 'emergency_stop'):
            loteria_bot_controller.emergency_stop()

        for t in threads:
            t.join(timeout=1.0)

        if cap0.isOpened():
            cap0.release()
        if cap1.isOpened():
            cap1.release()
        cv2.destroyAllWindows()

        if device.type == "cuda":
            torch.cuda.empty_cache()
        print("Cleanup complete")


if __name__ == "__main__":
    testing_middle_dot()