from ultralytics import YOLO
import cv2
import time
import os
import torch
import threading
import loteria_bot_controller

# ==========================================
# CONFIGURATION
# ==========================================
# Set this to True if you only have 1 camera connected and want to test the CNC movement.
# When True, it will drop a bean on ANY card it detects immediately.
TEST_MODE_SINGLE_CAMERA = False


class_color = {}
camera_class_ids = {}
tablas = [
    "tabla 1",
    "tabla 2",
    "tabla 3",
    "tabla 4",
    "tabla 5",
    "tabla 6",
    "tabla 7",
    "tabla 8",
    "tabla 9",
    "tabla 10",
]


green_cards = set()


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
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())

                detected_classes.add(cls)

                if cls not in tablas:
                    color = (0, 0, 255)  # Red
                    if class_color.get(cls) == True:
                        color = (0, 255, 0)  # Green
                    elif (shared_classes and cls in shared_classes) or TEST_MODE_SINGLE_CAMERA:
                        color = (0, 255, 0)  # Green
                        class_color[cls] = True  # remember it was seen by both cameras (or we are testing)
                        green_cards.add(cls)

                        print(f"\n=======================================================")
                        print(f"🟢 BINGO! Class {cls} matched at pixel coords ({x_mid}, {y_mid})!")
                        print(f"Triggering robot to drop bean...")
                        print(f"=======================================================\n")

                        # Trigger CNC to drop bean at the midpoint of the detected square
                        # We run this in a separate thread so it doesn't freeze the camera feed
                        threading.Thread(target=loteria_bot_controller.drop_bean, args=(x_mid, y_mid), daemon=True).start()

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

    if green_cards == total_cards:
        cv2.putText(
            frame, "LOTERIA", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5
        )

    return detected_classes


def testing_middle_dot():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    model = YOLO("runs/detect/runs/detect/loteria_yolo/weights/best.pt")  # for .pt
    model.to(device)  # for .pt

    # -------------------------------------------------------------
    # CAMERA FIX FOR RASPBERRY PI WITH TWO USB CAMERAS
    # USB Camera #1 is always 0.
    # USB Camera #2 is always 2. (Index 1 is taken by Camera #1's metadata/audio).
    # -------------------------------------------------------------
    cap0 = cv2.VideoCapture(8)
    cap1 = cv2.VideoCapture(10)

    print(f"Camera 1 (Index 0) open: {cap0.isOpened()}")
    print(f"Camera 2 (Index 2) open: {cap1.isOpened()}")

    # Reduce resolution and throttle FPS to 10 to prevent Raspberry Pi USB 2.0 bandwidth crashes
    if cap0.isOpened():
        cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap0.set(cv2.CAP_PROP_FPS, 10)
    if cap1.isOpened():
        cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap1.set(cv2.CAP_PROP_FPS, 10)

    # Setting up lock
    frames = {}
    frames_lock = threading.Lock()
    running = True

    # Skipping frames to reduce load
    skip_frames = 2

    def capture_process(cap, camera_id):
        nonlocal running

        # Give USB camera time to fully initialize before first read
        time.sleep(2.0)

        frame_count = 0
        consecutive_failures = 0
        MAX_FAILURES = 10  # Allow some transient failures before giving up

        while running and cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                consecutive_failures += 1
                print(f"Camera {camera_id}: Read failed ({consecutive_failures}/{MAX_FAILURES})")
                if consecutive_failures >= MAX_FAILURES:
                    print(f"Lost connection to Camera {camera_id}")
                    break
                time.sleep(0.1)  # Brief wait before retry
                continue

            consecutive_failures = 0  # Reset on successful read
            frame_count += 1

            if frame_count % skip_frames != 0:
                continue

            try:
                confidence = 0.50
                results = model(frame, conf=confidence, imgsz=320)
                annotated_frame = results[0].plot()

                # Grab other camera's classes
                other_camera_id = 1 - camera_id
                with frames_lock:
                    shared_classes = camera_class_ids.get(other_camera_id, set())

                # Detect classes in this frame and draw
                detected_classes = coordinate_objects(
                    results, annotated_frame, shared_classes
                )

                # Save this frame and its detected classes
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

    for thread in threads:
        thread.daemon = True
        thread.start()

    try:
        last_frames = {}

        while running:
            frames_to_show = {}
            with frames_lock:
                frames_to_show = frames.copy()
                # Only clear frames we already consumed (avoids dropping frames mid-read)
                for k in list(frames_to_show.keys()):
                    frames.pop(k, None)

            for camera_id, frame in frames_to_show.items():
                last_frames[camera_id] = frame

            # Show most recent frames
            for camera_id, frame in last_frames.items():
                cv2.imshow(f"Camera {camera_id}", frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord("r"):
                green_cards.clear()
                class_color.clear()
                print("GAME RESET - All cards cleared")
                with frames_lock:
                    camera_class_ids.clear()
            elif key == ord("q"):
                running = False
                print("Quitting application...")
                break

            # Release GPU memory
            torch.cuda.empty_cache() if device.type == "cuda" else None
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        running = False
        print("Cleaning up resources...")

        # CRITICAL STOP: Power off motors and stop queue
        if hasattr(loteria_bot_controller, 'emergency_stop'):
            loteria_bot_controller.emergency_stop()

        # Wait for threads to finish
        for thread in threads:
            thread.join(timeout=1.0)

        if cap0.isOpened():
            cap0.release()
        if cap1.isOpened():
            cap1.release()
        cv2.destroyAllWindows()

        torch.cuda.empty_cache() if device.type == "cuda" else None
        print("Cleanup complete")


if __name__ == "__main__":
    testing_middle_dot()