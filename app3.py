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
TEST_MODE_SINGLE_CAMERA = False

class_color = {}
camera_class_ids = {}
tablas = [
    "tabla 1", "tabla 2", "tabla 3", "tabla 4", "tabla 5",
    "tabla 6", "tabla 7", "tabla 8", "tabla 9", "tabla 10",
]

green_cards = set()


def coordinate_objects(results, frame, called_class=None):
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
                    color = (0, 0, 255)  # Red by default
                    if class_color.get(cls) == True:
                        color = (0, 255, 0)  # Already confirmed green
                    elif called_class is not None and cls == called_class:
                        color = (0, 255, 0)  # Green — matched the typed ID!
                        class_color[cls] = True
                        green_cards.add(cls)

                        print(f"\n=======================================================")
                        print(f"🟢 MATCH! Class {cls} found on board at ({x_mid}, {y_mid})!")
                        print(f"Triggering robot to drop bean...")
                        print(f"=======================================================\n")

                        threading.Thread(
                            target=loteria_bot_controller.drop_bean,
                            args=(x_mid, y_mid),
                            daemon=True
                        ).start()

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

    if len(green_cards) == total_cards:
        cv2.putText(
            frame, "LOTERIA", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5
        )

    return detected_classes


def keyboard_input_thread(frames_lock, running_ref):
    """
    Runs in a background thread. You type a class ID (integer) and press Enter.
    It checks if camera 0 currently sees that class — if yes, marks it green.
    """
    print("\n📢 Keyboard caller ready! Type a class ID and press Enter to call it.")
    print("   (Type 'r' to reset, 'q' to quit)\n")

    while running_ref[0]:
        try:
            user_input = input("Call class ID: ").strip()
        except EOFError:
            break  # stdin closed

        if user_input.lower() == 'q':
            running_ref[0] = False
            print("Quitting...")
            break
        elif user_input.lower() == 'r':
            green_cards.clear()
            class_color.clear()
            with frames_lock:
                camera_class_ids.clear()
            print("🔄 GAME RESET - All cards cleared")
            continue

        try:
            called_id = int(user_input)
        except ValueError:
            print(f"❌ '{user_input}' is not a valid class ID. Enter an integer.")
            continue

        with frames_lock:
            visible = camera_class_ids.get(0, set())

        if called_id in visible:
            # Trigger the match — set it as the "called class" so the next frame paints it green
            # We use a shared dict so the capture thread picks it up
            with frames_lock:
                camera_class_ids['pending_call'] = called_id
            print(f"✅ Class {called_id} is ON THE BOARD — marking green!")
        else:
            print(f"⬜  Class {called_id} not currently visible on the board.")


def testing_middle_dot():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    model = YOLO("runs/detect/runs/detect/loteria_yolo/weights/best.pt")
    model.to(device)

    cap0 = cv2.VideoCapture(0)
    cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frames = {}
    frames_lock = threading.Lock()
    running_ref = [True]  # list so threads can mutate it
    skip_frames = 2

    def capture_process(cap, camera_id):
        frame_count = 0

        while running_ref[0] and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print(f"Lost connection to Camera {camera_id}")
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            try:
                # Check if keyboard thread has a pending called class
                with frames_lock:
                    called_class = camera_class_ids.pop('pending_call', None)

                confidence = 0.50
                results = model(frame, conf=confidence, imgsz=320)
                annotated_frame = results[0].plot()

                detected_classes = coordinate_objects(results, annotated_frame, called_class)

                with frames_lock:
                    frames[camera_id] = annotated_frame.copy()
                    camera_class_ids[camera_id] = detected_classes

            except Exception as e:
                print(f"Error processing frame from camera {camera_id}: {e}")

            time.sleep(0.001)

    # Start camera thread
    cam_thread = threading.Thread(target=capture_process, args=(cap0, 0))
    cam_thread.daemon = True
    cam_thread.start()

    # Start keyboard input thread
    kb_thread = threading.Thread(target=keyboard_input_thread, args=(frames_lock, running_ref))
    kb_thread.daemon = True
    kb_thread.start()

    try:
        last_frames = {}

        while running_ref[0]:
            frames_to_show = {}
            with frames_lock:
                frames_to_show = frames.copy()
                frames.clear()

            for camera_id, frame in frames_to_show.items():
                if isinstance(camera_id, int):  # skip 'pending_call' key
                    last_frames[camera_id] = frame

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
                running_ref[0] = False
                print("Quitting application...")
                break

            torch.cuda.empty_cache() if device.type == "cuda" else None
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        running_ref[0] = False
        print("Cleaning up resources...")

        if hasattr(loteria_bot_controller, 'emergency_stop'):
            loteria_bot_controller.emergency_stop()

        cam_thread.join(timeout=1.0)
        cap0.release()
        cv2.destroyAllWindows()
        torch.cuda.empty_cache() if device.type == "cuda" else None
        print("Cleanup complete")


if __name__ == "__main__":
    testing_middle_dot()