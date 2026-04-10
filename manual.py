import cv2
import time
import threading
import loteria_bot_controller

# ==============================================================================
# app4.py — Manual CNC control with live camera feed
# No automation, no YOLO, just you and the robot.
#
# CONTROLS (click the camera window first to focus it):
#   W = move up    (Y-)
#   S = move down  (Y+)
#   A = move left  (X-)
#   D = move right (X+)
#   SPACE = drop bean (servo)
#   O = go to origin (park)
#   Z = zero current position as (0,0)
#   + = increase step size
#   - = decrease step size
#   P = print current position to terminal
#   Q = quit
# ==============================================================================

CAMERA_INDEX = 8

# Jog steps (GRBL units)
JOG_STEPS  = [0.05, 0.1, 0.2, 0.5]
JOG_LABELS = ["fine 0.05", "medium 0.1", "coarse 0.2", "large 0.5"]
jog_idx    = 1  # start at medium


def get_pos_string():
    pos = loteria_bot_controller.get_position()
    if pos:
        return f"X:{pos[0]:+.3f}  Y:{pos[1]:.3f}"
    return "pos unknown"


def draw_overlay(frame, jog_idx, status_msg):
    h, w = frame.shape[:2]

    # Dark banner at bottom
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 130), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    step = JOG_STEPS[jog_idx]
    label = JOG_LABELS[jog_idx]

    cv2.putText(frame, "MANUAL CONTROL MODE",
                (10, h - 108), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, "W=up  S=down  A=left  D=right",
                (10, h - 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"Step: {label}   (+) bigger  (-) smaller",
                (10, h - 57), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "SPACE=drop bean   O=origin   Z=zero   P=pos   Q=quit",
                (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, status_msg,
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    return frame


def main():
    global jog_idx

    print("\n[app4] Manual CNC Control")
    print("[app4] Click the camera window and use WASD to move the robot.")
    print("[app4] Press Q to quit.\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[app4] ❌ Could not open camera {CAMERA_INDEX}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Warm up camera
    for _ in range(5):
        cap.read()

    status_msg = f"Ready — {get_pos_string()}"
    win = "app4 — Manual Control (click here to focus)"
    cv2.namedWindow(win)

    print(f"[app4] {status_msg}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[app4] Camera read failed.")
            break

        frame = draw_overlay(frame, jog_idx, status_msg)
        cv2.imshow(win, frame)

        key = cv2.waitKey(30) & 0xFF

        if key == 255:
            continue  # no key pressed

        step = JOG_STEPS[jog_idx]

        # Movement
        if key == ord('w') or key == ord('W'):
            loteria_bot_controller.jog('Y', -step)
            status_msg = f"↑ W  step={step}  {get_pos_string()}"
            print(f"[app4] {status_msg}")

        elif key == ord('s') or key == ord('S'):
            loteria_bot_controller.jog('Y', +step)
            status_msg = f"↓ S  step={step}  {get_pos_string()}"
            print(f"[app4] {status_msg}")

        elif key == ord('a') or key == ord('A'):
            loteria_bot_controller.jog('X', -step)
            status_msg = f"← A  step={step}  {get_pos_string()}"
            print(f"[app4] {status_msg}")

        elif key == ord('d') or key == ord('D'):
            loteria_bot_controller.jog('X', +step)
            status_msg = f"→ D  step={step}  {get_pos_string()}"
            print(f"[app4] {status_msg}")

        # Drop bean
        elif key == ord(' '):
            print("[app4] 👇 Dropping bean...")
            loteria_bot_controller.send_gcode("M3 S90")
            time.sleep(0.5)
            loteria_bot_controller.send_gcode("M3 S0")
            status_msg = f"Bean dropped!  {get_pos_string()}"
            print(f"[app4] {status_msg}")

        # Origin / park
        elif key == ord('o') or key == ord('O'):
            print("[app4] 🅿️  Going to origin...")
            loteria_bot_controller.send_gcode("G90")
            loteria_bot_controller.send_gcode(f"G1 Y{loteria_bot_controller.ORIGIN_Y} F{loteria_bot_controller.SPEED_Y}")
            loteria_bot_controller.send_gcode(f"G1 X{loteria_bot_controller.ORIGIN_X} F{loteria_bot_controller.SPEED_X}")
            status_msg = f"At origin  {get_pos_string()}"
            print(f"[app4] {status_msg}")

        # Zero current position
        elif key == ord('z') or key == ord('Z'):
            loteria_bot_controller.send_gcode("G10 L20 P1 X0 Y0 Z0")
            status_msg = "Zeroed — current position is now (0, 0)"
            print(f"[app4] ✅ {status_msg}")

        # Print position
        elif key == ord('p') or key == ord('P'):
            pos_str = get_pos_string()
            status_msg = f"Position: {pos_str}"
            print(f"[app4] 📍 {status_msg}")

        # Step size up
        elif key == ord('+') or key == ord('='):
            jog_idx = min(jog_idx + 1, len(JOG_STEPS) - 1)
            status_msg = f"Step → {JOG_LABELS[jog_idx]}"
            print(f"[app4] {status_msg}")

        # Step size down
        elif key == ord('-'):
            jog_idx = max(jog_idx - 1, 0)
            status_msg = f"Step → {JOG_LABELS[jog_idx]}"
            print(f"[app4] {status_msg}")

        # Quit
        elif key == ord('q') or key == ord('Q') or key == 27:
            print("[app4] Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[app4] Done.")


if __name__ == "__main__":
    main()