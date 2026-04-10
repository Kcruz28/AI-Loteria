import serial
import time
import cv2
import numpy as np
import os
import json
import threading
import serial.tools.list_ports

# ==============================================================================
# 1. SETUP & CONFIGURATION
# ==============================================================================

SPEED_X = 100
SPEED_Y = 100

# Parking position — top-right corner, where gantry starts
ORIGIN_X = 0
ORIGIN_Y = 0

# Camera index — must match app2.py
CAMERA_INDEX = 8

# Machine travel limits in GRBL units
# Origin (0, 0) = top-right corner of board (where gantry parks)
# X goes negative to reach the left side of the board
# Y goes positive to reach the bottom of the board
MACHINE_MIN_X = -6.0
MACHINE_MAX_X =  0.0
MACHINE_MIN_Y =  0.0
MACHINE_MAX_Y =  3.0

# Calibration file
CALIBRATION_FILE = "calibration.json"

# ==============================================================================
# GRBL CONNECTION
# ==============================================================================

def connect_grbl():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if 'USB' in port.device or 'ACM' in port.device or 'usbserial' in port.device or 'usbmodem' in port.device:
            try:
                print(f"[SETUP] Attempting to connect to GRBL on {port.device}...")
                s = serial.Serial(port.device, 115200, timeout=1)
                s.write(b"\r\n\r\n")
                time.sleep(2)
                s.flushInput()
                s.write(b"G21\n")  # Force millimeters
                time.sleep(0.1)
                print(f"[SETUP] -> Successfully connected to GRBL on {port.device}!")
                return s
            except Exception as e:
                print(f"[SETUP] -> Failed on {port.device}: {e}")
    print("[SETUP] -> No GRBL device found. Running in SIMULATED mode.")
    return None

grbl = connect_grbl()
robot_lock = threading.Lock()


def emergency_stop():
    global grbl
    print("\n\n[CNC] 🛑 TRIGGERING EMERGENCY STOP TO MOTORS! 🛑")
    if grbl is not None:
        try:
            grbl.write(b'!')
            time.sleep(0.1)
            grbl.write(b'\x18')
            print("[CNC] 🛑 Robot halted.")
        except Exception as e:
            print(f"Failed to send stop command: {e}")


# ==============================================================================
# 2. CALIBRATION
# ==============================================================================
#
# The board corners map to machine coordinates like this:
#
#   Camera view (what you see overhead):
#
#   Top-Left -------- Top-Right
#      |                  |
#      |   LOTERIA BOARD  |
#      |                  |
#   Bot-Left -------- Bot-Right
#
#   Machine coordinates:
#   Top-Left     = X=-6, Y=0   (far left,  top)
#   Top-Right    = X= 0, Y=0   (origin,    top)    ← gantry parks here
#   Bottom-Right = X= 0, Y=3   (origin,    bottom)
#   Bottom-Left  = X=-6, Y=3   (far left,  bottom)
#
# During calibration, click the corners ON CAMERA in this exact order:
#   1. Top-Left
#   2. Top-Right
#   3. Bottom-Right
#   4. Bottom-Left
#
# ==============================================================================

def calibrate():
    print("\n===== CALIBRATION =====")
    print(f"Opening camera {CAMERA_INDEX}...")
    print("")
    print("Click the 4 corners of the Loteria board in this order:")
    print("  1. TOP-LEFT     (machine: X=-6, Y=0)")
    print("  2. TOP-RIGHT    (machine: X= 0, Y=0)  ← where gantry parks")
    print("  3. BOTTOM-RIGHT (machine: X= 0, Y=3)")
    print("  4. BOTTOM-LEFT  (machine: X=-6, Y=3)")
    print("")
    print("Press 'q' to cancel.\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"❌ Could not open camera {CAMERA_INDEX}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    pixel_pts = []
    labels = [
        "1: Top-Left     (X=-6, Y=0)",
        "2: Top-Right    (X= 0, Y=0)",
        "3: Bottom-Right (X= 0, Y=3)",
        "4: Bottom-Left  (X=-6, Y=3)",
    ]
    short_labels = ["1:TL", "2:TR", "3:BR", "4:BL"]

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN and len(pixel_pts) < 4:
            pixel_pts.append([x, y])
            print(f"  ✅ Point {len(pixel_pts)}/4 — {labels[len(pixel_pts)-1]} — pixel ({x}, {y})")

    win = "Calibration — click 4 corners in order"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, click_event)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for i, pt in enumerate(pixel_pts):
            cv2.circle(frame, (pt[0], pt[1]), 8, (0, 0, 255), -1)
            cv2.putText(frame, short_labels[i], (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if len(pixel_pts) < 4:
            instruction = f"Click: {labels[len(pixel_pts)]}"
            cv2.putText(frame, instruction, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "All 4 points captured! Saving...",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(win, frame)

        if len(pixel_pts) == 4:
            cv2.waitKey(1500)
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(pixel_pts) != 4:
        print("❌ Calibration cancelled.")
        return

    data = {"corners": pixel_pts}
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ Calibration saved to {CALIBRATION_FILE}")
    print(f"   1. Top-Left     pixel: {pixel_pts[0]}  → machine (-6.0,  0.0)")
    print(f"   2. Top-Right    pixel: {pixel_pts[1]}  → machine ( 0.0,  0.0)")
    print(f"   3. Bottom-Right pixel: {pixel_pts[2]}  → machine ( 0.0,  3.0)")
    print(f"   4. Bottom-Left  pixel: {pixel_pts[3]}  → machine (-6.0,  3.0)")


def load_calibration():
    if not os.path.exists(CALIBRATION_FILE):
        print(f"❌ '{CALIBRATION_FILE}' not found. Run option 1 (Calibration) first.")
        return None
    with open(CALIBRATION_FILE, 'r') as f:
        data = json.load(f)
    return data["corners"]


# ==============================================================================
# 3. PIXEL → MACHINE COORDINATE (bilinear interpolation)
# ==============================================================================

def pixel_to_machine(px, py, corners):
    """
    Converts a camera pixel (px, py) to machine coordinates (mx, my)
    using bilinear interpolation across the 4 calibrated corners.

    corners order: [top_left, top_right, bottom_right, bottom_left]

    Machine coords:
      top_left     → (-6,  0)
      top_right    → ( 0,  0)
      bottom_right → ( 0,  3)
      bottom_left  → (-6,  3)
    """
    tl = np.array(corners[0], dtype=np.float32)
    tr = np.array(corners[1], dtype=np.float32)
    br = np.array(corners[2], dtype=np.float32)
    bl = np.array(corners[3], dtype=np.float32)

    # Compute v (0=top, 1=bottom) from left and right edge Y spans
    left_span  = bl[1] - tl[1]
    right_span = br[1] - tr[1]

    v_left  = (py - tl[1]) / left_span  if left_span  > 0 else 0.5
    v_right = (py - tr[1]) / right_span if right_span > 0 else 0.5
    v = (v_left + v_right) / 2.0
    v = max(0.0, min(1.0, v))  # clamp

    # Interpolate left and right X boundaries at this v
    left_x  = tl[0] + v * (bl[0] - tl[0])
    right_x = tr[0] + v * (br[0] - tr[0])

    # Compute u (0=left, 1=right)
    x_span = right_x - left_x
    u = (px - left_x) / x_span if x_span > 0 else 0.5
    u = max(0.0, min(1.0, u))  # clamp

    # Map u, v → machine coordinates
    mx = MACHINE_MIN_X + u * (MACHINE_MAX_X - MACHINE_MIN_X)
    my = MACHINE_MIN_Y + v * (MACHINE_MAX_Y - MACHINE_MIN_Y)

    return round(mx, 2), round(my, 2)


# ==============================================================================
# 4. GCODE SENDER
# ==============================================================================

def send_gcode(cmd):
    print(f"[CNC] --> {cmd}")
    if grbl is None:
        print(f"[CNC] <-- [SIMULATED ok]")
        time.sleep(0.3)
        return

    try:
        grbl.write((cmd + '\n').encode())
        timeout_counter = 0
        while True:
            line = grbl.readline().decode('utf-8').strip()
            if line:
                print(f"[CNC] <-- {line}")
                timeout_counter = 0
                if 'ok' in line.lower():
                    break
                if 'error' in line.lower():
                    print(f"[CNC] !!! GRBL ERROR !!!")
                    break
            else:
                timeout_counter += 1
                if timeout_counter >= 3:
                    print("[CNC] ⚠️  Arduino stopped responding. Try resetting it.")
                    break
    except KeyboardInterrupt:
        print("\n[CNC] 🛑 EMERGENCY STOP (Ctrl+C)")
        grbl.write(b'!')
        time.sleep(0.1)
        grbl.write(b'\x18')
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"[CNC] Serial error: {e}")


# ==============================================================================
# 5. DROP BEAN — called by app2.py drop worker
# ==============================================================================

def drop_bean(pixel_x, pixel_y):
    with robot_lock:
        print(f"\n[CNC] *** DROP SEQUENCE — pixel ({pixel_x}, {pixel_y}) ***")

        corners = load_calibration()
        if corners is None:
            return

        mx, my = pixel_to_machine(pixel_x, pixel_y, corners)
        print(f"[CNC] Pixel ({pixel_x}, {pixel_y}) → Machine (X:{mx}, Y:{my})")

        # Safety bounds check
        if not (MACHINE_MIN_X <= mx <= MACHINE_MAX_X and MACHINE_MIN_Y <= my <= MACHINE_MAX_Y):
            print(f"[CNC] 🛑 OUT OF BOUNDS: ({mx}, {my}) — move blocked.")
            return

        # Move to target
        send_gcode("G90")                          # Absolute mode
        send_gcode(f"G1 X{mx} F{SPEED_X}")        # Move X first
        send_gcode(f"G1 Y{my} F{SPEED_Y}")        # Then Y
        send_gcode("G4 P1.5")                     # Dwell 1.5s to stabilize
        time.sleep(1.5)

        # Drop bean
        print(f"[CNC] 👇 Dropping bean at (X:{mx}, Y:{my})...")
        send_gcode("M3 S90")
        time.sleep(0.5)
        send_gcode("M3 S0")

        # Park back at origin
        print(f"[CNC] 🅿️  Parking at origin (0, 0)...")
        send_gcode("G90")
        send_gcode(f"G1 Y{ORIGIN_Y} F{SPEED_Y}")  # Y first to avoid collision
        send_gcode(f"G1 X{ORIGIN_X} F{SPEED_X}")  # Then X
        print(f"[CNC] ✅ Drop sequence complete.")


# ==============================================================================
# 6. MAIN CLI
# ==============================================================================

if __name__ == "__main__":
    print("\nLoteria CNC Bot Controller")
    print("1. Run Calibration")
    print("2. Test Drop Bean (by pixel)")
    print("3. Manual Motor Jogging")
    choice = input("Select an option: ").strip()

    if choice == "1":
        calibrate()

    elif choice == "2":
        print("Enter a pixel coordinate to test the full drop sequence.")
        try:
            x = float(input("X pixel (e.g. 320): "))
            y = float(input("Y pixel (e.g. 240): "))
            drop_bean(x, y)
        except ValueError:
            print("Invalid input.")

    elif choice == "3":
        print("\n-------------------------------------------")
        print("MANUAL JOG MODE")
        print("Commands:")
        print("  X-3      move X by -3 units")
        print("  Y1.5     move Y by 1.5 units")
        print("  SERVO    trigger bean drop")
        print("  ORIGIN   return to park position")
        print("  ZERO     set current position as (0,0)")
        print("  POS      print current machine position")
        print("  SETTINGS print GRBL config")
        print("  Q        quit")
        print("-------------------------------------------\n")

        send_gcode("G91")  # Relative mode for jogging

        while True:
            cmd = input("Jog -> ").strip().upper()

            if cmd == 'Q':
                send_gcode("G90")
                break

            elif cmd == 'SERVO':
                send_gcode("M3 S90")
                time.sleep(0.5)
                send_gcode("M3 S0")

            elif cmd == 'ORIGIN':
                send_gcode("G90")
                send_gcode(f"G1 Y{ORIGIN_Y} F{SPEED_Y}")
                send_gcode(f"G1 X{ORIGIN_X} F{SPEED_X}")
                send_gcode("G91")

            elif cmd == 'ZERO':
                send_gcode("G10 L20 P1 X0 Y0 Z0")
                print("✅ Current position saved as (0, 0).")

            elif cmd == 'POS':
                if grbl:
                    grbl.write(b"?")
                    time.sleep(0.15)
                    while grbl.in_waiting > 0:
                        line = grbl.readline().decode('utf-8').strip()
                        if line:
                            print(f"[POS] {line}")
                else:
                    print("SIMULATED: X:0.0 Y:0.0")

            elif cmd == 'SETTINGS':
                if grbl:
                    grbl.write(b"$$\n")
                    time.sleep(0.5)
                    while grbl.in_waiting > 0:
                        line = grbl.readline().decode('utf-8').strip()
                        if line:
                            print(f"[CFG] {line}")
                else:
                    print("SIMULATED settings.")

            elif cmd.startswith('$'):
                send_gcode(cmd)

            elif cmd:
                try:
                    cmd_clean = cmd.replace(" ", "")
                    axis = cmd_clean[0]
                    val  = float(cmd_clean[1:])
                    if axis in ['X', 'Y', 'Z']:
                        feed = SPEED_Y if axis == 'Y' else SPEED_X
                        send_gcode(f"G1 {axis}{val} F{feed}")
                    else:
                        print(f"Unknown axis '{axis}'. Use X, Y, or Z.")
                except Exception as e:
                    print(f"Bad command '{cmd}': {e}")