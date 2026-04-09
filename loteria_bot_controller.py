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

# Movement Speeds (mm/min). Using G1 instead of G0 allows speed control.
# X is heavy, so we limit the feed rate to prevent motor stutter/skipping steps.
SPEED_X = 500 
SPEED_Y = 300

# Safe Origin / Parking Coordinates
ORIGIN_X = 0   # fully left
ORIGIN_Y = -5  # negative to park UP without hitting the top frame hard

# Hardware connections
# Auto-detect the USB serial port for the GRBL Arduino
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
                
                # ENFORCE CRITICAL STARTUP CONFIG
                s.write(b"G21\n") # Force Millimeters (Prevents inch-scaling bugs)
                time.sleep(0.1)
                
                print(f"[SETUP] -> Successfully connected to GRBL on {port.device}!")
                return s
            except Exception as e:
                print(f"[SETUP] -> Failed to connect on {port.device}: {e}")
    
    print("[SETUP] -> No suitable GRBL device found. Will run in SIMULATED mode.")
    return None

grbl = connect_grbl()
robot_lock = threading.Lock() # Ensures only one command accesses the robot at a time

# Calibration file path
CALIBRATION_FILE = "homography_matrix.json"

# ==============================================================================
# 2. HOMOGRAPHY CALIBRATION (PIXEL TO MM)
# ==============================================================================

def calibrate_homography():
    """
    Run this function ONCE to map the camera pixels to the physical board.
    You will need to manually jog the CNC to 4 corners and find their pixels.
    """
    print("--- HOMOGRAPHY CALIBRATION ---")
    print("We need to map 4 points from the Camera (Pixels) to the Gantry (Millimeters).")
    
    # 1. Physical Coordinates (Where the CNC actually is in mm)
    # Example: A 150mm x 240mm Loteria Board
    # You should measure exactly where these 4 points are on your machine.
    # Format: [X_mm, Y_mm]
    physical_pts = np.array([
        [0, 0],         # Top-Left corner of board
        [150, 0],       # Top-Right corner
        [150, 240],     # Bottom-Right corner
        [0, 240]        # Bottom-Left corner
    ], dtype=np.float32)

    # 2. Pixel Coordinates (Camera click calibration)
    print("\nOpening camera for calibration...")
    cam_choice = input("Press ENTER to use Camera 0, or type '1' for Camera 1: ")
    cam_id = 1 if cam_choice.strip() == '1' else 0
    cap = cv2.VideoCapture(cam_id)
    
    if not cap.isOpened():
        print("❌ Could not open camera. Check connection.")
        return None
        
    pixel_pts = []
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(pixel_pts) < 4:
                pixel_pts.append([x, y])
                print(f"🎯 Captured point {len(pixel_pts)}/4 at ({x}, {y})")

    win_name = 'Calibration - Click the 4 corners'
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, click_event)
    
    print("\n===== INSTRUCTIONS =====")
    print("Click the 4 corners of your game board on the video feed.")
    print("Do it in this EXACT order:")
    print("  1. Top-Left corner (0,0)")
    print("  2. Top-Right corner")
    print("  3. Bottom-Right corner")
    print("  4. Bottom-Left corner")
    print("Press 'q' to cancel.")
    print("========================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        for idx, pt in enumerate(pixel_pts):
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 6, (0, 0, 255), -1)
            cv2.putText(frame, str(idx + 1), (int(pt[0]) + 10, int(pt[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        
        cv2.imshow(win_name, frame)
        
        if len(pixel_pts) == 4:
            print("\nAll 4 points captured, calculating matrix...")
            cv2.waitKey(1000)
            break
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    if len(pixel_pts) != 4:
        print("❌ Calibration cancelled.")
        return None

    pixel_pts_arr = np.array(pixel_pts, dtype=np.float32)
    matrix, _ = cv2.findHomography(pixel_pts_arr, physical_pts)

    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(matrix.tolist(), f)
    
    print(f"✅ Calibration successful! Saved to {CALIBRATION_FILE}")
    return matrix


def load_calibration():
    """
    Loads the homography matrix from the JSON file.
    """
    if not os.path.exists(CALIBRATION_FILE):
        print("Calibration file not found. You must run calibrate_homography() first!")
        return None
        
    with open(CALIBRATION_FILE, 'r') as f:
        matrix_list = json.load(f)
        return np.array(matrix_list, dtype=np.float32)

# ==============================================================================
# 3. GANTRY CONTROL & MATH
# ==============================================================================

def send_gcode(cmd):
    """
    Sends a G-code command to the GRBL controller via serial
    and waits for the 'ok' response.
    """
    print(f"\n[CNC] --> Sending: {cmd}")
    if grbl is None:
        print(f"[CNC] <-- [SIMULATED SUCCESS, waiting for 'ok']")
        time.sleep(0.5) # Fake movement delay
        return
        
    try:
        grbl.write((cmd + '\n').encode())
        timeout_counter = 0
        while True:
            line = grbl.readline().decode('utf-8').strip()
            if line:
                print(f"[CNC] <-- Received: {line}")
                timeout_counter = 0 # reset on valid data
                if 'ok' in line.lower():
                    break
                if 'error' in line.lower():
                    print(f"[CNC] !!! GRBL EXCEPTION CAUGHT !!!")
                    break
            else:
                timeout_counter += 1
                if timeout_counter >= 3: # 3 empty reads = 3 seconds hung/frozen
                    print("\n" + "!"*60)
                    print("[CNC] ⚠️ CRITICAL: Arduino stopped responding to serial commands!")
                    print("-> Why? The Arduino received a command but never said 'ok'.")
                    print("-> Fix: Press the physical 'RESET' button on the board or unplug the USB.")
                    print("!"*60 + "\n")
                    break
    except KeyboardInterrupt:
        print("\n\n[CNC] 🛑 EMERGENCY STOP DETECTED! (Ctrl+C pressed) 🛑")
        grbl.write(b'!') # GRBL Feed Hold (Stop instantly)
        time.sleep(0.1)
        grbl.write(b'\x18') # GRBL Soft Reset (Clear memory)
        print("[CNC] 🛑 Sent INSTANT HALT command to motors! Shutting down script...\n")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"[CNC] !!! SERIAL RUNTIME ERROR: {e}")


def pixel_to_mm(pixel_x, pixel_y, transform_matrix):
    """
    Uses the homography matrix to convert a camera pixel to physical mm.
    """
    # math magic requires a 3D vector for perspective transform
    # Convert point to homogeneous coordinates [x, y, 1]
    point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    
    # Apply transformation
    transformed = cv2.perspectiveTransform(point, transform_matrix)
    
    # Extract physical X and Y
    target_x = transformed[0][0][0]
    target_y = transformed[0][0][1]
    
    return target_x, target_y


def drop_bean(pixel_x, pixel_y):
    """
    Reads the calibration, calculates the exact mm, and drops the bean.
    """
    with robot_lock:
        print(f"\n[{time.strftime('%H:%M:%S')}] *** INITIATING ROBOT DROP SEQUENCE ***")
        matrix = load_calibration()
        if matrix is None:
            return

    # 1. Do the Math
    target_x, target_y = pixel_to_mm(pixel_x, pixel_y, matrix)
    
    # Round to 2 decimal places for GCODE
    target_x = round(target_x, 2)
    target_y = round(target_y, 2)
    
    print(f"\nMatch found at pixel ({pixel_x}, {pixel_y})!")
    print(f"Transformed to Physical Board: X:{target_x}mm, Y:{target_y}mm")
    
    # 2. Add safety limits (prevent crashing the CNC)
    # Accounting for the physical size of the carriage on each rail
    # so the motor doesn't crash into the end.
    X_RAIL_LENGTH = 300
    X_CARRIAGE_WIDTH = 70
    MAX_X = X_RAIL_LENGTH - X_CARRIAGE_WIDTH  # ~230 mm true usable travel
    
    Y_RAIL_LENGTH = 200
    Y_CARRIAGE_WIDTH = 107
    MAX_Y = Y_RAIL_LENGTH - Y_CARRIAGE_WIDTH  # ~93 mm true usable travel
    
    if target_x < 0 or target_x > MAX_X or target_y < 0 or target_y > MAX_Y:
         print(f"ERROR: Safety limit reached. Coordinate ({target_x}, {target_y}) is out of bounds.")
         print(f"-> Allowed ranges: X (0 to {MAX_X}), Y (0 to {MAX_Y})")
         return

    # 3. Move the CNC (Sequentially)
    print("\n=======================================================")
    print(f"[CNC] 🚗 MOVING SEQUENTIALLY TO X-Axis: {target_x} mm, then Y-Axis: {target_y} mm...")
    print("=======================================================")
    send_gcode("G90") # Ensure Absolute Mode before executing coordinates
    send_gcode(f"G1 X{target_x} F{SPEED_X}") # Move X axis first
    send_gcode(f"G1 Y{target_y} F{SPEED_Y}") # Then move Y axis slower
    
    # 4. Activate Servo (Using M3 Spindle command in GRBL)
    print(f"[CNC] 👇 DROPPING BEAN AT ({target_x}, {target_y})...")
    send_gcode("M3 S90") # Servo Drop
    time.sleep(0.5)
    send_gcode("M3 S0")  # Servo Reset
    
    # 5. MOVE TO PARK (Origin)
    print(f"Parking gantry at safe origin (X:{ORIGIN_X}, Y:{ORIGIN_Y})...")
    send_gcode("G90") # Ensure Absolute Mode for parking
    send_gcode(f"G1 Y{ORIGIN_Y} F{SPEED_Y}") # Park Y axis first
    send_gcode(f"G1 X{ORIGIN_X} F{SPEED_X}") # Park X axis second 


if __name__ == "__main__":
    # If you run this script directly, trigger the calibration wizard
    print("Loteria CNC Bot Controller")
    print("1. Run Calibration")
    print("2. Test Drop Bean")
    print("3. Manual Motor Jogging (Test Individual Axis)")
    choice = input("Select an option: ")
    
    if choice == "1":
        calibrate_homography()
    elif choice == "2":
        print("-------------------------------------------")
        print("Test a bean drop manually WITHOUT the YOLO active.")
        print("You type exactly which camera pixel (X, Y) to navigate to.")
        try:
            x = float(input("Enter test X pixel (e.g. 300): "))
            y = float(input("Enter test Y pixel (e.g. 200): "))
            drop_bean(x, y)
        except ValueError:
            print("Please enter valid numbers.")
    elif choice == "3":
        print("-------------------------------------------")
        print("MANUAL JOG MODE")
        print("Nudge your motors safely by small measurements.")
        print("Type an Axis and a Millimeter value (e.g., 'X 10', 'Y -5')")
        print("  -> X Positive (+): Moves LEFT (towards the X motor)")
        print("  -> Y Positive (+): Moves DOWN (away from the Y motor)")
        print("Type 'SERVO' to drop a bean.")
        print("Type 'ORIGIN' to return the gantry to the safe parking origin.")
        print("Type 'ZERO' to set the current position as the new absolute (0,0) origin.")
        print("Type 'POS' to print the current physical coordinates of the machine.")
        print("Type 'SETTINGS' to see current GRBL hardware config (steps/mm, etc).")
        print("To change a setting, type it directly (e.g., '$100=80').")
        print("Type 'q' to quit.")
        print("-------------------------------------------")
        
        print(f"[{time.strftime('%H:%M:%S')}] [DEBUG] Setting GRBL to RELATIVE positioning mode (G91)...")
        try:
            send_gcode("G91") # Set GRBL to RELATIVE positioning mode
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] [ERROR] Failed to send G91: {e}")
            
        while True:
            cmd = input("Jog Command -> ").strip().upper()
            if cmd == 'Q':
                print(f"[{time.strftime('%H:%M:%S')}] [DEBUG] Quitting manual jog mode. Restoring Absolute Mode (G90)...")
                try:
                    send_gcode("G90") # Restore GRBL back to Absolute Mode
                except Exception as e:
                    print(f"[{time.strftime('%H:%M:%S')}] [ERROR] Failed to send G90: {e}")
                break
            elif cmd == 'SERVO':
                print(f"[{time.strftime('%H:%M:%S')}] [DEBUG] Initiating SERVO drop sequence...")
                try:
                    send_gcode("M3 S90")
                    time.sleep(0.5)
                    send_gcode("M3 S0")
                    print(f"[{time.strftime('%H:%M:%S')}] [DEBUG] SERVO sequence complete.")
                except Exception as e:
                    print(f"[{time.strftime('%H:%M:%S')}] [ERROR] Servo drop failed: {e}")
            elif cmd == 'ORIGIN':
                print(f"[{time.strftime('%H:%M:%S')}] [DEBUG] Testing parking to safe origin (X:{ORIGIN_X}, Y:{ORIGIN_Y})...")
                try:
                    send_gcode("G90") # Switch temporarily to Absolute Mode
                    send_gcode(f"G1 Y{ORIGIN_Y} F{SPEED_Y}") # Move Y safely
                    send_gcode(f"G1 X{ORIGIN_X} F{SPEED_X}") # Move X safely
                    send_gcode("G91") # Switch back to Relative positioning for jogging
                    print(f"[{time.strftime('%H:%M:%S')}] [DEBUG] Origin sequence complete.")
                except Exception as e:
                    print(f"[{time.strftime('%H:%M:%S')}] [ERROR] Origin sequence failed: {e}")
            elif cmd == 'ZERO':
                print(f"[{time.strftime('%H:%M:%S')}] [DEBUG] Zeroing machine coordinates (G10 L20 P1 X0 Y0 Z0)...")
                try:
                    send_gcode("G10 L20 P1 X0 Y0 Z0")
                    print(f"[{time.strftime('%H:%M:%S')}] [DEBUG] Current position successfully saved to EEPROM as the absolute (0,0) origin.")
                except Exception as e:
                    print(f"[{time.strftime('%H:%M:%S')}] [ERROR] Failed to zero coordinates: {e}")
            elif cmd == 'POS':
                print(f"[{time.strftime('%H:%M:%S')}] [DEBUG] Requesting machine position...")
                if grbl is None:
                    print("--> SIMULATED: At X:0.0 Y:0.0")
                else:
                    try:
                        grbl.write(b"?")
                        time.sleep(0.1)
                        while grbl.in_waiting > 0:
                            line = grbl.readline().decode('utf-8').strip()
                            if line:
                                print(f"[{time.strftime('%H:%M:%S')}] [POSITION] {line}")
                    except Exception as e:
                        print(f"[{time.strftime('%H:%M:%S')}] [ERROR] Failed to get position: {e}")
            elif cmd == 'SETTINGS':
                print(f"[{time.strftime('%H:%M:%S')}] [DEBUG] Requesting GRBL settings ($$)...")
                if grbl is None:
                    print("--> SIMULATED: $100=250.000, $101=250.000, etc.")
                else:
                    try:
                        grbl.write(b"$$\n")
                        time.sleep(0.5)
                        while grbl.in_waiting > 0:
                            line = grbl.readline().decode('utf-8').strip()
                            if line:
                                print(f"[GRBL CONFIG] {line}")
                    except Exception as e:
                        print(f"[{time.strftime('%H:%M:%S')}] [ERROR] Failed to get settings: {e}")
            elif cmd.startswith('$'):
                print(f"[{time.strftime('%H:%M:%S')}] [DEBUG] Sending RAW config command: {cmd}")
                if grbl:
                    send_gcode(cmd)
                else:
                    print("--> SIMULATED: Config command accepted.")
            elif cmd:
                try:
                    cmd_clean = cmd.replace(" ", "")
                    axis = cmd_clean[0]
                    val = float(cmd_clean[1:])
                    if axis in ['X', 'Y', 'Z']:
                        feed_rate = SPEED_Y if axis == 'Y' else SPEED_X
                        gcode_cmd = f"G1 {axis}{val} F{feed_rate}"
                        print(f"[{time.strftime('%H:%M:%S')}] [DEBUG] Preparing to move {axis}-Axis by {val}mm...")
                        print(f"[{time.strftime('%H:%M:%S')}] [DEBUG] Sending command: {gcode_cmd}")
                        try:
                            send_gcode(gcode_cmd)
                            print(f"[{time.strftime('%H:%M:%S')}] [DEBUG] Post-move check complete for {gcode_cmd}.")
                        except Exception as e:
                            print(f"[{time.strftime('%H:%M:%S')}] [ERROR] Failed to execute move command {gcode_cmd}: {e}")
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] [WARNING] Unrecognized axis '{axis}'. Please start with X, Y, or Z.")
                except Exception as e:
                    print(f"[{time.strftime('%H:%M:%S')}] [ERROR] Exception parsing input '{cmd}': {e}")
                    print("Invalid input! Try something like: X 10 or X -10")
