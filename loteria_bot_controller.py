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
    # Adjust these to your gantry's MAX physical size
    MAX_X = 200 
    MAX_Y = 300
    if target_x < 0 or target_x > MAX_X or target_y < 0 or target_y > MAX_Y:
         print("ERROR: Safety limit reached. Coordinate is out of bounds.")
         return

    # 3. Move the CNC
    print("\n=======================================================")
    print(f"[CNC] 🚗 MOVING MOTORS TO X-Axis: {target_x} mm, Y-Axis: {target_y} mm...")
    print("=======================================================")
    send_gcode(f"G0 X{target_x} Y{target_y}")
    
    # 4. Activate Servo (Using M3 Spindle command in GRBL)
    print(f"[CNC] 👇 DROPPING BEAN AT ({target_x}, {target_y})...")
    send_gcode("M3 S90") # Servo Drop
    time.sleep(0.5)
    send_gcode("M3 S0")  # Servo Reset
    
    # 5. MOVE TO PARK (Out of camera view)
    print("Parking gantry...")
    send_gcode("G0 X-50 Y-50") 


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
        print("Type an Axis and a Millimeter value (e.g., 'X 10' to go right, 'X -10' to go left)")
        print("Type 'SERVO' to drop a bean.")
        print("Type 'q' to quit.")
        print("-------------------------------------------")
        send_gcode("G91") # Set GRBL to RELATIVE positioning mode
        while True:
            cmd = input("Jog Command -> ").strip().upper()
            if cmd == 'Q':
                send_gcode("G90") # Restore GRBL back to Absolute Mode
                break
            elif cmd == 'SERVO':
                send_gcode("M3 S90")
                time.sleep(0.5)
                send_gcode("M3 S0")
            elif cmd:
                try:
                    parts = cmd.split()
                    axis = parts[0]
                    val = float(parts[1])
                    if axis in ['X', 'Y', 'Z']:
                        print(f"Moving {axis}-Axis by {val}mm...")
                        send_gcode(f"G0 {axis}{val}")
                    else:
                        print("Please start with X or Y.")
                except Exception:
                    print("Invalid input! Try something like: X 10")
