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

    # 2. Pixel Coordinates (Where the camera sees these points)
    # Use your YOLO camera feed or a test script to find these (X, Y) pixels.
    print("\nLook at your camera feed and input the (X, Y) pixel coordinates for each corner:")
    
    pixel_pts = []
    corners = ["Top-Left (0,0)", "Top-Right", "Bottom-Right", "Bottom-Left"]
    
    for corner in corners:
        x = float(input(f"Enter X pixel for {corner}: "))
        y = float(input(f"Enter Y pixel for {corner}: "))
        pixel_pts.append([x, y])
        
    pixel_pts = np.array(pixel_pts, dtype=np.float32)

    # 3. Calculate Transformation Matrix
    # This magic math figures out how to translate any pixel to any mm
    matrix, _ = cv2.findHomography(pixel_pts, physical_pts)

    # 4. Save to file so we don't have to calibrate every time
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(matrix.tolist(), f)
    
    print("\nCalibration successful! Matrix saved to " + CALIBRATION_FILE)
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
        while True:
            line = grbl.readline().decode('utf-8').strip()
            if line:
                print(f"[CNC] <-- Received: {line}")
            if 'ok' in line.lower():
                break
            if 'error' in line.lower():
                print(f"[CNC] !!! ERROR from GRBL controller !!!")
                break
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
    send_gcode(f"G0 X{target_x} Y{target_y}")
    
    # 4. Activate Servo (Using M3 Spindle command in GRBL)
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
    choice = input("Select an option: ")
    
    if choice == "1":
        calibrate_homography()
    elif choice == "2":
        x = float(input("Enter test X pixel: "))
        y = float(input("Enter test Y pixel: "))
        drop_bean(x, y)
