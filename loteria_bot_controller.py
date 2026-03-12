import serial
import time

# 1. Setup Serial Connection to Arduino
# Usually /dev/ttyACM0 or /dev/ttyUSB0 on Raspberry Pi
grbl = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

# 2. Calibration Constants (You must measure these!)
RATIO = 0.25  # mm per pixel
X_OFFSET = 10 # mm offset from gantry 0 to card corner
Y_OFFSET = 10 

def send_gcode(cmd):
    """
    Sends a G-code command to the GRBL controller via serial
    and waits for the 'ok' response.
    """
    grbl.write((cmd + '\n').encode())
    while True:
        line = grbl.readline().decode().strip()
        if 'ok' in line.lower():
            break

def drop_bean(pixel_x, pixel_y):
    """
    Converts YOLO pixel coordinates to physical mm coordinates,
    moves the gantry to the location, drops the bean using the servo,
    and parks the gantry back out of the camera's view.
    """
    # Convert pixels to physical mm
    target_x = (pixel_x * RATIO) + X_OFFSET
    target_y = (pixel_y * RATIO) + Y_OFFSET
    
    print(f"Match found at pixel ({pixel_x}, {pixel_y})! Moving gantry to X:{target_x} Y:{target_y}")
    
    # Move to the square
    send_gcode(f"G0 X{target_x} Y{target_y}")
    
    # Activate Servo (Using M3 command if using GRBL)
    send_gcode("M3 S90") # Drop position
    time.sleep(0.5)
    send_gcode("M3 S0")  # Reset position
    
    # MOVE TO PARK (Out of camera view)
    print("Parking gantry...")
    send_gcode("G0 X-50 Y-50") 

# --- HOW TO USE IN YOUR MAIN YOLO LOOP ---
#
# # Assuming you have a match found in your YOLO detection loop
# if match_found:
#     # Extract center coordinates from YOLO bounding box results
#     # This assumes a format like YOLOv8 results.xywh
#     center_x = results.xywh[0][0] 
#     center_y = results.xywh[0][1]
#     
#     # Call the dropping function
#     drop_bean(center_x, center_y)
