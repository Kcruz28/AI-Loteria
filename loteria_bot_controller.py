import serial
import time
import cv2
import numpy as np
import os
import json
import threading
import serial.tools.list_ports

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

SPEED_X = 100
SPEED_Y = 100

ORIGIN_X = 0
ORIGIN_Y = 0

CAMERA_INDEX = 8

# Machine travel limits in GRBL units
# (0,0) = top-right corner of board (gantry parks here)
# X goes negative toward the left side of the board
# Y goes positive toward the bottom of the board
MACHINE_MIN_X = -6.0
MACHINE_MAX_X =  0.0
MACHINE_MIN_Y =  0.0
MACHINE_MAX_Y =  3.0

CALIBRATION_FILE = "calibration.json"

# ==============================================================================
# 2. CARD GRID — class ID → (row, col) on the 4×4 Loteria board
#
#   Col:  0        1        2        3
# Row 0: Rosa(50) Calav(35) Mundo(17) Apache(1)
# Row 1: Pesc(23) Palma(47) Sol(25)   Corona(38)
# Row 2: Para(22) Siren(52) Gallo(14) Diabl(13)
# Row 3: Muer(46) Pera(48)  Arbol(2)  Melon(16)
#
# Machine coords for each cell center:
#   mx = MACHINE_MIN_X + (col + 0.5) * (|MACHINE_MAX_X - MACHINE_MIN_X| / 4)
#   my = MACHINE_MIN_Y + (row + 0.5) * (|MACHINE_MAX_Y - MACHINE_MIN_Y| / 4)
# ==============================================================================

CARD_GRID = {
    50: (0, 0),   # La Rosa
    35: (0, 1),   # La Calavera
    17: (0, 2),   # El Mundo
     1: (0, 3),   # El Apache
    23: (1, 0),   # El Pescado
    47: (1, 1),   # La Palma
    25: (1, 2),   # El Sol
    38: (1, 3),   # La Corona
    22: (2, 0),   # El Paraguas
    52: (2, 1),   # La Sirena
    14: (2, 2),   # El Gallo
    13: (2, 3),   # El Diablito
    46: (3, 0),   # La Muerte
    48: (3, 1),   # La Pera
     2: (3, 2),   # El Arbol
    16: (3, 3),   # El Melon
}

def class_to_machine(cls_id):
    """
    Converts a YOLO class ID directly to machine (X, Y) coordinates
    using the known 4x4 grid layout. No pixel math needed.
    Returns None if class not in grid.
    """
    if cls_id not in CARD_GRID:
        return None

    row, col = CARD_GRID[cls_id]

    x_range = abs(MACHINE_MAX_X - MACHINE_MIN_X)  # 6.0
    y_range = abs(MACHINE_MAX_Y - MACHINE_MIN_Y)  # 3.0

    mx = MACHINE_MIN_X + (col + 0.5) * (x_range / 4.0)
    my = MACHINE_MIN_Y + (row + 0.5) * (y_range / 4.0)

    return round(mx, 3), round(my, 3)


# ==============================================================================
# 3. GRBL CONNECTION
# ==============================================================================

def connect_grbl():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if any(k in port.device for k in ['USB', 'ACM', 'usbserial', 'usbmodem']):
            try:
                print(f"[SETUP] Connecting to GRBL on {port.device}...")
                s = serial.Serial(port.device, 115200, timeout=1)
                s.write(b"\r\n\r\n")
                time.sleep(2)
                s.flushInput()
                s.write(b"G21\n")  # Force millimeters
                time.sleep(0.1)
                print(f"[SETUP] -> Connected on {port.device}!")
                return s
            except Exception as e:
                print(f"[SETUP] -> Failed on {port.device}: {e}")
    print("[SETUP] -> No GRBL device found. Running in SIMULATED mode.")
    return None

grbl = connect_grbl()
robot_lock = threading.Lock()


def emergency_stop():
    global grbl
    print("\n[CNC] 🛑 EMERGENCY STOP")
    if grbl:
        try:
            grbl.write(b'!')
            time.sleep(0.1)
            grbl.write(b'\x18')
            print("[CNC] 🛑 Robot halted.")
        except Exception as e:
            print(f"[CNC] Stop failed: {e}")


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
                    print("[CNC] ⚠️  Arduino not responding.")
                    break
    except KeyboardInterrupt:
        print("\n[CNC] 🛑 Ctrl+C — stopping motors")
        grbl.write(b'!')
        time.sleep(0.1)
        grbl.write(b'\x18')
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"[CNC] Serial error: {e}")


# ==============================================================================
# 5. DROP BEAN — called by app2.py drop worker
#    Uses class ID directly → no pixel calibration needed
# ==============================================================================

def drop_bean(cls_id, pixel_x=None, pixel_y=None):
    """
    Moves gantry to the card's known grid position and drops a bean.

    Args:
        cls_id:  YOLO class ID (used for grid lookup — most accurate)
        pixel_x: fallback pixel X (only used if cls_id not in grid)
        pixel_y: fallback pixel Y (only used if cls_id not in grid)
    """
    with robot_lock:
        result = class_to_machine(cls_id)

        if result is not None:
            mx, my = result
            row, col = CARD_GRID[cls_id]
            print(f"\n[CNC] *** DROP — class {cls_id} → grid ({row},{col}) → machine (X:{mx}, Y:{my}) ***")
        else:
            print(f"\n[CNC] ⚠️  Class {cls_id} not in grid map. Skipping drop.")
            return

        # Safety bounds check
        if not (MACHINE_MIN_X <= mx <= MACHINE_MAX_X and MACHINE_MIN_Y <= my <= MACHINE_MAX_Y):
            print(f"[CNC] 🛑 OUT OF BOUNDS: ({mx}, {my}) — move blocked.")
            return

        # Move to target
        send_gcode("G90")
        send_gcode(f"G1 X{mx} F{SPEED_X}")
        send_gcode(f"G1 Y{my} F{SPEED_Y}")
        send_gcode("G4 P1.5")
        time.sleep(1.5)

        # Drop bean
        print(f"[CNC] 👇 Dropping bean...")
        send_gcode("M3 S90")
        time.sleep(0.5)
        send_gcode("M3 S0")

        # Park
        print(f"[CNC] 🅿️  Parking at origin...")
        send_gcode("G90")
        send_gcode(f"G1 Y{ORIGIN_Y} F{SPEED_Y}")
        send_gcode(f"G1 X{ORIGIN_X} F{SPEED_X}")
        print(f"[CNC] ✅ Done.")


# ==============================================================================
# 6. MAIN CLI
# ==============================================================================

if __name__ == "__main__":
    print("\nLoteria CNC Bot Controller")
    print("1. Test Drop Bean (by class ID)")
    print("2. Show grid — print all class → machine mappings")
    print("3. Manual Motor Jogging")
    choice = input("Select an option: ").strip()

    if choice == "1":
        try:
            cls = int(input("Enter class ID (e.g. 50 for La Rosa): "))
            drop_bean(cls)
        except ValueError:
            print("Invalid input.")

    elif choice == "2":
        print("\n{'Class':>6} | {'Name':20} | {'Grid':8} | Machine (X, Y)")
        print("-" * 60)
        names = {
            50:"La Rosa", 35:"La Calavera", 17:"El Mundo", 1:"El Apache",
            23:"El Pescado", 47:"La Palma", 25:"El Sol", 38:"La Corona",
            22:"El Paraguas", 52:"La Sirena", 14:"El Gallo", 13:"El Diablito",
            46:"La Muerte", 48:"La Pera", 2:"El Arbol", 16:"El Melon"
        }
        for cls_id, (row, col) in sorted(CARD_GRID.items(), key=lambda x: (x[1][0], x[1][1])):
            mx, my = class_to_machine(cls_id)
            print(f"  {cls_id:>4}   | {names.get(cls_id,'?'):20} | ({row},{col})   | X:{mx:+.3f}, Y:{my:.3f}")

    elif choice == "3":
        print("\n-------------------------------------------")
        print("MANUAL JOG MODE")
        print("Commands: X-3  Y1.5  SERVO  ORIGIN  ZERO  POS  SETTINGS  Q")
        print("-------------------------------------------\n")

        send_gcode("G91")  # Relative mode

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
                print("✅ Zeroed.")
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
            elif cmd.startswith('$'):
                send_gcode(cmd)
            elif cmd:
                try:
                    cmd_clean = cmd.replace(" ", "")
                    axis = cmd_clean[0]
                    val = float(cmd_clean[1:])
                    if axis in ['X', 'Y', 'Z']:
                        feed = SPEED_Y if axis == 'Y' else SPEED_X
                        send_gcode(f"G1 {axis}{val} F{feed}")
                    else:
                        print(f"Unknown axis '{axis}'.")
                except Exception as e:
                    print(f"Bad command: {e}")