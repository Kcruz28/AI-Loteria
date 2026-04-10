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

MACHINE_MIN_X = -6.0
MACHINE_MAX_X =  0.0
MACHINE_MIN_Y =  0.0
MACHINE_MAX_Y =  3.0

# --- SERVO SETTINGS ---
# Most GRBL servos use a 0-1000 scale ($30 setting).
# Standard servos want a pulse between S40 and S140.
SERVO_OPEN  = 140   # Max rotation for the drop
SERVO_CLOSE = 40    # Idle/closed position
# ----------------------

# Calibration file — stores confirmed measured positions per card
CALIBRATION_FILE = "calibration.json"

# Jog step sizes (GRBL units)
JOG_STEPS = [0.05, 0.1, 0.2]
JOG_STEP_LABELS = ["fine (0.05)", "medium (0.1)", "coarse (0.2)"]
DEFAULT_JOG_STEP_IDX = 1  # start at medium

# ==============================================================================
# 2. CARD GRID — class ID → (row, col)
#
#   Col:  0        1        2        3
# Row 0: Rosa(50) Calav(35) Mundo(17) Apache(1)
# Row 1: Pesc(23) Palma(47) Sol(25)   Corona(38)
# Row 2: Para(22) Siren(52) Gallo(14) Diabl(13)
# Row 3: Muer(46) Pera(48)  Arbol(2)  Melon(16)
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

CARD_NAMES = {
    50:"La Rosa", 35:"La Calavera", 17:"El Mundo", 1:"El Apache",
    23:"El Pescado", 47:"La Palma", 25:"El Sol", 38:"La Corona",
    22:"El Paraguas", 52:"La Sirena", 14:"El Gallo", 13:"El Diablito",
    46:"La Muerte", 48:"La Pera", 2:"El Arbol", 16:"El Melon"
}


def grid_to_machine(row, col):
    """Compute estimated machine position from grid row/col."""
    x_range = abs(MACHINE_MAX_X - MACHINE_MIN_X)
    y_range = abs(MACHINE_MAX_Y - MACHINE_MIN_Y)
    mx = MACHINE_MIN_X + (col + 0.5) * (x_range / 4.0)
    my = MACHINE_MIN_Y + (row + 0.5) * (y_range / 4.0)
    return round(mx, 3), round(my, 3)


def class_to_machine(cls_id, calibration_data=None):
    """
    Returns (mx, my, is_measured) for a class ID.
    Uses measured position if available, otherwise grid estimate.
    """
    if cls_id not in CARD_GRID:
        return None, None, False

    row, col = CARD_GRID[cls_id]

    # Use measured position if we have it
    if calibration_data and str(cls_id) in calibration_data:
        entry = calibration_data[str(cls_id)]
        return entry["mx"], entry["my"], True

    # Fall back to grid estimate
    mx, my = grid_to_machine(row, col)
    return mx, my, False


# ==============================================================================
# 3. CALIBRATION FILE — load/save measured positions
# ==============================================================================

def load_calibration_data():
    """Load measured card positions from file. Returns {} if not found."""
    if not os.path.exists(CALIBRATION_FILE):
        return {}
    with open(CALIBRATION_FILE, 'r') as f:
        return json.load(f)


def save_calibration_data(data):
    """Save measured card positions to file."""
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[CAL] 💾 Calibration saved ({len(data)} cards measured).")


def save_card_position(cls_id, mx, my):
    """Save a single confirmed card position."""
    data = load_calibration_data()
    data[str(cls_id)] = {
        "cls_id": cls_id,
        "name": CARD_NAMES.get(cls_id, f"class_{cls_id}"),
        "mx": round(mx, 3),
        "my": round(my, 3),
        "grid": CARD_GRID.get(cls_id, None)
    }
    save_calibration_data(data)
    return data


# ==============================================================================
# 4. GRBL CONNECTION
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
                s.write(b"G21\n")
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


def get_position():
    """Ask GRBL for current machine position. Returns (x, y) or None."""
    if grbl is None:
        return None
    try:
        grbl.write(b"?")
        time.sleep(0.15)
        response = ""
        while grbl.in_waiting > 0:
            response += grbl.readline().decode('utf-8').strip()

        # Parse MPos from response like: <Idle|MPos:-3.750,1.125,0.000|...>
        if 'MPos:' in response:
            mpos_str = response.split('MPos:')[1].split('|')[0]
            parts = mpos_str.split(',')
            mx = float(parts[0])
            my = float(parts[1])
            return round(mx, 3), round(my, 3)
    except Exception as e:
        print(f"[CNC] Failed to get position: {e}")
    return None


# ==============================================================================
# 5. GCODE SENDER
# ==============================================================================

def send_gcode(cmd, silent=False):
    if not silent:
        print(f"[CNC] --> {cmd}")
    if grbl is None:
        if not silent:
            print(f"[CNC] <-- [SIMULATED ok]")
        time.sleep(0.1)
        return

    try:
        grbl.write((cmd + '\n').encode())
        timeout_counter = 0
        while True:
            line = grbl.readline().decode('utf-8').strip()
            if line:
                if not silent:
                    print(f"[CNC] <-- {line}")
                timeout_counter = 0
                if 'ok' in line.lower():
                    break
                if 'error' in line.lower():
                    print(f"[CNC] !!! GRBL ERROR: {line}")
                    break
            else:
                timeout_counter += 1
                if timeout_counter >= 3:
                    print("[CNC] ⚠️  Arduino not responding.")
                    break
    except KeyboardInterrupt:
        print("\n[CNC] 🛑 Ctrl+C")
        grbl.write(b'!')
        time.sleep(0.1)
        grbl.write(b'\x18')
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"[CNC] Serial error: {e}")


def jog(axis, distance, silent=False):
    """Send a relative jog move on one axis."""
    send_gcode("G91", silent=True)
    feed = SPEED_Y if axis == 'Y' else SPEED_X
    send_gcode(f"G1 {axis}{distance:+.3f} F{feed}", silent=silent)
    send_gcode("G90", silent=True)


# ==============================================================================
# 6. LIVE CALIBRATION + DROP
#
# Called by app2.py when a match is detected.
# Moves robot to estimated position, then lets user fine-tune with WASD,
# confirm with SPACE (saves position), or skip with E (uses estimate only).
#
# Returns True if drop was executed, False if aborted.
# ==============================================================================

# Shared state for WASD jog commands from app2.py camera window
_jog_command   = None          # set by app2 key handler: 'W','A','S','D'
_jog_step_idx  = DEFAULT_JOG_STEP_IDX
_confirm_event = threading.Event()   # SPACE pressed
_skip_event    = threading.Event()   # E pressed
_jog_lock      = threading.Lock()


def set_jog_command(cmd):
    """Called from app2.py key handler to inject a jog direction."""
    global _jog_command
    with _jog_lock:
        _jog_command = cmd


def set_jog_step(idx):
    global _jog_step_idx
    _jog_step_idx = idx


def confirm_position():
    """Called from app2.py when SPACE is pressed."""
    _confirm_event.set()


def skip_calibration():
    """Called from app2.py when E is pressed."""
    _skip_event.set()


def drop_bean(cls_id, pixel_x=None, pixel_y=None):
    """
    Main drop sequence called by app2.py drop worker.

    1. Looks up measured or estimated machine position for cls_id
    2. Moves robot to that position
    3. Enters WASD jog mode — user fine-tunes position
    4. SPACE = confirm (saves position, drops bean)
       E     = skip (drops at current position without saving)
    5. Parks robot at origin
    """
    global _jog_command, _jog_step_idx

    with robot_lock:
        cal_data = load_calibration_data()
        mx, my, is_measured = class_to_machine(cls_id, cal_data)

        if mx is None:
            print(f"[CNC] ⚠️  Class {cls_id} not in grid. Skipping.")
            return

        name = CARD_NAMES.get(cls_id, f"class_{cls_id}")
        src  = "MEASURED" if is_measured else "ESTIMATED"
        print(f"\n[CNC] *** {name} (cls {cls_id}) — position from {src}: X:{mx}, Y:{my} ***")

        # Safety check
        if not (MACHINE_MIN_X <= mx <= MACHINE_MAX_X and MACHINE_MIN_Y <= my <= MACHINE_MAX_Y):
            print(f"[CNC] 🛑 OUT OF BOUNDS ({mx}, {my}). Skipping.")
            return

        # Move to estimated/measured position
        send_gcode("G90")
        send_gcode(f"G1 X{mx} F{SPEED_X}")
        send_gcode(f"G1 Y{my} F{SPEED_Y}")

        # Reset events and jog command
        _confirm_event.clear()
        _skip_event.clear()
        with _jog_lock:
            _jog_command = None

        print(f"\n[CAL] ══════════════════════════════════════════")
        print(f"[CAL]  Card: {name}")
        print(f"[CAL]  Use WASD in camera window to fine-tune.")
        print(f"[CAL]  +/- to change step size.")
        print(f"[CAL]  SPACE = confirm & save position")
        print(f"[CAL]  E     = skip, drop here without saving")
        print(f"[CAL] ══════════════════════════════════════════\n")

        # WASD jog loop — wait for SPACE or E
        while not _confirm_event.is_set() and not _skip_event.is_set():
            with _jog_lock:
                cmd = _jog_command
                _jog_command = None

            if cmd == 'W':
                step = JOG_STEPS[_jog_step_idx]
                jog('Y', -step, silent=True)
                pos = get_position()
                if pos:
                    print(f"[CAL] Jogged W → pos now: X:{pos[0]}, Y:{pos[1]}  (step={step})")
            elif cmd == 'S':
                step = JOG_STEPS[_jog_step_idx]
                jog('Y', +step, silent=True)
                pos = get_position()
                if pos:
                    print(f"[CAL] Jogged S → pos now: X:{pos[0]}, Y:{pos[1]}  (step={step})")
            elif cmd == 'A':
                step = JOG_STEPS[_jog_step_idx]
                jog('X', -step, silent=True)
                pos = get_position()
                if pos:
                    print(f"[CAL] Jogged A → pos now: X:{pos[0]}, Y:{pos[1]}  (step={step})")
            elif cmd == 'D':
                step = JOG_STEPS[_jog_step_idx]
                jog('X', +step, silent=True)
                pos = get_position()
                if pos:
                    print(f"[CAL] Jogged D → pos now: X:{pos[0]}, Y:{pos[1]}  (step={step})")

            time.sleep(0.05)

        # Read final confirmed position
        final_pos = get_position()

        if _confirm_event.is_set() and final_pos:
            fx, fy = final_pos
            save_card_position(cls_id, fx, fy)
            total = len(load_calibration_data())
            print(f"[CAL] ✅ Saved: {name} → X:{fx}, Y:{fy}  ({total} cards calibrated total)")
        else:
            print(f"[CAL] ⏭️  Skipped calibration for {name}. Dropping at current position.")

        # Drop bean
        send_gcode("G4 P1.0")
        time.sleep(1.0)
        print(f"[CNC] 👇 Dropping bean (S{SERVO_OPEN})...")
        send_gcode(f"M3 S{SERVO_OPEN}")
        time.sleep(0.8) # Wait for servo to reach full position
        send_gcode(f"M3 S{SERVO_CLOSE}")

        # Park
        print(f"[CNC] 🅿️  Parking...")
        send_gcode("G90")
        send_gcode(f"G1 Y{ORIGIN_Y} F{SPEED_Y}")
        send_gcode(f"G1 X{ORIGIN_X} F{SPEED_X}")
        print(f"[CNC] ✅ Done.\n")


# ==============================================================================
# 7. MAIN CLI
# ==============================================================================

if __name__ == "__main__":
    print("\nLoteria CNC Bot Controller")
    print("1. Test Drop Bean (by class ID)")
    print("2. Show all card → machine mappings")
    print("3. Manual Motor Jogging")
    print("4. Print calibration data")
    choice = input("Select an option: ").strip()

    if choice == "1":
        try:
            cls = int(input("Enter class ID (e.g. 50 for La Rosa): "))
            drop_bean(cls)
        except ValueError:
            print("Invalid input.")

    elif choice == "2":
        cal = load_calibration_data()
        print(f"\n  {'ID':>4} | {'Name':20} | {'Grid':6} | {'Estimate':16} | {'Measured':16}")
        print("-" * 75)
        for cls_id, (row, col) in sorted(CARD_GRID.items(), key=lambda x: (x[1][0], x[1][1])):
            est_x, est_y = grid_to_machine(row, col)
            measured = ""
            if str(cls_id) in cal:
                e = cal[str(cls_id)]
                measured = f"X:{e['mx']:+.3f}, Y:{e['my']:.3f}"
            print(f"  {cls_id:>4} | {CARD_NAMES.get(cls_id,'?'):20} | ({row},{col}) | X:{est_x:+.3f}, Y:{est_y:.3f} | {measured}")

    elif choice == "3":
        print("\n-------------------------------------------")
        print("MANUAL JOG MODE")
        print("Commands: X-3  Y1.5  SERVO  ORIGIN  ZERO  POS  SETTINGS  Q")
        print("-------------------------------------------\n")

        send_gcode("G91")

        while True:
            cmd = input("Jog -> ").strip().upper()
            if cmd == 'Q':
                send_gcode("G90")
                break
            elif cmd == 'SERVO':
                print(f"[CNC] Servo Test: Open (S{SERVO_OPEN}) then Close (S{SERVO_CLOSE})")
                send_gcode(f"M3 S{SERVO_OPEN}")
                time.sleep(0.8)
                send_gcode(f"M3 S{SERVO_CLOSE}")
            elif cmd == 'ORIGIN':
                send_gcode("G90")
                send_gcode(f"G1 Y{ORIGIN_Y} F{SPEED_Y}")
                send_gcode(f"G1 X{ORIGIN_X} F{SPEED_X}")
                send_gcode("G91")
            elif cmd == 'ZERO':
                send_gcode("G10 L20 P1 X0 Y0 Z0")
                print("✅ Zeroed.")
            elif cmd == 'POS':
                pos = get_position()
                if pos:
                    print(f"[POS] X:{pos[0]}, Y:{pos[1]}")
                elif grbl:
                    grbl.write(b"?")
                    time.sleep(0.15)
                    while grbl.in_waiting > 0:
                        line = grbl.readline().decode('utf-8').strip()
                        if line:
                            print(f"[POS] {line}")
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
                    val  = float(cmd_clean[1:])
                    if axis in ['X', 'Y', 'Z']:
                        feed = SPEED_Y if axis == 'Y' else SPEED_X
                        send_gcode(f"G1 {axis}{val} F{feed}")
                    else:
                        print(f"Unknown axis '{axis}'.")
                except Exception as e:
                    print(f"Bad command: {e}")

    elif choice == "4":
        cal = load_calibration_data()
        if not cal:
            print("No calibration data yet.")
        else:
            print(f"\n{len(cal)} cards measured:")
            for cls_id, entry in cal.items():
                print(f"  Class {cls_id:>3} | {entry['name']:20} | X:{entry['mx']:+.3f}, Y:{entry['my']:.3f}")