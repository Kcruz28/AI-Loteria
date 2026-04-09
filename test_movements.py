import time
import loteria_bot_controller as lbc

print("====================================")
print("🤖 AUTOMATED JOG TEST SCRIPT 🤖")
print("====================================")

try:
    # Switch to RELATIVE mode so commands move it by X/Y amounts from its *current* position
    print("\nSetting GRBL to RELATIVE positioning mode (G91)...")
    lbc.send_gcode("G91")
    
    cycle = 1
    while True:
        print(f"\n--- Starting Demo Cycle #{cycle} ---")
        
        # 1. Move X positive by 5
        print("--> Moving X axis positive (+5)...")
        lbc.send_gcode(f"G1 X5 F{lbc.SPEED_X}")
        time.sleep(2)  # Wait for movement to finish
        
        # 2. Move X negative by 5
        print("--> Moving X axis negative (-5)...")
        lbc.send_gcode(f"G1 X-5 F{lbc.SPEED_X}")
        time.sleep(2)

        # 3. Move Y positive by 2
        print("--> Moving Y axis positive (+2)...")
        lbc.send_gcode(f"G1 Y2 F{lbc.SPEED_Y}")
        time.sleep(2)

        # 4. Move Y negative by 2
        print("--> Moving Y axis negative (-2)...")
        lbc.send_gcode(f"G1 Y-2 F{lbc.SPEED_Y}")
        time.sleep(2)
        
        cycle += 1

except KeyboardInterrupt:
    print("\n\n🛑 Demo stopped by user (Ctrl+C). Triggering emergency stop!")
    if hasattr(lbc, 'emergency_stop'):
        lbc.emergency_stop()
except Exception as e:
    print(f"\n❌ Error during movements: {e}")

finally:
    # Always restore ABSOLUTE mode before exiting so app2.py works correctly after
    print("\nRestoring GRBL back to ABSOLUTE mode (G90)...")
    lbc.send_gcode("G90")
    print("Test complete! ✅\n")
