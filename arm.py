import time
import xarm

arm = xarm.Controller('USB')

print('Battery voltage in volts:', arm.getBatteryVoltage())

def set_state(state_name):
    states = {
    'home': [-3.50, 0.50, -85.25, 80.50, 68.75, 12.25],
    'ready_to_grab': [-52.00, 1.00, -69.00, 84.25, -13.50, 11.75],
    'ready_to_move': [-46.25, 1.0, -69.50, 81.25, -2.75, 12.25],
    }

    if state_name in states:
        target = states[state_name]

        moves = [[i + 1, float(angle)] for i, angle in enumerate(target)]

        arm.setPosition(moves, duration=2000, wait=True)
        print(f"Set to {state_name}")
        time.sleep(2.0)
    else:
        print("Invalid state:", state_name)

def get_joints():
    joints = []
    for sid in range(1, 7):
        pos = arm.getPosition(sid, True) 
        joints.append(pos)
    return joints

def check_grip_success(threshold=5.0):
    """
    Simple placeholder for grip success.
    """
    pos6 = arm.getPosition(6, True)
    if pos6 > threshold:
        print("Gripped (simple check).")
        return True
    else:
        print("Not gripped (simple check).")
        return False

def main():
    print("Commands: home / ready_to_grab / ready_to_move / joints / q")

    while True:
        cmd = input("Enter command: ").strip().lower()

        if cmd in ("q", "quit", "exit"):
            print("Exiting.")
            break
        elif cmd == "joints":
            print("Current joints:", get_joints())
        else:
            set_state(cmd)

if __name__ == "__main__":
    main()