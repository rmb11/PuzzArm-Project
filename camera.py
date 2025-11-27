import cv2
import os
import json

from arm import set_state, get_joints, arm

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: could not open camera on index 0.")
        return

    dataset_dir = "nav_dataset"
    os.makedirs(dataset_dir, exist_ok=True)

    existing = [
        f for f in os.listdir(dataset_dir)
        if f.startswith("image_") and f.endswith(".jpg")
    ]
    counter = len(existing)

    print("SPACE = capture one demo (image + joints)")

    set_state("ready_to_move")

    print("Servos off, position the arm by hand into grip alignment.")
    try:
        arm.servoOff()
    except Exception as e:
        print("Warning: servoOff() failed - arm may not relax:", e)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to read from camera. Exiting.")
            break

        cv2.imshow("PuzzArm Grip View", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '): # Press spacebar to take picture
            img_path = os.path.join(dataset_dir, f"image_{counter:03d}.jpg")
            json_path = os.path.join(dataset_dir, f"joints_{counter:03d}.json")

            cv2.imwrite(img_path, frame)
            joints = [round(j) for j in get_joints(degrees=True)]

            data = {
                "joints": joints,
                "deltas": [0.0] * 6  # placeholder for now
            }

            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)

            print(f"Captured demo {counter}: {img_path}")
            print(f"  Joints: {joints}")
            counter += 1

        elif key == ord('q'):
            print("Quitting data collection.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()