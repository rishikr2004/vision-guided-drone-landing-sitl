"""Detect marker ID 0 with a webcam. No drone commands or video recording."""
import argparse
import time
import cv2

# Optional arguments let us change the camera without editing the script.
parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--backend", choices=["dshow", "auto"], default="dshow")
args = parser.parse_args()

TARGET_ID = 0
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
backend = cv2.CAP_DSHOW if args.backend == "dshow" else cv2.CAP_ANY
camera = cv2.VideoCapture(args.camera, backend)
window = "Landing marker detector - Q to quit"

try:
    if not camera.isOpened():
        raise RuntimeError("Cannot open webcam. Close other camera apps and check camera permissions.")
    # These are requested dimensions; the camera may choose different ones.
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print("Show marker ID 0 to the webcam. Click the video window and press Q to quit.")
    last_print = 0.0

    while True:
        ok, frame = camera.read()
        if not ok:
            raise RuntimeError("Camera opened but no image was received.")
        # Keep the camera image unmirrored for detection.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)
        height, width = frame.shape[:2]
        screen_x, screen_y = width // 2, height // 2
        cv2.drawMarker(frame, (screen_x, screen_y), (255, 0, 0),
                       cv2.MARKER_CROSS, 20, 2)
        status = "Target ID 0: NOT VISIBLE"
        colour = (0, 0, 255)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                if int(marker_id) != TARGET_ID:
                    continue
                # Average the four detected corner positions for a simple centre.
                points = marker_corners.reshape(4, 2)
                centre_x, centre_y = points.mean(axis=0)
                dx = float(centre_x - screen_x)
                dy = float(centre_y - screen_y)
                centre = (round(float(centre_x)), round(float(centre_y)))
                cv2.circle(frame, centre, 5, (0, 255, 0), -1)
                cv2.line(frame, (screen_x, screen_y), centre, (0, 255, 255), 2)
                status = f"ID 0 | dx={dx:+.0f} px | dy={dy:+.0f} px"
                colour = (0, 255, 0)
                break

        cv2.rectangle(frame, (0, 0), (width, 38), (0, 0, 0), -1)
        cv2.putText(frame, status, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, colour, 2)
        cv2.imshow(window, frame)
        # Print twice per second instead of filling the terminal every frame.
        now = time.monotonic()
        if now - last_print >= 0.5:
            print(status)
            last_print = now
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Webcam released.")
