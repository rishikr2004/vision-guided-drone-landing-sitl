"""Save raw webcam images only when all 54 inner corners are detected."""
from pathlib import Path
from datetime import datetime
import argparse
import json
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--backend", choices=["dshow", "auto"], default="dshow")
args = parser.parse_args()
folder = Path(__file__).resolve().parent / "calibration" / datetime.now().strftime("session_%Y%m%d_%H%M%S_%f")
folder.mkdir(parents=True)
backend = cv2.CAP_DSHOW if args.backend == "dshow" else cv2.CAP_ANY
camera = cv2.VideoCapture(args.camera, backend)
count = 0
image_size = None
window = "Calibration - S save, Q quit"
try:
    if not camera.isOpened():
        raise RuntimeError("Cannot open camera. Close other camera apps.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print("Hold tablet steady. S saves one image. Change position/tilt before saving again.")
    print("Aim for 25-30 varied, sharp views, then Q. Images stay on this laptop.")
    print(f"Session: {folder}")
    while True:
        ok, frame = camera.read()
        if not ok:
            raise RuntimeError("No camera image received.")
        size = (frame.shape[1], frame.shape[0])
        if image_size is None:
            image_size = size
            print(f"Actual image size: {size[0]} x {size[1]}")
        if size != image_size:
            raise RuntimeError("Camera resolution changed during capture.")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCornersSB(gray, (9, 6), cv2.CALIB_CB_NORMALIZE_IMAGE)
        preview = frame.copy()
        if found:
            cv2.drawChessboardCorners(preview, (9, 6), corners, found)
        text = f"{'FOUND: S to save' if found else 'Show entire checkerboard'} | Saved: {count}"
        cv2.rectangle(preview, (0, 0), (size[0], 35), (0, 0, 0), -1)
        cv2.putText(preview, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0) if found else (0, 0, 255), 1)
        cv2.imshow(window, preview)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        if key in (ord('s'), ord('S')):
            if not found:
                print("Not saved: all inner corners must be detected.")
                continue
            path = folder / f"view_{count + 1:03d}.png"
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError("Image save failed.")
            count += 1
            print(f"Saved {path.name}. Change position or angle before the next image.")
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
except KeyboardInterrupt:
    print("Capture stopped.")
finally:
    camera.release()
    cv2.destroyAllWindows()
    metadata = dict(camera_index=args.camera, backend=args.backend, image_size=image_size,
                    pattern_inner_corners=[9, 6], captured_images=count)
    (folder / "capture_settings.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved {count} images in {folder}")
