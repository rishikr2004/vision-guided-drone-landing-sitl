"""Estimate camera intrinsics in pixels; board coordinates use square units."""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--session", help="Session folder name printed by the capture script")
args = parser.parse_args()
root = Path(__file__).resolve().parent / "calibration"
sessions = sorted(p for p in root.glob("session_*") if list(p.glob("view_*.png")))
if args.session:
    folder = root / args.session
elif sessions:
    folder = sessions[-1]
else:
    raise SystemExit("No captured images. Run capture_calibration.py first.")
print(f"Using session: {folder.name}")
# 54 ideal corners on a flat plane. One unit equals one checkerboard square.
board = np.zeros((9 * 6, 3), np.float32)
board[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
object_points, image_points, names = [], [], []
image_size = None
for path in sorted(folder.glob("view_*.png")):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Cannot read {path.name}")
    size = (gray.shape[1], gray.shape[0])
    if image_size is None:
        image_size = size
    if size != image_size:
        raise RuntimeError("Mixed image resolutions. Capture a new consistent session.")
    found, corners = cv2.findChessboardCornersSB(gray, (9, 6), cv2.CALIB_CB_NORMALIZE_IMAGE)
    if found:
        object_points.append(board.copy())
        image_points.append(corners)
        names.append(path.name)
    else:
        print(f"Skipped {path.name}: checkerboard not detected.")
if len(names) < 20:
    raise SystemExit(f"Only {len(names)} usable images. Collect a new session with 25-30 varied views.")
rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
    object_points, image_points, image_size, None, None)
if not (np.isfinite(matrix).all() and np.isfinite(distortion).all() and np.isfinite(rms)):
    raise RuntimeError("Non-finite calibration. Capture a new, more varied session.")
errors = []
for obj, observed, rotation, translation in zip(object_points, image_points, rotations, translations):
    projected, _ = cv2.projectPoints(obj, rotation, translation, matrix, distortion)
    delta = observed.reshape(-1, 2) - projected.reshape(-1, 2)
    errors.append(float(np.sqrt(np.mean(np.sum(delta ** 2, axis=1)))))
np.savez(folder / "camera_calibration.npz", camera_matrix=matrix, dist_coeffs=distortion,
         image_size=np.array(image_size), rms=np.array(rms))
report = dict(session=folder.name, usable_images=len(names), image_size=image_size,
              rms_pixels=float(rms), camera_matrix=matrix.tolist(), distortion=distortion.tolist(),
              board_coordinate_units="checkerboard squares, NOT metres",
              per_image_rms_pixels=dict(zip(names, errors)),
              status="Candidate calibration; independent validation still required")
(folder / "calibration_report.json").write_text(json.dumps(report, indent=2))
lines = [f"Session: {folder.name}", f"Usable images: {len(names)}", f"Image size: {image_size}",
         f"Overall fitting RMS: {rms:.3f} pixels", f"Worst image RMS: {max(errors):.3f} pixels",
         "Camera matrix:", str(matrix), "Distortion coefficients:", str(distortion),
         "", "Per-image fitting RMS (pixels):"]
lines.extend(f"{name}: {err:.3f}" for name, err in zip(names, errors))
lines.extend(["", "These are fitting errors on the calibration images, not independent accuracy measurements.",
              "Small fitting error alone does not prove correct distance estimates.",
              "Use the same camera, image size, focus and zoom for later measurements.",
              "Calibration is a candidate until checked with new views and physical measurements."])
text = "\n".join(lines)
(folder / "calibration_report.txt").write_text(text)
print(text)
print(f"Saved settings and reports in: {folder}")
