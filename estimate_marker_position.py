"""Stage 2: webcam pose estimates only. No drone commands or video recording."""
from pathlib import Path
from datetime import datetime
import argparse
import csv
import time
import cv2
import numpy as np

# Keep the phone marker at the size you measured, including its black border.
MARKER_SIZE_M = 0.0525
TARGET_ID = 0
CALIBRATION_SESSION = "session_20261109_032931_545000"
MAX_REPROJECTION_RMS_PX = 3.0  # Diagnostic filter, NOT a distance-accuracy guarantee.


def estimate_pose(corners, camera_matrix, distortion):
    """Choose the positive-depth square-pose solution with the lowest image error."""
    half = MARKER_SIZE_M / 2
    # Required IPPE_SQUARE corner order; matches ArUco's canonical corner order.
    object_points = np.array([[-half, half, 0], [half, half, 0],
                              [half, -half, 0], [-half, -half, 0]], dtype=np.float64)
    image_points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    result = cv2.solvePnPGeneric(object_points, image_points, camera_matrix,
                                distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    candidates = []
    if not result[0]:
        return None
    for rotation, translation in zip(result[1], result[2]):
        if not (np.isfinite(rotation).all() and np.isfinite(translation).all()):
            continue
        rotation_matrix, _ = cv2.Rodrigues(rotation)
        camera_points = (rotation_matrix @ object_points.T + translation.reshape(3, 1)).T
        if np.any(camera_points[:, 2] <= 0):
            continue
        projected, _ = cv2.projectPoints(object_points, rotation, translation,
                                         camera_matrix, distortion)
        residual = projected.reshape(4, 2) - image_points
        rms = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
        candidates.append((rms, rotation, translation))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--backend', choices=['dshow', 'auto'], default='dshow')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    calibration_path = root / 'calibration' / CALIBRATION_SESSION / 'camera_calibration.npz'
    if not calibration_path.exists():
        raise FileNotFoundError(f'Calibration not found: {calibration_path}')
    with np.load(calibration_path, allow_pickle=False) as data:
        matrix = data['camera_matrix'].copy()
        distortion = data['dist_coeffs'].copy()
        width, height = map(int, data['image_size'])
    print(f'Calibration: {calibration_path}')
    print(f'Required resolution: {width} x {height}; marker width: {MARKER_SIZE_M * 100:.2f} cm')
    print('Camera coordinates: X right, Y down, Z forward. Display is not mirrored.')
    print('Q quits. S saves the CURRENT accepted reading. No smoothing is applied.')
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    logs = root / 'position_checks'
    logs.mkdir(exist_ok=True)
    log_path = logs / f"position_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
    backend = cv2.CAP_DSHOW if args.backend == 'dshow' else cv2.CAP_ANY
    camera = cv2.VideoCapture(args.camera, backend)
    window = 'Marker position - S save reading, Q quit'
    last_print = 0.0
    try:
        if not camera.isOpened():
            raise RuntimeError('Cannot open camera. Close other camera apps.')
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        with log_path.open('x', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['received_at', 'x_m', 'y_m', 'z_m', 'range_m',
                             'reprojection_rms_px', 'marker_size_m', 'calibration_session'])
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError('No camera image received.')
                if (frame.shape[1], frame.shape[0]) != (width, height):
                    raise RuntimeError('Actual camera resolution differs from calibration. Do not resize the image to hide this mismatch.')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = detector.detectMarkers(gray)
                reading = None  # Clear every frame: never reuse a missing target's pose.
                status = 'ID 0 NOT VISIBLE'
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    for marker_corners, marker_id in zip(corners, ids.flatten()):
                        if int(marker_id) != TARGET_ID:
                            continue
                        pose = estimate_pose(marker_corners, matrix, distortion)
                        if pose is None:
                            status = 'Pose unavailable'
                        elif pose[0] > MAX_REPROJECTION_RMS_PX:
                            status = f'Pose rejected: image error {pose[0]:.2f} px'
                        else:
                            rms, rotation, translation = pose
                            x, y, z = map(float, translation.reshape(3))
                            distance = float(np.linalg.norm(translation))
                            reading = (x, y, z, distance, rms)
                            status = f'ID 0 | image error: {rms:.2f} px'
                            cv2.drawFrameAxes(frame, matrix, distortion, rotation,
                                             translation, MARKER_SIZE_M / 2, 2)
                        break
                lines = [status]
                if reading is not None:
                    x, y, z, distance, rms = reading
                    lines += [f'X right: {x*100:+.1f} cm   Y down: {y*100:+.1f} cm',
                              f'Z forward depth: {z*100:.1f} cm',
                              f'Straight-line range: {distance*100:.1f} cm']
                # Add a separate text panel so the original camera frame is not cropped.
                panel = np.zeros((110, width, 3), dtype=np.uint8)
                for index, line in enumerate(lines):
                    cv2.putText(panel, line, (8, 23 + 25*index), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 255, 0) if reading is not None else (0, 180, 255), 1)
                cv2.imshow(window, np.vstack([panel, frame]))
                now = time.monotonic()
                if now - last_print >= 0.5:
                    print(' | '.join(lines))
                    last_print = now
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                if key in (ord('s'), ord('S')):
                    if reading is None:
                        print('Not saved: no accepted marker position in this frame.')
                    else:
                        writer.writerow([datetime.now().astimezone().isoformat(), *reading,
                                         MARKER_SIZE_M, CALIBRATION_SESSION])
                        file.flush()
                        print(f'SAVED reading: Z={reading[2]*100:.1f} cm; range={reading[3]*100:.1f} cm')
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        print('Stopped by user.')
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print(f'Webcam released. Manual readings file: {log_path}')


if __name__ == '__main__':
    main()
