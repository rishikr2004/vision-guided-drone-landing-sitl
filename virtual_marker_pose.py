"""Ideal, level downward camera: render pixels, detect, estimate, compare.

No webcam or flight connection. All lengths are metres.
Controls: A/D left/right, W/S forward/back, E/C higher/lower,
H hide/show, R reset, Q quit. Run --self-test for a headless check.
"""
import sys
import cv2
import numpy as np

WIDTH, HEIGHT = 640, 480
FOCAL_PX = 600.0
MARKER_SIZE_M = 0.40
K = np.array([[FOCAL_PX, 0, WIDTH / 2],
              [0, FOCAL_PX, HEIGHT / 2], [0, 0, 1]], dtype=float)
DISTORTION = np.zeros(5)
HALF = MARKER_SIZE_M / 2
# Marker-local coordinates: x right, y down, z out of the plane.
# Order matches decoded ArUco corners: top-left clockwise.
OBJECT_POINTS = np.array([[-HALF, -HALF, 0], [HALF, -HALF, 0],
                          [HALF, HALF, 0], [-HALF, HALF, 0]], dtype=float)
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
MARKER = cv2.aruco.generateImageMarker(DICTIONARY, 0, 240)
PARAMETERS = cv2.aruco.DetectorParameters()
PARAMETERS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
DETECTOR = cv2.aruco.ArucoDetector(DICTIONARY, PARAMETERS)
WINDOW = "Virtual marker position - Q to quit"


def render_camera(right, forward, height, visible=True):
    """Only the renderer uses simulated camera position to generate pixels."""
    if not visible:
        return np.full((HEIGHT, WIDTH), 230, dtype=np.uint8)
    size = FOCAL_PX * MARKER_SIZE_M / height
    u = WIDTH / 2 - FOCAL_PX * right / height
    v = HEIGHT / 2 + FOCAL_PX * forward / height
    half = size / 2
    destination = np.float32([[u-half, v-half], [u+half, v-half],
                              [u+half, v+half], [u-half, v+half]])
    # Use pixel boundaries, not pixel centres, to define physical edges.
    source = np.float32([[-0.5, -0.5], [239.5, -0.5],
                         [239.5, 239.5], [-0.5, 239.5]])
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(MARKER, transform, (WIDTH, HEIGHT),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=230)


def estimate_from_image(image):
    """Estimator gets pixels and known camera/marker geometry, never truth."""
    corners, ids, _ = DETECTOR.detectMarkers(image)
    if ids is None:
        return None
    for index, marker_id in enumerate(ids.flatten()):
        if marker_id != 0:
            continue
        points = corners[index].reshape(4, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            OBJECT_POINTS, points, K, DISTORTION,
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or not np.all(np.isfinite(tvec)) or not np.all(np.isfinite(rvec)):
            return None
        rotation, _ = cv2.Rodrigues(rvec)
        camera_points = (rotation @ OBJECT_POINTS.T + tvec).T
        if np.any(camera_points[:, 2] <= 0):
            return None
        projected, _ = cv2.projectPoints(OBJECT_POINTS, rvec, tvec, K, DISTORTION)
        residual = projected.reshape(4, 2) - points
        rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        if rms > 3.0:
            return None
        return tvec.reshape(3), rms, corners[index]
    return None


def self_test():
    # Integration checks on images, not direct access to true corner positions.
    for right, forward, height in [(0, 0, 2), (0.25, -0.15, 2),
                                   (-0.2, 0.15, 1), (0, 0, 3)]:
        result = estimate_from_image(render_camera(right, forward, height))
        assert result is not None, "Expected marker detection"
        estimate, rms, _ = result
        truth = np.array([-right, forward, height])
        error = float(np.linalg.norm(estimate - truth))
        assert error < 0.05, f"Synthetic check error too large: {error}"
        print(f"height={height:.1f} m, position error={error:.4f} m, RMS={rms:.3f} px")
    assert estimate_from_image(render_camera(0, 0, 2, False)) is None
    assert estimate_from_image(render_camera(4, 0, 2)) is None
    assert estimate_from_image(render_camera(0, 0, 2)) is not None
    print("PASS: synthetic position, hidden, out-of-view and return checks.")


def main():
    right, forward, height, visible = 0.0, 0.0, 2.0, True
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    print("Click the image window. A/D W/S move; E/C height; H hide; R reset; Q quit.")
    print("This is ideal synthetic geometry, not a flight or real-camera accuracy test.")
    try:
        while True:
            image = render_camera(right, forward, height, visible)
            result = estimate_from_image(image)  # Fresh result on EVERY frame.
            preview = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            truth = np.array([-right, forward, height])  # Evaluation only.
            truth_range = float(np.linalg.norm(truth))
            lines = [("Camera coordinates: X image-right, Y image-down, Z optical depth", (255,255,255)),
                     (f"TRUE target: X={truth[0]:+.3f} Y={truth[1]:+.3f} Z={truth[2]:.3f} m", (255,255,255)),
                     (f"TRUE range={truth_range:.3f} m | virtual height={height:.2f} m", (255,255,255))]
            if result is None:
                lines += [("NO VALID TARGET - no current position estimate", (0,0,255)),
                          ("Estimate and errors unavailable", (0,0,255))]
            else:
                position, rms, marker_corners = result
                x, y, z = position
                distance = float(np.linalg.norm(position))
                error = float(np.linalg.norm(position - truth))
                cv2.aruco.drawDetectedMarkers(preview, [marker_corners], np.array([[0]], dtype=np.int32))
                lines += [(f"ESTIMATE: X={x:+.3f} Y={y:+.3f} Z={z:.3f} m", (0,255,0)),
                          (f"EST range={distance:.3f} m | position error={error*100:.2f} cm", (0,255,0)),
                          (f"Range error={(distance-truth_range)*100:+.2f} cm | image RMS={rms:.3f} px", (0,255,0))]
            lines += [("A/D W/S: move | E/C: height | H: hide | R: reset | Q: quit", (255,255,255))]
            panel = np.zeros((190, WIDTH, 3), dtype=np.uint8)
            for index, (text, colour) in enumerate(lines):
                cv2.putText(panel, text, (8, 22+index*24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
            cv2.drawMarker(preview, (WIDTH//2, HEIGHT//2), (255,0,0),
                           cv2.MARKER_CROSS, 16, 1)
            cv2.imshow(WINDOW, np.vstack((panel, preview)))
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('a'): right -= 0.05
            elif key == ord('d'): right += 0.05
            elif key == ord('w'): forward += 0.05
            elif key == ord('s'): forward -= 0.05
            elif key == ord('e'): height = min(5.0, height + 0.10)
            elif key == ord('c'): height = max(0.50, height - 0.10)
            elif key == ord('h'): visible = not visible
            elif key == ord('r'): right, forward, height, visible = 0.0, 0.0, 2.0, True
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()
        print("Virtual position window closed.")


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        self_test()
    else:
        main()
