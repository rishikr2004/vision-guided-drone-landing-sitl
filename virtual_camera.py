import cv2
import numpy as np

# Virtual camera settings, independent of your laptop webcam calibration.
WIDTH = 640
HEIGHT = 480
FOCAL_LENGTH = 600.0
CX = WIDTH / 2
CY = HEIGHT / 2

# A larger virtual landing marker: 40 cm across its black outer square.
MARKER_SIZE_M = 0.40

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)
marker = cv2.aruco.generateImageMarker(dictionary, 0, 400)

parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

WINDOW = "Virtual downward camera - Q to quit"


def render_camera(camera_x, camera_y, height, visible):
    """Create an ideal image of a flat marker below a level camera."""
    frame = np.full((HEIGHT, WIDTH), 235, dtype=np.uint8)

    if not visible:
        return frame

    # Marker stays at ground position (0, 0).
    # Moving the camera right makes the marker appear to move left.
    centre_x = CX - FOCAL_LENGTH * camera_x / height
    centre_y = CY - FOCAL_LENGTH * camera_y / height

    # Pinhole camera model: projected size decreases with height.
    side_pixels = FOCAL_LENGTH * MARKER_SIZE_M / height
    half = side_pixels / 2

    destination = np.array(
        [
            [centre_x - half, centre_y - half],
            [centre_x + half, centre_y - half],
            [centre_x + half, centre_y + half],
            [centre_x - half, centre_y + half],
        ],
        dtype=np.float32,
    )

    source = np.array(
        [[0, 0], [399, 0], [399, 399], [0, 399]],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(source, destination)

    return cv2.warpPerspective(
        marker,
        transform,
        (WIDTH, HEIGHT),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=235,
    )


def detect_target(frame):
    """Read only image pixels; do not use the virtual camera position."""
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is None:
        return None

    for index, marker_id in enumerate(ids.flatten()):
        if marker_id == 0:
            points = corners[index].reshape(4, 2)
            centre = points.mean(axis=0)
            return points, centre

    return None


def main():
    # Camera starts directly above the marker, two metres high.
    camera_x = 0.0
    camera_y = 0.0
    height = 2.0
    visible = True

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    print("Click the image window before pressing keys.")
    print("A/D: move camera left/right")
    print("W/S: move camera towards image top/bottom")
    print("E/C: increase/decrease camera height")
    print("H: hide/show marker | R: reset | Q: quit")

    try:
        while True:
            frame = render_camera(
                camera_x, camera_y, height, visible
            )

            # Detection happens before any labels or drawings are added.
            result = detect_target(frame)
            display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            cv2.drawMarker(
                display,
                (int(CX), int(CY)),
                (255, 0, 0),
                cv2.MARKER_CROSS,
                16,
                2,
            )

            if result is None:
                status = "Target ID 0 NOT VISIBLE"
                colour = (0, 0, 255)
            else:
                points, centre = result

                dx = centre[0] - CX
                dy = centre[1] - CY

                cv2.polylines(
                    display,
                    [np.round(points).astype(np.int32)],
                    True,
                    (0, 180, 0),
                    2,
                )

                cv2.circle(
                    display,
                    tuple(np.round(centre).astype(int)),
                    4,
                    (0, 0, 255),
                    -1,
                )

                status = (
                    f"ID 0 | dx={dx:+.1f} px | dy={dy:+.1f} px"
                )
                colour = (0, 180, 0)

            # Labels occupy a separate panel above the camera image.
            panel = np.zeros((110, WIDTH, 3), dtype=np.uint8)

            cv2.putText(
                panel, status, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1
            )

            cv2.putText(
                panel,
                f"Simulator height: {height:.1f} m (known setting)",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1,
            )

            cv2.putText(
                panel, "A/D W/S: move | E/C: height",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1,
            )

            cv2.putText(
                panel, "H: hide/show | R: reset | Q: quit",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1,
            )

            cv2.imshow(WINDOW, np.vstack((panel, display)))
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("a"):
                camera_x -= 0.05
            elif key == ord("d"):
                camera_x += 0.05
            elif key == ord("w"):
                camera_y -= 0.05
            elif key == ord("s"):
                camera_y += 0.05
            elif key == ord("e"):
                height = min(5.0, height + 0.1)
            elif key == ord("c"):
                height = max(0.5, height - 0.1)
            elif key == ord("h"):
                visible = not visible
            elif key == ord("r"):
                camera_x = 0.0
                camera_y = 0.0
                height = 2.0
                visible = True

            if cv2.getWindowProperty(
                WINDOW, cv2.WND_PROP_VISIBLE
            ) < 1:
                break

    finally:
        cv2.destroyAllWindows()
        print("Virtual camera closed.")


if __name__ == "__main__":
    main()