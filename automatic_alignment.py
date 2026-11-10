"""Image-based alignment in an ideal kinematic simulator, not ArduPilot.

Requires virtual_marker_pose.py in the same folder. No camera or drone opened.
"""
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from virtual_marker_pose import render_camera, estimate_from_image, WIDTH, HEIGHT

DT = 0.05                    # Simulation seconds per update (20 updates/s).
HEIGHT_M = 2.0               # Fixed height; no descent in this exercise.
GAIN = 0.7                  # Velocity per metre of estimated horizontal error.
MAX_SPEED = 0.25             # Maximum combined horizontal speed, metres/second.
TOLERANCE = 0.03             # Estimated horizontal error must be below 3 cm.
HOLD_STEPS = 20              # Require 1 second continuously inside tolerance.
START = (0.45, -0.30)        # Initial camera right/forward position, metres.
WINDOW = "Automatic alignment - SPACE start/pause - Q quit"


class AlignmentController:
    def __init__(self):
        self.inside_steps = 0

    def update(self, result, enabled):
        # This function never receives simulator position or target truth.
        zero = np.zeros(2)
        if not enabled:
            self.inside_steps = 0
            return zero, "PAUSED"
        if result is None:
            self.inside_steps = 0
            return zero, "TARGET LOST - STOPPED"
        x, y, _ = result[0]
        horizontal_error = float(np.hypot(x, y))
        if horizontal_error <= TOLERANCE:
            self.inside_steps += 1
            state = "ALIGNED - HOLDING" if self.inside_steps >= HOLD_STEPS else "CHECKING ALIGNMENT"
            return zero, state
        self.inside_steps = 0
        # Image right maps to camera right. Image down maps to camera backward.
        velocity = GAIN * np.array([x, -y])
        speed = float(np.linalg.norm(velocity))
        if speed > MAX_SPEED:
            velocity *= MAX_SPEED / speed
        return velocity, "ALIGNING"


def self_test():
    for start in [START, (-0.40, 0.25), (0.25, 0.25)]:
        camera = np.array(start, dtype=float)
        control = AlignmentController()
        for step in range(400):
            result = estimate_from_image(render_camera(*camera, HEIGHT_M))
            assert result is not None, "Unexpected loss in alignment test"
            velocity, state = control.update(result, True)
            assert np.linalg.norm(velocity) <= MAX_SPEED + 1e-9
            camera += velocity * DT
            if state == "ALIGNED - HOLDING":
                break
        assert state == "ALIGNED - HOLDING", "Did not converge"
        assert np.linalg.norm(camera) < 0.04, "True final offset too large"
        print(f"Start={start}: aligned in {(step+1)*DT:.2f} simulated s; true offset={np.linalg.norm(camera)*100:.2f} cm")
    control = AlignmentController()
    camera = np.array(START)
    for _ in range(20):
        result = estimate_from_image(render_camera(*camera, HEIGHT_M, False))
        velocity, state = control.update(result, True)
        assert np.array_equal(velocity, np.zeros(2))
        camera += velocity * DT
    assert np.array_equal(camera, np.array(START))
    result = estimate_from_image(render_camera(*camera, HEIGHT_M))
    velocity, state = control.update(result, True)
    assert state == "ALIGNING" and np.linalg.norm(velocity) > 0
    velocity, state = control.update(result, False)
    assert np.array_equal(velocity, np.zeros(2)) and state == "PAUSED"
    result = estimate_from_image(render_camera(4, 0, HEIGHT_M))
    velocity, state = control.update(result, True)
    assert result is None and np.array_equal(velocity, np.zeros(2))
    print("PASS: convergence, speed cap, target loss, reacquisition and pause.")


def main():
    camera = np.array(START, dtype=float)
    controller = AlignmentController()
    enabled, visible = False, True
    sim_time, trial = 0.0, 1
    directory = Path(__file__).resolve().parent / "alignment_logs"
    directory.mkdir(exist_ok=True)
    log_path = directory / f"alignment_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
    fields = ["sim_time_s", "trial", "state", "auto_enabled", "target_detected",
              "camera_right_m", "camera_forward_m", "height_m", "estimated_x_m",
              "estimated_y_m", "estimated_z_m", "estimated_horizontal_error_m",
              "true_horizontal_error_m", "command_right_m_s", "command_forward_m_s"]
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    print("SPACE: start/pause | H: hide/show | R: reset paused | Q: quit")
    print("A/D W/S reposition only while paused. Height stays at 2 m.")
    print("Time is simulated: each processed frame advances 0.05 seconds.")
    print(f"Recording to: {log_path}")
    previous_state = None
    try:
        with log_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            while True:
                started = time.perf_counter()
                image = render_camera(*camera, HEIGHT_M, visible)
                result = estimate_from_image(image)
                velocity, state = controller.update(result, enabled)
                true_error = float(np.linalg.norm(camera))  # Evaluation only.
                if state != previous_state:
                    print(f"t={sim_time:.2f} s | {state} | true offset={true_error*100:.2f} cm")
                    previous_state = state
                row = dict(sim_time_s=f"{sim_time:.3f}", trial=trial, state=state,
                           auto_enabled=enabled, target_detected=result is not None,
                           camera_right_m=camera[0], camera_forward_m=camera[1],
                           height_m=HEIGHT_M, true_horizontal_error_m=true_error,
                           command_right_m_s=velocity[0], command_forward_m_s=velocity[1])
                preview = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                lines = [state, f"Sim time={sim_time:.2f} s | height={HEIGHT_M:.2f} m | trial={trial}"]
                if result is not None:
                    position, _, corners = result
                    error = float(np.hypot(position[0], position[1]))
                    row.update(estimated_x_m=position[0], estimated_y_m=position[1],
                               estimated_z_m=position[2], estimated_horizontal_error_m=error)
                    cv2.aruco.drawDetectedMarkers(preview, [corners], np.array([[0]], dtype=np.int32))
                    lines += [f"Image estimate: X={position[0]:+.3f} Y={position[1]:+.3f} Z={position[2]:.3f} m",
                              f"Estimated horizontal error={error*100:.2f} cm (limit 3 cm)"]
                else:
                    lines += ["No current target estimate", "Motion command is zero"]
                lines += [f"TRUE horizontal offset={true_error*100:.2f} cm (evaluation only)",
                          f"Speed: right={velocity[0]:+.3f} forward={velocity[1]:+.3f} m/s",
                          "SPACE start/pause | H hide/show | R reset | Q quit",
                          "A/D W/S reposition while paused | Fixed height"]
                panel = np.zeros((210, WIDTH, 3), dtype=np.uint8)
                for index, text in enumerate(lines):
                    colour = (0,255,0) if result is not None else (0,180,255)
                    cv2.putText(panel, text, (8, 22+24*index), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, colour, 1, cv2.LINE_AA)
                cv2.drawMarker(preview, (WIDTH//2, HEIGHT//2), (255,0,0), cv2.MARKER_CROSS, 16, 1)
                writer.writerow(row)
                file.flush()
                cv2.imshow(WINDOW, np.vstack((panel, preview)))
                # Ideal kinematics: immediate velocity response, no inertia or wind.
                camera += velocity * DT
                sim_time += DT
                delay = max(1, int(1000*(DT-(time.perf_counter()-started))))
                key = cv2.waitKey(delay) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif key == ord(' '):
                    enabled = not enabled
                elif key == ord('h'):
                    visible = not visible
                elif key == ord('r'):
                    camera = np.array(START, dtype=float)
                    enabled, visible = False, True
                    controller = AlignmentController()
                    trial += 1
                elif not enabled:
                    if key == ord('a'): camera[0] -= 0.05
                    elif key == ord('d'): camera[0] += 0.05
                    elif key == ord('w'): camera[1] += 0.05
                    elif key == ord('s'): camera[1] -= 0.05
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        print("Stopped by keyboard interrupt.")
    finally:
        cv2.destroyAllWindows()
        print(f"Alignment log saved to: {log_path}")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
