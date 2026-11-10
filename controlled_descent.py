"""Ideal image-based alignment and descent to a stand-off height, NOT landing.
Requires virtual_marker_pose.py beside this file. No hardware/ArduPilot connection.
"""
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from virtual_marker_pose import render_camera, estimate_from_image, WIDTH, HEIGHT

DT = 0.05
START = (0.45, -0.30, 2.0)  # Camera right, forward and height, metres.
STOP_HEIGHT = 0.80          # Estimated optical depth; level downward camera only.
DESCENT_SPEED = 0.15        # Metres per second, positive DOWN.
GAIN = 0.7
MAX_HORIZONTAL_SPEED = 0.25
ALIGN_TOLERANCE = 0.03
DESCENT_ERROR_LIMIT = 0.06  # Stop descending and realign above this offset.
HOLD_STEPS = 20
WINDOW = "Controlled descent - SPACE start/pause - Q quit"


class DescentController:
    def __init__(self):
        self.aligned = False
        self.inside_steps = 0
        self.complete = False

    def update(self, result, enabled):
        # Input is image-derived pose only, never simulated position.
        zero = np.zeros(3)
        if not enabled:
            self.aligned = False
            self.inside_steps = 0
            return zero, "PAUSED"
        if result is None:
            self.aligned = False
            self.inside_steps = 0
            return zero, "TARGET LOST - ALL MOTION STOPPED"
        if self.complete:
            return zero, "COMPLETE - HOLDING ABOVE MARKER"
        x, y, z = result[0]
        error = float(np.hypot(x, y))
        horizontal = GAIN * np.array([x, -y])
        speed = float(np.linalg.norm(horizontal))
        if speed > MAX_HORIZONTAL_SPEED:
            horizontal *= MAX_HORIZONTAL_SPEED / speed
        if self.aligned and error > DESCENT_ERROR_LIMIT:
            self.aligned = False
            self.inside_steps = 0
        if not self.aligned:
            if error > ALIGN_TOLERANCE:
                self.inside_steps = 0
                return np.array([*horizontal, 0.0]), "ALIGNING - HEIGHT HELD"
            self.inside_steps += 1
            if self.inside_steps < HOLD_STEPS:
                return zero, "CONFIRMING ALIGNMENT - HEIGHT HELD"
            self.aligned = True
        if z <= STOP_HEIGHT:
            self.complete = True
            return zero, "COMPLETE - HOLDING ABOVE MARKER"
        # Constant slow descent. The final step can undershoot by one time step
        # plus estimation error. The true simulator height is not used here.
        return np.array([*horizontal, DESCENT_SPEED]), "DESCENDING"


def advance(camera, command):
    # Ideal kinematics: instant velocity changes, no inertia, wind or tilt.
    updated = camera + np.array([command[0], command[1], -command[2]]) * DT
    if updated[2] < 0.30:
        raise RuntimeError("SIMULATOR GUARD: height below 0.30 m; run aborted, not completed")
    return updated


def self_test():
    for start in [START, (-0.4, 0.25, 2.0), (0.25, 0.25, 1.5)]:
        camera = np.array(start, dtype=float)
        controller = DescentController()
        for step in range(1000):
            result = estimate_from_image(render_camera(*camera))
            assert result is not None
            command, state = controller.update(result, True)
            assert np.linalg.norm(command[:2]) <= MAX_HORIZONTAL_SPEED + 1e-9
            assert 0 <= command[2] <= DESCENT_SPEED
            camera = advance(camera, command)
            if controller.complete:
                break
        assert controller.complete, "Descent did not complete"
        assert 0.75 <= camera[2] <= 0.85, "Unexpected true final height"
        assert np.linalg.norm(camera[:2]) < 0.04
        print(f"Completed: true height={camera[2]:.3f} m, horizontal offset={np.linalg.norm(camera[:2])*100:.2f} cm")
    # Lose the target during descent; confirm the entire position stays fixed.
    camera = np.array(START)
    controller = DescentController()
    for _ in range(500):
        result = estimate_from_image(render_camera(*camera))
        command, state = controller.update(result, True)
        camera = advance(camera, command)
        if state == "DESCENDING":
            break
    assert state == "DESCENDING"
    stopped = camera.copy()
    for _ in range(40):
        result = estimate_from_image(render_camera(*camera, visible=False))
        command, state = controller.update(result, True)
        assert np.array_equal(command, np.zeros(3))
        camera = advance(camera, command)
    assert np.array_equal(camera, stopped)
    for _ in range(HOLD_STEPS - 1):
        command, state = controller.update(estimate_from_image(render_camera(*camera)), True)
        assert command[2] == 0, "Descended before reacquisition hold finished"
    command, state = controller.update(estimate_from_image(render_camera(*camera)), True)
    assert state == "DESCENDING"
    # A visible lateral displacement pauses descent and requests alignment.
    camera[0] += 0.12
    command, state = controller.update(estimate_from_image(render_camera(*camera)), True)
    assert command[2] == 0 and state == "ALIGNING - HEIGHT HELD"
    command, state = controller.update(estimate_from_image(render_camera(*camera)), False)
    assert np.array_equal(command, np.zeros(3))
    print("PASS: descent, speed limits, target loss, reacquisition, realignment and pause.")


def main():
    camera = np.array(START, dtype=float)
    controller = DescentController()
    enabled, visible = False, True
    sim_time, trial = 0.0, 1
    folder = Path(__file__).resolve().parent / "descent_logs"
    folder.mkdir(exist_ok=True)
    path = folder / f"descent_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
    fields = ["sim_time_s", "trial", "state", "enabled", "target_detected", "completed",
              "true_right_m", "true_forward_m", "true_height_m", "true_horizontal_error_m",
              "estimated_x_m", "estimated_y_m", "estimated_z_m", "estimated_horizontal_error_m",
              "command_right_m_s", "command_forward_m_s", "command_down_m_s"]
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    previous_state = None
    print("SPACE start/pause | H hide/show | R reset paused | Q quit")
    print("A/D W/S reposition while paused. Reset before a new descent after completion.")
    print("Stop height: estimated 0.8 m. This is NOT touchdown or ArduPilot flight.")
    print(f"Recording to: {path}")
    try:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            while True:
                started = time.perf_counter()
                image = render_camera(*camera, visible=visible)
                result = estimate_from_image(image)
                command, state = controller.update(result, enabled)
                true_error = float(np.linalg.norm(camera[:2]))
                if state != previous_state:
                    print(f"t={sim_time:.2f} s | {state} | true height={camera[2]:.3f} m | true offset={true_error*100:.2f} cm")
                    previous_state = state
                row = dict(sim_time_s=round(sim_time, 3), trial=trial, state=state,
                           enabled=enabled, target_detected=result is not None, completed=controller.complete,
                           true_right_m=camera[0], true_forward_m=camera[1], true_height_m=camera[2],
                           true_horizontal_error_m=true_error, command_right_m_s=command[0],
                           command_forward_m_s=command[1], command_down_m_s=command[2])
                preview = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                lines = [state, f"Sim time={sim_time:.2f} s | trial={trial} | stop at estimated 0.80 m"]
                if result is None:
                    lines += ["No valid image-derived position", "No horizontal or downward movement"]
                else:
                    position, _, corners = result
                    error = float(np.hypot(*position[:2]))
                    row.update(estimated_x_m=position[0], estimated_y_m=position[1],
                               estimated_z_m=position[2], estimated_horizontal_error_m=error)
                    cv2.aruco.drawDetectedMarkers(preview, [corners], np.array([[0]], dtype=np.int32))
                    lines += [f"EST X={position[0]:+.3f} Y={position[1]:+.3f} Z={position[2]:.3f} m",
                              f"EST horizontal error={error*100:.2f} cm"]
                lines += [f"TRUE height={camera[2]:.3f} m | TRUE horizontal error={true_error*100:.2f} cm",
                          f"Speed right={command[0]:+.2f} forward={command[1]:+.2f} DOWN={command[2]:.2f} m/s",
                          "SPACE start/pause | H hide/show | R reset | Q quit",
                          "A/D W/S move while paused | Ideal simulation, NO touchdown"]
                panel = np.zeros((210, WIDTH, 3), dtype=np.uint8)
                for index, text in enumerate(lines):
                    colour = (0,255,0) if result is not None else (0,180,255)
                    cv2.putText(panel, text, (8, 22+24*index), cv2.FONT_HERSHEY_SIMPLEX,
                                0.44, colour, 1, cv2.LINE_AA)
                cv2.drawMarker(preview, (WIDTH//2, HEIGHT//2), (255,0,0), cv2.MARKER_CROSS, 16, 1)
                writer.writerow(row)
                file.flush()
                cv2.imshow(WINDOW, np.vstack((panel, preview)))
                camera = advance(camera, command)
                sim_time += DT
                delay = max(1, int(1000*(DT-(time.perf_counter()-started))))
                key = cv2.waitKey(delay) & 0xFF
                if key in (ord('q'), 27): break
                elif key == ord(' '): enabled = not enabled
                elif key == ord('h'): visible = not visible
                elif key == ord('r'):
                    camera = np.array(START, dtype=float)
                    controller = DescentController()
                    enabled, visible = False, True
                    trial += 1
                elif not enabled and not controller.complete:
                    if key == ord('a'): camera[0] -= 0.05
                    elif key == ord('d'): camera[0] += 0.05
                    elif key == ord('w'): camera[1] += 0.05
                    elif key == ord('s'): camera[1] -= 0.05
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1: break
    except KeyboardInterrupt:
        print("Stopped by user; check recorded states for completion.")
    finally:
        cv2.destroyAllWindows()
        print(f"Descent log saved to: {path}")


if __name__ == "__main__":
    if "--self-test" in sys.argv: self_test()
    else: main()
