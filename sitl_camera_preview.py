"""Telemetry-driven ideal camera preview. No flight commands.

Requires virtual_marker_pose.py. Camera fixed to vehicle: image-right=body-right,
image-down=body-back, optical axis=body-down. No camera mounting translation.
Ground plane is anchored while stationary/disarmed; marker is 0.3 m north/east.
Telemetry is the rendering reference, not independent physics ground truth.
"""
import math
import sys
import time
import cv2
import numpy as np
from virtual_marker_pose import K, MARKER, HALF, WIDTH, HEIGHT, estimate_from_image

WINDOW = "SITL camera preview - H hide marker - Q quit (does NOT land)"
CAMERA_FROM_BODY = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
CORNER_OFFSETS_NED = np.array([[HALF, -HALF, 0], [HALF, HALF, 0],
                               [-HALF, HALF, 0], [-HALF, -HALF, 0]])


def body_to_ned(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # Aerospace roll/pitch/yaw: body forward/right/down -> north/east/down.
    return np.array([[cp*cy, sr*sp*cy-cr*sy, cr*sp*cy+sr*sy],
                     [cp*sy, sr*sp*sy+cr*cy, cr*sp*sy-sr*cy],
                     [-sp, sr*cp, cr*cp]])


def render_scene(position_ned, attitude, marker_ned, visible=True):
    image = np.full((HEIGHT, WIDTH), 230, dtype=np.uint8)
    rotation = CAMERA_FROM_BODY @ body_to_ned(*attitude).T
    reference = rotation @ (marker_ned - position_ned)
    height = marker_ned[2] - position_ned[2]
    if height < 0.30:
        return image, reference, "GROUND / TOO LOW FOR PREVIEW"
    if not visible:
        return image, reference, "MARKER HIDDEN"
    corners_camera = (rotation @ (marker_ned + CORNER_OFFSETS_NED - position_ned).T).T
    if np.any(corners_camera[:, 2] <= 0.05):
        return image, reference, "MARKER OUTSIDE RENDERING GEOMETRY"
    projected = (K @ corners_camera.T).T
    pixels = projected[:, :2] / projected[:, 2:3]
    source = np.float32([[-0.5, -0.5], [239.5, -0.5],
                         [239.5, 239.5], [-0.5, 239.5]])
    transform = cv2.getPerspectiveTransform(source, pixels.astype(np.float32))
    image = cv2.warpPerspective(MARKER, transform, (WIDTH, HEIGHT),
                               flags=cv2.INTER_NEAREST, borderValue=230)
    return image, reference, "LIVE TELEMETRY CAMERA"


def self_test():
    marker = np.array([0.1, 0.1, 0.0])
    for attitude_deg in [(0,0,0), (5,-5,30), (-6,4,-60), (0,0,90)]:
        attitude = np.radians(attitude_deg)
        image, reference, _ = render_scene(np.array([0.,0.,-2.]), attitude, marker)
        result = estimate_from_image(image)
        assert result is not None, f"No detection for {attitude_deg}"
        error = np.linalg.norm(result[0] - reference)
        assert error < 0.06, f"Pose mismatch {error}"
        print(f"roll/pitch/yaw={attitude_deg}: image-derived position error={error*100:.2f} cm")
    # Independent simple direction expectations at headings 0 and 90 degrees.
    _, ref0, _ = render_scene(np.array([0.,0.,-2.]), (0,0,0), marker)
    _, ref90, _ = render_scene(np.array([0.,0.,-2.]), (0,0,math.pi/2), marker)
    assert np.allclose(ref0, [0.1,-0.1,2])
    assert np.allclose(ref90, [-0.1,-0.1,2])
    image, _, _ = render_scene(np.zeros(3), (0,0,0), marker)
    assert estimate_from_image(image) is None
    image, _, _ = render_scene(np.array([0.,0.,-2.]), (0,0,0), marker, False)
    assert estimate_from_image(image) is None
    print("PASS: tilted/yawed projection, coordinate directions, ground and hidden marker.")


def main():
    from pymavlink import mavutil
    link = mavutil.mavlink_connection("udpin:127.0.0.1:14600", source_system=245, source_component=191)
    cache = {}
    ground_samples = []
    ground = marker = None
    visible = True
    started = time.monotonic()
    requested = False
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    print("Start SITL on the ground, stationary and DISARMED.")
    print("Waiting for autopilot 1/1. This script only requests telemetry.")
    print("After GROUND ANCHORED: use Mission Planner to take off to 2 m, then land.")
    print("Q closes the preview only. It does NOT land or stop a flight.")
    try:
        while True:
            # Drain available packets, bounded so the UI remains responsive.
            for _ in range(500):
                msg = link.recv_match(blocking=False)
                if msg is None: break
                if msg.get_srcSystem() != 1 or msg.get_srcComponent() != 1: continue
                kind = msg.get_type()
                if kind not in ("HEARTBEAT", "LOCAL_POSITION_NED", "ATTITUDE"): continue
                if kind == "HEARTBEAT" and msg.autopilot != mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA: continue
                previous = cache.get(kind)
                if kind != "HEARTBEAT" and previous and msg.time_boot_ms < previous[0].time_boot_ms:
                    raise RuntimeError("Telemetry clock reset. Land/disarm if needed, then restart preview on the ground.")
                now = time.monotonic()
                cache[kind] = (msg, now)
                if kind == "HEARTBEAT" and not requested:
                    if msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                        raise RuntimeError("Start this preview with SITL landed and disarmed.")
                    for message_id in (mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED,
                                       mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE):
                        link.mav.command_long_send(1, 1, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                                                   0, message_id, 100000, 0, 0, 0, 0, 0)
                    requested = True
                if kind == "LOCAL_POSITION_NED" and ground is None and requested:
                    heartbeat = cache.get("HEARTBEAT")
                    if not heartbeat or now-heartbeat[1] > 3: continue
                    if heartbeat[0].base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                        raise RuntimeError("Armed before ground anchor was ready. Land and restart preview.")
                    if np.linalg.norm([msg.vx,msg.vy,msg.vz]) > 0.15:
                        ground_samples.clear()
                        continue
                    if ground_samples and msg.time_boot_ms == ground_samples[-1][0]: continue
                    ground_samples.append((msg.time_boot_ms, [msg.x,msg.y,msg.z]))
                    if len(ground_samples) >= 20:
                        samples = np.array([sample[1] for sample in ground_samples[-20:]])
                        if np.max(np.ptp(samples, axis=0)) > 0.10:
                            ground_samples.clear()
                            continue
                        ground = np.median(samples, axis=0)
                        marker = ground + np.array([0.30,0.30,0.0])
                        print("GROUND ANCHORED. Marker: 0.30 m north and 0.30 m east of start.")
                        print("Ready for a manual SITL takeoff to 2 m in Mission Planner.")
            now = time.monotonic()
            if not requested and now-started > 20:
                raise RuntimeError("No autopilot heartbeat. Check SITL UDP port 14600.")
            frame = np.full((HEIGHT,WIDTH), 230, dtype=np.uint8)
            lines = []
            result = None
            status = "WAITING FOR STATIONARY DISARMED GROUND ANCHOR"
            if ground is not None:
                fresh = all(name in cache and now-cache[name][1] < limit for name,limit in
                            [("HEARTBEAT",3), ("LOCAL_POSITION_NED",0.5), ("ATTITUDE",0.5)])
                if not fresh:
                    status = "STALE / MISSING TELEMETRY - NO CAMERA ESTIMATE"
                else:
                    p = cache["LOCAL_POSITION_NED"][0]
                    a = cache["ATTITUDE"][0]
                    if abs(p.time_boot_ms-a.time_boot_ms) > 150:
                        status = "POSITION / ATTITUDE TIMES TOO FAR APART"
                    else:
                        position = np.array([p.x,p.y,p.z])
                        frame, reference, status = render_scene(position, (a.roll,a.pitch,a.yaw), marker, visible)
                        result = estimate_from_image(frame)
                        lines += [f"Height over anchored plane={ground[2]-p.z:.2f} m",
                                  f"Roll={math.degrees(a.roll):+.1f} Pitch={math.degrees(a.pitch):+.1f} Yaw={math.degrees(a.yaw):+.1f} deg"]
                        if result is not None:
                            estimate, rms, _ = result
                            error = np.linalg.norm(estimate-reference)
                            lines += [f"IMAGE EST: X={estimate[0]:+.3f} Y={estimate[1]:+.3f} Z={estimate[2]:.3f} m",
                                      f"Render reference error={error*100:.2f} cm | reprojection={rms:.2f} px"]
                        else:
                            lines += ["No marker pose available in this frame"]
            preview = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if result is not None:
                cv2.aruco.drawDetectedMarkers(preview,[result[2]],np.array([[0]],dtype=np.int32))
            cv2.drawMarker(preview,(WIDTH//2,HEIGHT//2),(255,0,0),cv2.MARKER_CROSS,16,1)
            panel = np.zeros((190,WIDTH,3),dtype=np.uint8)
            lines = [status] + lines + ["H hide/show | Q closes preview ONLY - use Mission Planner to land",
                                        "Ideal camera driven by telemetry; no vision flight control yet"]
            for index,text in enumerate(lines):
                cv2.putText(panel,text,(8,22+23*index),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,255,0),1,cv2.LINE_AA)
            cv2.imshow(WINDOW,np.vstack((panel,preview)))
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'),27): break
            if key == ord('h'): visible = not visible
            if cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE) < 1: break
    finally:
        link.close()
        cv2.destroyAllWindows()
        print("Preview closed. No flight commands were sent. Manage any ongoing flight in Mission Planner.")


if __name__ == "__main__":
    try:
        if "--self-test" in sys.argv: self_test()
        else: main()
    except KeyboardInterrupt: print("Stopped by user.")
    except (RuntimeError,OSError) as error:
        print("PREVIEW STOPPED:", error)
        raise SystemExit(1)
