"""Check ArduPilot SITL telemetry. Requests message rates, no flight commands."""
import math
import time
from pymavlink import mavutil

ADDRESS = "udpin:127.0.0.1:14600"
SYSTEM, COMPONENT = 1, 1


def from_autopilot(message):
    return (message.get_srcSystem() == SYSTEM
            and message.get_srcComponent() == COMPONENT)


def main():
    connection = mavutil.mavlink_connection(
        ADDRESS, source_system=245, source_component=191)
    try:
        print(f"Listening on {ADDRESS}")
        print("Waiting up to 20 seconds for autopilot 1/1...")
        deadline = time.monotonic() + 20
        heartbeat = None
        while time.monotonic() < deadline:
            message = connection.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
            if message is not None and from_autopilot(message):
                if message.autopilot == mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA:
                    heartbeat = message
                    break
        if heartbeat is None:
            raise RuntimeError("No ArduPilot heartbeat from 1/1. Check SITL and UDP port 14600.")
        print("Autopilot connected. Mode:", mavutil.mode_string_v10(heartbeat))
        armed = bool(heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        print("Armed:", armed)
        if armed:
            raise RuntimeError("Complete the current simulated flight and disarm before this check.")
        # Request three telemetry messages at 2 Hz (500,000 microseconds).
        message_ids = [mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
                       mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED,
                       mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE]
        for message_id in message_ids:
            connection.mav.command_long_send(
                SYSTEM, COMPONENT, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0, message_id, 500000, 0, 0, 0, 0, 0)
        print("Requested position and attitude at 2 Hz. Reading for 15 seconds...")
        names = ["GLOBAL_POSITION_INT", "LOCAL_POSITION_NED", "ATTITUDE"]
        counts = dict.fromkeys(names, 0)
        latest = {}
        started = time.monotonic()
        next_print = started + 1
        last_heartbeat = started
        while time.monotonic() - started < 15:
            message = connection.recv_match(blocking=True, timeout=0.5)
            now = time.monotonic()
            if message is not None and from_autopilot(message):
                kind = message.get_type()
                if kind == "HEARTBEAT":
                    last_heartbeat = now
                    if message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                        raise RuntimeError("Simulator was armed during check; stopping telemetry check.")
                elif kind in counts:
                    counts[kind] += 1
                    latest[kind] = (message, now)
            if now >= next_print:
                for name in names:
                    if name not in latest:
                        print(f"{name}: waiting...")
                        continue
                    value, received = latest[name]
                    age = now - received
                    if age > 2:
                        print(f"{name}: STALE ({age:.1f} s old)")
                    elif name == "GLOBAL_POSITION_INT":
                        print(f"GPS: lat={value.lat/1e7:.7f} lon={value.lon/1e7:.7f} home-relative altitude={value.relative_alt/1000:.2f} m")
                    elif name == "LOCAL_POSITION_NED":
                        print(f"LOCAL: North={value.x:+.2f} East={value.y:+.2f} Down={value.z:+.2f} m")
                    else:
                        print(f"ATTITUDE: roll={math.degrees(value.roll):+.1f} pitch={math.degrees(value.pitch):+.1f} yaw={math.degrees(value.yaw):+.1f} deg")
                print()
                next_print = now + 1
        print("Received message counts:", counts)
        now = time.monotonic()
        passed = all(counts[name] >= 5 and now-latest[name][1] < 2 for name in names)
        passed = passed and now-last_heartbeat < 3
        if not passed:
            raise RuntimeError("Check incomplete: missing or stale telemetry. Keep SITL open and inspect the output.")
        print("PASS: autopilot heartbeat, global position, local position and attitude received.")
        print("No arming, mode-change, movement or landing commands were sent.")
    finally:
        connection.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Check stopped by user.")
    except (RuntimeError, OSError) as error:
        print("CHECK FAILED:", error)
        raise SystemExit(1)
