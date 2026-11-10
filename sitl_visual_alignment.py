"""SITL-only visual alignment. Manual takeoff/landing; SPACE enables control.
Dependencies beside this file: sitl_camera_preview.py, virtual_marker_pose.py.
No arming, takeoff, mode changes, descent or landing commands are sent.
"""
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from pymavlink import mavutil
from sitl_camera_preview import render_scene, body_to_ned, CAMERA_FROM_BODY
from virtual_marker_pose import estimate_from_image, WIDTH, HEIGHT

WINDOW = "SITL visual alignment - SPACE start/pause - H hide - Q stop preview"
GAIN, MAX_SPEED, TOLERANCE = 0.35, 0.15, 0.05
MIN_HEIGHT, MAX_HEIGHT = 1.5, 2.5


def guidance(result, attitude):
    """Only image pose and measured orientation determine direction/speed."""
    if result is None:
        return np.zeros(3), None
    # Camera -> vehicle -> north/east/down. The marker coordinates and vehicle
    # position used by the renderer are NOT inputs to this controller.
    target_ned = body_to_ned(*attitude) @ CAMERA_FROM_BODY.T @ result[0]
    error = float(np.linalg.norm(target_ned[:2]))
    velocity = np.zeros(3)
    if error > TOLERANCE:
        velocity[:2] = GAIN * target_ned[:2]
        speed = float(np.linalg.norm(velocity[:2]))
        if speed > MAX_SPEED:
            velocity[:2] *= MAX_SPEED / speed
    # Down velocity always zero. This requests altitude hold, not descent.
    return velocity, error


def send_velocity(link, velocity, started):
    link.mav.set_position_target_local_ned_send(
        int((time.monotonic()-started)*1000) & 0xFFFFFFFF,
        1, 1, mavutil.mavlink.MAV_FRAME_LOCAL_NED, 3527,
        0, 0, 0, float(velocity[0]), float(velocity[1]), 0,
        0, 0, 0, 0, 0)


def self_test():
    marker = np.array([0.3,0.3,0.0])
    for yaw in (0, np.pi/2, -np.pi/3):
        position = np.array([0.,0.,-2.])
        attitude = (0,0,yaw)
        for _ in range(500):
            image, _, _ = render_scene(position, attitude, marker)
            result = estimate_from_image(image)
            assert result is not None
            velocity, error = guidance(result, attitude)
            assert velocity[2] == 0 and np.linalg.norm(velocity) <= MAX_SPEED+1e-9
            position += velocity*0.1
            if error <= TOLERANCE: break
        final = float(np.linalg.norm((marker-position)[:2]))
        assert final < 0.06
        print(f"Ideal-model heading {np.degrees(yaw):+.0f} deg: final offset={final*100:.2f} cm")
    velocity, error = guidance(None,(0,0,0))
    assert np.array_equal(velocity,np.zeros(3)) and error is None
    # Check the outgoing frame, mask and velocity fields without any connection.
    class FakeMav:
        def set_position_target_local_ned_send(self,*args): self.args=args
    class FakeLink:
        def __init__(self): self.mav=FakeMav()
    fake=FakeLink()
    send_velocity(fake,np.array([0.1,-0.1,0]),time.monotonic())
    assert fake.mav.args[3:5] == (1,3527)
    assert fake.mav.args[8:11] == (0.1,-0.1,0)
    print("PASS: image-based directions, ideal-model convergence, speed limit, missing target and message fields.")
    print("These are offline checks, NOT a live ArduPilot flight test.")


def main():
    link=mavutil.mavlink_connection('udpin:127.0.0.1:14600',source_system=245,source_component=191)
    cache={}
    samples=[]
    ground=marker=None
    requested=enabled=owned=False
    visible=True
    started=time.monotonic()
    next_tick=started
    good_since=aligned_since=None
    previous_state=None
    logdir=Path(__file__).resolve().parent/'sitl_alignment_logs'
    logdir.mkdir(exist_ok=True)
    path=logdir/f'alignment_{datetime.now():%Y%m%d_%H%M%S_%f}.csv'
    fields=['elapsed_s','state','enabled','detected','height_m','estimated_horizontal_error_m',
            'render_reference_horizontal_error_m','command_sent','command_north_m_s','command_east_m_s','command_down_m_s']
    cv2.namedWindow(WINDOW,cv2.WINDOW_AUTOSIZE)
    print('Start on the ground and DISARMED. Wait for GROUND ANCHORED.')
    print('Then take off manually to 2 m in GUIDED. SPACE enables visual alignment.')
    print('SPACE again pauses; H hides marker. Land manually in Mission Planner.')
    print('Q sends zero velocity if this script owns Guided control, then exits. Q does NOT land.')
    print('Log:',path)

    def guided_and_armed(now):
        hb=cache.get('HEARTBEAT')
        return bool(hb and now-hb[1]<3 and hb[0].custom_mode==4
                    and hb[0].base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    try:
        with path.open('w',newline='',encoding='utf-8') as file:
            writer=csv.DictWriter(file,fieldnames=fields)
            writer.writeheader()
            while True:
                for _ in range(500):
                    msg=link.recv_match(blocking=False)
                    if msg is None: break
                    if msg.get_srcSystem()!=1 or msg.get_srcComponent()!=1: continue
                    kind=msg.get_type()
                    if kind not in ('HEARTBEAT','LOCAL_POSITION_NED','ATTITUDE'): continue
                    if kind=='HEARTBEAT' and msg.autopilot!=mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA: continue
                    old=cache.get(kind)
                    if kind!='HEARTBEAT' and old and msg.time_boot_ms<old[0].time_boot_ms:
                        raise RuntimeError('Telemetry clock reset. Restart on the ground after landing.')
                    now=time.monotonic()
                    cache[kind]=(msg,now)
                    if kind=='HEARTBEAT' and not requested:
                        if msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                            raise RuntimeError('Land and disarm before starting this script.')
                        for message_id in (32,30):
                            link.mav.command_long_send(1,1,mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                                                       0,message_id,100000,0,0,0,0,0)
                        requested=True
                    if kind=='LOCAL_POSITION_NED' and requested and ground is None:
                        hb=cache.get('HEARTBEAT')
                        if not hb or now-hb[1]>3: continue
                        if hb[0].base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                            raise RuntimeError('Armed before ground anchor. Land and restart.')
                        if np.linalg.norm([msg.vx,msg.vy,msg.vz])>0.15:
                            samples.clear()
                            continue
                        if samples and msg.time_boot_ms==samples[-1][0]: continue
                        samples.append((msg.time_boot_ms,[msg.x,msg.y,msg.z]))
                        if len(samples)>=20:
                            xyz=np.array([s[1] for s in samples[-20:]])
                            if np.max(np.ptp(xyz,axis=0))>0.1:
                                samples.clear()
                                continue
                            ground=np.median(xyz,axis=0)
                            marker=ground+np.array([0.3,0.3,0])
                            print('GROUND ANCHORED. Take off to 2 m using Mission Planner; then press SPACE here.')
                now=time.monotonic()
                if not requested and now-started>20: raise RuntimeError('No autopilot heartbeat on port 14600.')
                # Controller/display evaluated at no more than 10 Hz.
                if now<next_tick:
                    time.sleep(min(0.005,next_tick-now))
                    continue
                next_tick=now+0.1
                frame=np.full((HEIGHT,WIDTH),230,dtype=np.uint8)
                result=None
                height=error=reference_error=None
                attitude=None
                fresh=ground is not None and all(name in cache and now-cache[name][1]<limit for name,limit in
                    [('LOCAL_POSITION_NED',0.4),('ATTITUDE',0.4),('HEARTBEAT',3)])
                if fresh:
                    p,a=cache['LOCAL_POSITION_NED'][0],cache['ATTITUDE'][0]
                    fresh=abs(p.time_boot_ms-a.time_boot_ms)<=150
                    if fresh:
                        position=np.array([p.x,p.y,p.z])
                        attitude=(a.roll,a.pitch,a.yaw)
                        height=float(ground[2]-p.z)
                        frame,_,_=render_scene(position,attitude,marker,visible)
                        result=estimate_from_image(frame)
                        reference_error=float(np.linalg.norm((marker-position)[:2]))
                ready=guided_and_armed(now)
                velocity=np.zeros(3)
                if result is not None:
                    velocity,error=guidance(result,attitude)
                state='PAUSED - NO MOVEMENT COMMANDS'
                if ground is None: state='WAITING FOR DISARMED GROUND ANCHOR'
                elif not ready: state='WAITING FOR ARMED GUIDED HOVER'
                if enabled:
                    if not ready:
                        enabled=owned=False
                        state='CONTROL RELEASED - MODE / ARM / HEARTBEAT CHANGED'
                    elif not fresh:
                        state='TELEMETRY STALE - ZERO VELOCITY'
                        velocity[:]=0
                        good_since=aligned_since=None
                    elif not MIN_HEIGHT<=height<=MAX_HEIGHT:
                        state='HEIGHT OUTSIDE 1.5-2.5 m - ZERO VELOCITY'
                        velocity[:]=0
                        good_since=aligned_since=None
                    elif result is None:
                        state='TARGET LOST - ZERO VELOCITY'
                        velocity[:]=0
                        good_since=aligned_since=None
                    else:
                        if good_since is None: good_since=now
                        if now-good_since<0.5:
                            state='CONFIRMING TARGET - ZERO VELOCITY'
                            velocity[:]=0
                        elif error<=TOLERANCE:
                            if aligned_since is None: aligned_since=now
                            state='ALIGNED - HOLDING' if now-aligned_since>=2 else 'CONFIRMING ALIGNMENT'
                        else:
                            aligned_since=None
                            state='ALIGNING'
                sent=False
                if enabled and ready:
                    send_velocity(link,velocity,started)
                    owned=sent=True
                else:
                    velocity[:]=0
                if state!=previous_state:
                    print(f'{now-started:.1f} s | {state} | reference offset={reference_error}')
                    previous_state=state
                row=dict(elapsed_s=round(now-started,3),state=state,enabled=enabled,detected=result is not None,
                         height_m=height,estimated_horizontal_error_m=error,
                         render_reference_horizontal_error_m=reference_error,command_sent=sent,
                         command_north_m_s=velocity[0],command_east_m_s=velocity[1],command_down_m_s=0)
                writer.writerow(row)
                file.flush()
                preview=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
                if result is not None:
                    cv2.aruco.drawDetectedMarkers(preview,[result[2]],np.array([[0]],dtype=np.int32))
                cv2.drawMarker(preview,(WIDTH//2,HEIGHT//2),(255,0,0),cv2.MARKER_CROSS,16,1)
                height_text='unavailable' if height is None else f'{height:.2f} m'
                error_text='unavailable' if error is None else f'{100*error:.2f} cm'
                lines=[state,f'Height above plane: {height_text} | target band: 1.5-2.5 m',
                       f'Image-derived horizontal error: {error_text}',
                       f'Command North={velocity[0]:+.3f} East={velocity[1]:+.3f} Down=0 m/s',
                       'SPACE start/pause | H hide/show | Q stop script (does NOT land)',
                       'Manual takeoff and LAND in Mission Planner | No descent control']
                panel=np.zeros((165,WIDTH,3),dtype=np.uint8)
                for index,text in enumerate(lines):
                    cv2.putText(panel,text,(8,22+24*index),cv2.FONT_HERSHEY_SIMPLEX,0.43,(0,255,0),1,cv2.LINE_AA)
                cv2.imshow(WINDOW,np.vstack((panel,preview)))
                key=cv2.waitKey(1)&0xFF
                if key in (ord('q'),27): break
                if key==ord('h'): visible=not visible
                if key==ord(' '):
                    if enabled:
                        if guided_and_armed(time.monotonic()): send_velocity(link,np.zeros(3),started)
                        enabled=False
                        good_since=aligned_since=None
                    elif fresh and ready and result is not None and MIN_HEIGHT<=height<=MAX_HEIGHT:
                        enabled=True
                        good_since=aligned_since=None
                    else:
                        print('Not enabled: need fresh telemetry, visible marker, ARMED GUIDED and 1.5-2.5 m height.')
                if cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1: break
    finally:
        try:
            if owned and guided_and_armed(time.monotonic()):
                send_velocity(link,np.zeros(3),started)
        finally:
            link.close()
            cv2.destroyAllWindows()
        print('Log saved:',path)
        print('Script closed. Land any ongoing simulated flight using Mission Planner.')


if __name__=='__main__':
    try:
        if '--self-test' in sys.argv: self_test()
        else: main()
    except KeyboardInterrupt: print('Stopped by user.')
    except (RuntimeError,OSError) as error:
        print('STOPPED:',error)
        raise SystemExit(1)
