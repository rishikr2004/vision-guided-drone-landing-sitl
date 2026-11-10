"""SITL-only visual alignment and descent to approximately 0.8 m.
Dependencies beside this file: sitl_camera_preview.py, virtual_marker_pose.py.
Manual takeoff and landing. This sends horizontal AND downward velocity commands.
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

WINDOW = "SITL descent - SPACE start/pause - H hide - Q quit (does NOT land)"
GAIN, MAX_SPEED = 0.35, 0.15
MIN_HEIGHT, MAX_HEIGHT = 0.65, 2.5


class DescentController:
    """Image-derived target drives control; telemetry height is a guard only."""
    def __init__(self):
        self.complete = False
        self.aborted = False
        self.reset_tracking()

    def reset_tracking(self):
        self.good_since = self.aligned_since = None
        self.descending = False

    def update(self, now, result, attitude, height, fresh):
        velocity = np.zeros(3)
        error = depth = None
        if self.aborted:
            return velocity, 'HEIGHT GUARD LATCHED - LAND MANUALLY', error, depth
        if not fresh:
            self.reset_tracking()
            return velocity, 'TELEMETRY STALE - ZERO VELOCITY', error, depth
        if not MIN_HEIGHT <= height <= MAX_HEIGHT:
            self.aborted = True
            self.reset_tracking()
            return velocity, 'HEIGHT GUARD LATCHED - LAND MANUALLY', error, depth
        if self.complete:
            return velocity, 'DESCENT COMPLETE - HOLDING', error, depth
        if result is None:
            self.reset_tracking()
            return velocity, 'TARGET LOST - ZERO VELOCITY', error, depth
        target = body_to_ned(*attitude) @ CAMERA_FROM_BODY.T @ result[0]
        if not np.all(np.isfinite(target)) or target[2] <= 0:
            self.reset_tracking()
            return velocity, 'INVALID POSE - ZERO VELOCITY', error, depth
        error, depth = float(np.linalg.norm(target[:2])), float(target[2])
        if self.good_since is None:
            self.good_since = now
        if now-self.good_since < 0.5:
            return velocity, 'CONFIRMING TARGET - ZERO VELOCITY', error, depth
        # Stop band: 0.80 m target plus 0.02 m allowance. Latches until restart.
        if depth <= 0.82:
            self.complete = True
            return velocity, 'DESCENT COMPLETE - HOLDING', error, depth
        if error > 0.02:
            velocity[:2] = GAIN * target[:2]
            speed = np.linalg.norm(velocity[:2])
            if speed > MAX_SPEED:
                velocity[:2] *= MAX_SPEED/speed
        # Hysteresis: align within 5 cm; interrupt descent above 8 cm.
        if error > 0.08:
            self.descending = False
            self.aligned_since = None
        if not self.descending:
            if error <= 0.05:
                if self.aligned_since is None:
                    self.aligned_since = now
                if now-self.aligned_since >= 2:
                    self.descending = True
            else:
                self.aligned_since = None
        if self.descending:
            # Positive Down means descent. Slow down near the stopping height.
            velocity[2] = min(0.10, max(0.02, 0.4*(depth-0.80)))
            state = 'DESCENDING'
        else:
            state = 'CONFIRMING ALIGNMENT' if error <= 0.05 else 'ALIGNING'
        return velocity, state, error, depth


def send_velocity(link, velocity, started):
    link.mav.set_position_target_local_ned_send(
        int((time.monotonic()-started)*1000) & 0xFFFFFFFF,
        1, 1, mavutil.mavlink.MAV_FRAME_LOCAL_NED, 3527,
        0, 0, 0, float(velocity[0]), float(velocity[1]), float(velocity[2]),
        0, 0, 0, 0, 0)


def self_test():
    for yaw in (0, np.pi/2, -np.pi/3):
        controller = DescentController()
        marker = np.array([0.3,0.3,0.])
        position = np.array([0.,0.,-2.])
        attitude = (0,0,yaw)
        for step in range(1500):
            image, _, _ = render_scene(position, attitude, marker)
            result = estimate_from_image(image)
            assert result is not None
            v, state, _, _ = controller.update(step*0.1,result,attitude,-position[2],True)
            assert np.linalg.norm(v[:2]) <= MAX_SPEED+1e-9
            assert 0 <= v[2] <= 0.10
            position += v*0.1
            if controller.complete:
                break
        assert controller.complete and 0.76 < -position[2] < 0.86
        assert np.linalg.norm((marker-position)[:2]) < 0.06
        v, _, _, _ = controller.update(200,None,attitude,-position[2],True)
        assert np.all(v == 0)  # Completion remains latched.
        print(f'Ideal model yaw={np.degrees(yaw):+.0f}: height={-position[2]:.3f} m')
    c = DescentController()
    pose = (np.array([0.,0.,2.]),0,None)
    for t in (0,0.6,2.7):
        v, _, _, _ = c.update(t,pose,(0,0,0),2,True)
    assert v[2] > 0
    v, state, _, _ = c.update(3,None,(0,0,0),1.8,True)
    assert np.all(v==0) and 'LOST' in state
    v, _, _, _ = c.update(4,pose,(0,0,0),1.8,True)
    assert np.all(v==0)  # Reacquire before resuming.
    c.update(4.6,pose,(0,0,0),1.8,True)
    v, _, _, _ = c.update(6.7,pose,(0,0,0),1.8,True)
    assert v[2]>0
    displaced=(np.array([0.2,0,1.8]),0,None)
    v, _, _, _ = c.update(6.8,displaced,(0,0,0),1.8,True)
    assert v[2]==0 and np.linalg.norm(v[:2])>0
    v, _, _, _ = c.update(7,pose,(0,0,0),1.8,False)
    assert np.all(v==0)
    c.reset_tracking()
    assert c.good_since is None and not c.descending
    v, _, _, _ = c.update(8,pose,(0,0,0),0.6,True)
    assert c.aborted and np.all(v==0)
    v, _, _, _ = c.update(9,pose,(0,0,0),2,True)
    assert np.all(v==0) and c.aborted
    class FakeMav:
        def set_position_target_local_ned_send(self,*args): self.args=args
    class FakeLink:
        def __init__(self): self.mav=FakeMav()
    fake=FakeLink()
    send_velocity(fake,np.array([0.1,-0.1,0.08]),time.monotonic())
    assert fake.mav.args[3:5]==(1,3527)
    assert fake.mav.args[8:11]==(0.1,-0.1,0.08)
    print('PASS: ideal descent, speed limits, loss/reacquisition, realignment, stale telemetry, guard and message fields.')
    print('Offline checks only; live SITL response still needs testing.')


def main():
    link=mavutil.mavlink_connection('udpin:127.0.0.1:14600',source_system=245,source_component=191)
    cache={}
    samples=[]
    ground=marker=None
    requested=enabled=owned=False
    visible=True
    started=time.monotonic()
    next_tick=started
    controller=DescentController()
    previous_state=None
    logdir=Path(__file__).resolve().parent/'sitl_descent_logs'
    logdir.mkdir(exist_ok=True)
    path=logdir/f'descent_{datetime.now():%Y%m%d_%H%M%S_%f}.csv'
    fields=['elapsed_s','state','enabled','detected','height_m','estimated_horizontal_error_m',
            'render_reference_horizontal_error_m','command_sent','command_north_m_s','command_east_m_s','command_down_m_s',
            'north_m','east_m','down_m','north_from_anchor_m','east_from_anchor_m',
            'measured_vn_m_s','measured_ve_m_s','measured_vd_m_s','measured_horizontal_speed_m_s',
            'position_boot_ms','position_age_s','attitude_age_s','telemetry_fresh',
            'estimated_vertical_distance_m','marker_visible','armed','custom_mode']
    cv2.namedWindow(WINDOW,cv2.WINDOW_AUTOSIZE)
    print('Start on the ground and DISARMED. Wait for GROUND ANCHORED.')
    print('Then take off manually to 2 m in GUIDED. SPACE enables alignment, then descent to about 0.8 m.')
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
                height=error=reference_error=depth=None
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
                state='PAUSED - ZERO VELOCITY' if owned else 'PAUSED - NO COMMANDS'
                if ground is None: state='WAITING FOR DISARMED GROUND ANCHOR'
                elif not ready: state='WAITING FOR ARMED GUIDED HOVER'
                if not ready and (enabled or owned):
                    enabled=owned=False
                    controller.reset_tracking()
                    state='CONTROL RELEASED - MODE / ARM / HEARTBEAT CHANGED'
                if enabled and ready:
                    velocity,state,error,depth=controller.update(now,result,attitude,height,fresh)
                sent=False
                if (enabled or owned) and ready:
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
                         command_north_m_s=velocity[0],command_east_m_s=velocity[1],command_down_m_s=float(velocity[2]))
                p_entry=cache.get('LOCAL_POSITION_NED')
                a_entry=cache.get('ATTITUDE')
                hb_entry=cache.get('HEARTBEAT')
                row.update(telemetry_fresh=bool(fresh),estimated_vertical_distance_m=depth,
                           marker_visible=visible,
                           armed=bool(hb_entry and hb_entry[0].base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED),
                           custom_mode=hb_entry[0].custom_mode if hb_entry else None,
                           position_age_s=now-p_entry[1] if p_entry else None,
                           attitude_age_s=now-a_entry[1] if a_entry else None)
                # Log last received telemetry even when stale; age/fresh columns identify it.
                if p_entry:
                    p=p_entry[0]
                    row.update(north_m=p.x,east_m=p.y,down_m=p.z,
                               north_from_anchor_m=p.x-ground[0] if ground is not None else None,
                               east_from_anchor_m=p.y-ground[1] if ground is not None else None,
                               measured_vn_m_s=p.vx,measured_ve_m_s=p.vy,measured_vd_m_s=p.vz,
                               measured_horizontal_speed_m_s=float(np.hypot(p.vx,p.vy)),
                               position_boot_ms=p.time_boot_ms)
                writer.writerow(row)
                file.flush()
                preview=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
                if result is not None:
                    cv2.aruco.drawDetectedMarkers(preview,[result[2]],np.array([[0]],dtype=np.int32))
                cv2.drawMarker(preview,(WIDTH//2,HEIGHT//2),(255,0,0),cv2.MARKER_CROSS,16,1)
                height_text='unavailable' if height is None else f'{height:.2f} m'
                error_text='unavailable' if error is None else f'{100*error:.2f} cm'
                lines=[state,f'Height above plane: {height_text} | stop near 0.8 m',
                       f'Image-derived horizontal error: {error_text}',
                       f'Command North={velocity[0]:+.3f} East={velocity[1]:+.3f} Down={velocity[2]:+.3f} m/s',
                       'SPACE start/pause | H hide/show | Q stop script (does NOT land)',
                       'Manual takeoff and LAND | 0.8 m hover is NOT touchdown']
                panel=np.zeros((165,WIDTH,3),dtype=np.uint8)
                for index,text in enumerate(lines):
                    cv2.putText(panel,text,(8,22+24*index),cv2.FONT_HERSHEY_SIMPLEX,0.43,(0,255,0),1,cv2.LINE_AA)
                cv2.imshow(WINDOW,np.vstack((panel,preview)))
                key=cv2.waitKey(1)&0xFF
                if key in (ord('q'),27): break
                if key==ord('h'): visible=not visible
                if key==ord(' '):
                    if enabled:
                        enabled=False
                        controller.reset_tracking()
                    elif (fresh and ready and result is not None and not controller.complete and not controller.aborted
                          and (MIN_HEIGHT if owned else 1.5)<=height<=MAX_HEIGHT):
                        enabled=True
                        controller.reset_tracking()
                    else:
                        print('Not enabled: need fresh target, ARMED GUIDED, initial height 1.5-2.5 m. Completed/guarded runs require landing and restart.')
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
