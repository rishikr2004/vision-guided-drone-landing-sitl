"""SITL-only visual alignment and descent to approximately 0.8 m.
Dependencies beside this file: sitl_camera_preview.py, virtual_marker_pose.py.
Manual takeoff; automatic LAND handover after low-hover checks. SITL ONLY.
"""
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from pymavlink import mavutil
from sitl_camera_preview import render_scene, body_to_ned, CAMERA_FROM_BODY
from virtual_marker_pose import estimate_from_image, WIDTH, HEIGHT

WINDOW = "SITL LAND handover - SPACE start/pause - H hide - Q exit"
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


def descent_self_test():
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


class LandHandover:
    def __init__(self):
        self.phase = 'APPROACH'
        self.stable_since = None
        self.finish_since = None
        self.request_s = self.mode_s = self.ground_s = self.disarmed_s = None
        self.success_s = None
        self.reason = ''
        self.events = []
        self.final_offset_m = None

    def event(self, now, name):
        self.events.append({'elapsed_s': round(now, 3), 'event': name})

    def fail(self, now, reason):
        self.phase, self.reason = 'FAILED', reason
        self.event(now, reason)

    def update(self, now, *, enabled, complete, fresh, detected, error, depth,
               height, horizontal_speed, down_speed, mode, armed, hb_age,
               hb_received_s, landed_state, ext_age, ext_received_s,
               position_received_s, reference_offset, ack_result=None):
        """Returns display state and whether to send ONE LAND request this tick."""
        if self.phase == 'SUCCESS':
            return 'SUCCESS - ON GROUND AND DISARMED', False
        if self.phase == 'FAILED':
            return 'HANDOVER STOPPED - CHECK MISSION PLANNER', False
        if self.phase == 'APPROACH':
            gate = (enabled and complete and fresh and detected and mode == 4
                    and armed and hb_age < 3 and ext_age < 1
                    and landed_state == 2  # MAV_LANDED_STATE_IN_AIR
                    and error is not None and math.isfinite(error) and error <= 0.05
                    and depth is not None and 0.70 <= depth <= 0.90
                    and height is not None and 0.70 <= height <= 0.90
                    and horizontal_speed is not None and horizontal_speed <= 0.05
                    and down_speed is not None and abs(down_speed) <= 0.03)
            if not gate:
                self.stable_since = None
                return 'LOW HOVER - WAITING FOR HANDOVER CHECKS', False
            if self.stable_since is None:
                self.stable_since = now
            remaining = max(0, 3-(now-self.stable_since))
            if remaining > 0:
                return f'CHECKING LOW HOVER - {remaining:.1f} s', False
            self.phase, self.request_s = 'WAIT_LAND', now
            self.event(now, 'LAND_REQUESTED')
            return 'LAND REQUESTED - WAITING FOR HEARTBEAT', True

        # After requesting LAND, never return to approach or issue Guided velocity.
        hb_fresh = hb_age < 3 and hb_received_s > self.request_s
        if self.phase == 'WAIT_LAND':
            if hb_fresh and mode == 9:
                self.phase, self.mode_s = 'LANDING', now
                self.event(now, 'LAND_MODE_CONFIRMED')
            elif ack_result is not None and ack_result not in (0, 5):
                self.fail(now, f'LAND request rejected: ACK result {ack_result}')
            elif hb_fresh and (mode != 4 or not armed):
                self.fail(now, 'Mode or arming changed before LAND confirmation')
            elif now-self.request_s > 8:
                self.fail(now, 'LAND mode not confirmed within 8 seconds')
            if self.phase != 'LANDING':
                return ('HANDOVER STOPPED - CHECK MISSION PLANNER' if self.phase == 'FAILED'
                        else 'WAITING FOR LAND MODE CONFIRMATION'), False

        if now-self.request_s > 90:
            self.fail(now, 'Landing/disarming not confirmed within 90 seconds')
            return 'TIMEOUT - CHECK MISSION PLANNER', False
        if hb_fresh and mode != 9:
            self.fail(now, 'Mode changed after handover; operator control respected')
            return 'MODE CHANGED - CONTROL RELEASED', False
        if not hb_fresh or ext_age >= 1:
            self.finish_since = None
            return 'LANDING - WAITING FOR FRESH STATUS', False
        on_ground = landed_state == 1 and ext_received_s > self.mode_s
        if on_ground and self.ground_s is None:
            self.ground_s = now
            self.event(now, 'ON_GROUND_CONFIRMED')
        if not armed and self.disarmed_s is None:
            self.disarmed_s = now
            self.event(now, 'DISARMED_CONFIRMED')
        final_valid = (on_ground and not armed and fresh
                       and self.disarmed_s is not None
                       and position_received_s > self.disarmed_s
                       and reference_offset is not None and math.isfinite(reference_offset))
        if final_valid:
            if self.finish_since is None:
                self.finish_since = now
            if now-self.finish_since >= 2:
                self.phase, self.success_s = 'SUCCESS', now
                self.final_offset_m = reference_offset
                self.event(now, 'SUCCESS_ON_GROUND_DISARMED_WITH_FRESH_POSITION')
                return 'SUCCESS - ON GROUND AND DISARMED', False
        else:
            self.finish_since = None
        return 'LANDING - MONITORING GROUND AND DISARMING', False


def handover_self_test():
    def sample(now, **changes):
        data = dict(enabled=True, complete=True, fresh=True, detected=True,
                    error=0.02, depth=0.815, height=0.815, horizontal_speed=0.025,
                    down_speed=0.001, mode=4, armed=True, hb_age=0.1,
                    hb_received_s=now-0.1, landed_state=2, ext_age=0.05,
                    ext_received_s=now-0.05, position_received_s=now-0.02,
                    reference_offset=0.03, ack_result=None)
        data.update(changes)
        return data
    # Loss, pause, speed, alignment, stale telemetry and height block/reset the gate.
    for bad in [dict(detected=False), dict(enabled=False), dict(fresh=False),
                dict(horizontal_speed=0.1), dict(down_speed=0.06), dict(error=0.08),
                dict(height=1.3), dict(depth=1.3), dict(ext_age=2), dict(landed_state=0)]:
        c = LandHandover()
        c.update(0, **sample(0))
        assert not c.update(2, **sample(2, **bad))[1]
        assert not c.update(3, **sample(3))[1]
        assert c.update(6.1, **sample(6.1))[1]
    c = LandHandover()
    c.update(0, **sample(0))
    assert c.update(3.1, **sample(3.1))[1]
    c.update(3.2, **sample(3.2, ack_result=0))
    assert c.phase == 'WAIT_LAND'  # ACK alone is not mode confirmation.
    c.update(4, **sample(4, mode=9))
    assert c.phase == 'LANDING'
    c.update(5, **sample(5, mode=9, armed=False))
    assert c.phase != 'SUCCESS'  # Disarmed alone is not ground confirmation.
    c.update(6, **sample(6, mode=9, armed=False, landed_state=1))
    c.update(7, **sample(7, mode=9, armed=False, landed_state=1, fresh=False))
    c.update(8, **sample(8, mode=9, armed=False, landed_state=1))
    c.update(10.1, **sample(10.1, mode=9, armed=False, landed_state=1))
    assert c.phase == 'SUCCESS' and c.final_offset_m == 0.03
    for kind in ('timeout', 'reject', 'override', 'landing_timeout'):
        c = LandHandover()
        c.update(0, **sample(0)); c.update(3.1, **sample(3.1))
        if kind == 'timeout': c.update(12, **sample(12))
        elif kind == 'reject': c.update(4, **sample(4, ack_result=2))
        elif kind == 'override':
            c.update(4, **sample(4, mode=9)); c.update(5, **sample(5, mode=5))
        else:
            c.update(4, **sample(4, mode=9)); c.update(95, **sample(95, mode=9))
        assert c.phase == 'FAILED'
    print('PASS: handover gates, reacquisition, ACK versus mode, ground/disarm confirmation, timeouts and override.')


def request_land(link):
    link.mav.command_long_send(1, 1, mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                              0, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                              9, 0, 0, 0, 0, 0)


def self_test():
    descent_self_test()
    handover_self_test()
    class FakeMav:
        def command_long_send(self, *args): self.args = args
    class FakeLink:
        def __init__(self): self.mav = FakeMav()
    fake = FakeLink()
    request_land(fake)
    assert fake.mav.args == (1, 1, 176, 0, 1, 9, 0, 0, 0, 0, 0)
    print('PASS: LAND command fields. Offline checks only, not a live flight test.')


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
    handover=LandHandover()
    ack_result=None
    previous_state=None
    logdir=Path(__file__).resolve().parent/'sitl_landing_logs'
    logdir.mkdir(exist_ok=True)
    path=logdir/f'landing_{datetime.now():%Y%m%d_%H%M%S_%f}.csv'
    fields=['elapsed_s','state','enabled','detected','height_m','estimated_horizontal_error_m',
            'render_reference_horizontal_error_m','command_sent','command_north_m_s','command_east_m_s','command_down_m_s',
            'north_m','east_m','down_m','north_from_anchor_m','east_from_anchor_m',
            'measured_vn_m_s','measured_ve_m_s','measured_vd_m_s','measured_horizontal_speed_m_s',
            'position_boot_ms','position_age_s','attitude_age_s','telemetry_fresh',
            'estimated_vertical_distance_m','marker_visible','armed','custom_mode',
            'handover_phase','land_request_sent','land_request_s','land_mode_confirmed_s',
            'on_ground_confirmed_s','disarmed_confirmed_s','landing_success',
            'landed_state','extended_state_age_s','land_ack_result',
            'final_offset_m','marker_north_m','marker_east_m']
    cv2.namedWindow(WINDOW,cv2.WINDOW_AUTOSIZE)
    print('Start on the ground and DISARMED. Wait for GROUND ANCHORED.')
    print('Then take off manually to 2 m in GUIDED. SPACE enables alignment, then descent to about 0.8 m.')
    print('SPACE pauses BEFORE handover; H hides marker. After handover ArduPilot lands.')
    print('Q exits; before handover it requests zero velocity. After handover LAND continues.')
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
                    if kind not in ('HEARTBEAT','LOCAL_POSITION_NED','ATTITUDE','EXTENDED_SYS_STATE','COMMAND_ACK'): continue
                    if kind=='HEARTBEAT' and msg.autopilot!=mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA: continue
                    old=cache.get(kind)
                    if kind in ('LOCAL_POSITION_NED','ATTITUDE') and old and msg.time_boot_ms<old[0].time_boot_ms:
                        raise RuntimeError('Telemetry clock reset. Restart on the ground after landing.')
                    now=time.monotonic()
                    if kind == 'COMMAND_ACK':
                        if (handover.request_s is not None and msg.command == mavutil.mavlink.MAV_CMD_DO_SET_MODE
                                and getattr(msg, 'target_system', 0) in (0, 245)
                                and getattr(msg, 'target_component', 0) in (0, 191)):
                            ack_result = int(msg.result)
                            handover.event(now-started, f'LAND_COMMAND_ACK_{ack_result}')
                    cache[kind]=(msg,now)
                    if kind=='HEARTBEAT' and not requested:
                        if msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                            raise RuntimeError('Land and disarm before starting this script.')
                        for message_id in (32,30,245):
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
                request_this_tick=False
                monitor_this_tick=handover.phase!='APPROACH'
                if not monitor_this_tick:
                    if not ready and (enabled or owned):
                        enabled=owned=False
                        controller.reset_tracking()
                        handover.stable_since=None
                        state='CONTROL RELEASED - MODE / ARM / HEARTBEAT CHANGED'
                    if enabled and ready:
                        velocity,state,error,depth=controller.update(now,result,attitude,height,fresh)
                    # Completed descent latches zero velocity. Recompute pose for the handover gate.
                    if controller.complete and fresh and result is not None:
                        target=body_to_ned(*attitude) @ CAMERA_FROM_BODY.T @ result[0]
                        if np.all(np.isfinite(target)):
                            error=float(np.linalg.norm(target[:2]))
                            depth=float(target[2])
                hb=cache.get('HEARTBEAT')
                ext=cache.get('EXTENDED_SYS_STATE')
                pe=cache.get('LOCAL_POSITION_NED')
                status,request_this_tick=handover.update(
                    now-started, enabled=enabled,complete=controller.complete and not controller.aborted,
                    fresh=bool(fresh),detected=result is not None,error=error,depth=depth,height=height,
                    horizontal_speed=float(np.hypot(pe[0].vx,pe[0].vy)) if pe else None,
                    down_speed=pe[0].vz if pe else None,
                    mode=hb[0].custom_mode if hb else None,
                    armed=bool(hb and hb[0].base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED),
                    hb_age=now-hb[1] if hb else float('inf'),
                    hb_received_s=hb[1]-started if hb else -1,
                    landed_state=ext[0].landed_state if ext else None,
                    ext_age=now-ext[1] if ext else float('inf'),
                    ext_received_s=ext[1]-started if ext else -1,
                    position_received_s=pe[1]-started if pe else -1,
                    reference_offset=reference_error,ack_result=ack_result)
                if monitor_this_tick or request_this_tick or (enabled and controller.complete):
                    state=status
                sent=False
                if request_this_tick:
                    # Send a final Guided stop BEFORE requesting LAND, never afterwards.
                    velocity[:]=0
                    send_velocity(link,velocity,started)
                    sent=True
                    enabled=owned=False
                    request_land(link)
                elif handover.phase=='APPROACH' and (enabled or owned) and ready:
                    send_velocity(link,velocity,started)
                    owned=sent=True
                else:
                    velocity[:]=0
                if state!=previous_state:
                    print(f'{now-started:.1f} s | {state} | reference offset={reference_error}')
                    if handover.phase=='FAILED':
                        print('Reason:',handover.reason)
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
                row.update(handover_phase=handover.phase,land_request_sent=request_this_tick,
                           land_request_s=handover.request_s,land_mode_confirmed_s=handover.mode_s,
                           on_ground_confirmed_s=handover.ground_s,disarmed_confirmed_s=handover.disarmed_s,
                           landing_success=handover.phase=='SUCCESS',land_ack_result=ack_result,
                           landed_state=ext[0].landed_state if ext else None,
                           extended_state_age_s=now-ext[1] if ext else None,
                           final_offset_m=handover.final_offset_m,
                           marker_north_m=float(marker[0]) if marker is not None else None,
                           marker_east_m=float(marker[1]) if marker is not None else None)
                writer.writerow(row)
                file.flush()
                if handover.phase=='SUCCESS':
                    print(f'SUCCESS: ON GROUND and DISARMED confirmed. Final simulated offset: {handover.final_offset_m*100:.2f} cm')
                    break
                preview=cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
                if result is not None:
                    cv2.aruco.drawDetectedMarkers(preview,[result[2]],np.array([[0]],dtype=np.int32))
                cv2.drawMarker(preview,(WIDTH//2,HEIGHT//2),(255,0,0),cv2.MARKER_CROSS,16,1)
                height_text='unavailable' if height is None else f'{height:.2f} m'
                error_text='unavailable' if error is None else f'{100*error:.2f} cm'
                lines=[state,f'Height above plane: {height_text} | stop near 0.8 m',
                       f'Image-derived horizontal error: {error_text}',
                       f'Command North={velocity[0]:+.3f} East={velocity[1]:+.3f} Down={velocity[2]:+.3f} m/s',
                       'SPACE pause before handover | H hide/show | Q exit',
                       f'LAND phase: {handover.phase} | After handover: ArduPilot controls landing']
                panel=np.zeros((165,WIDTH,3),dtype=np.uint8)
                for index,text in enumerate(lines):
                    cv2.putText(panel,text,(8,22+24*index),cv2.FONT_HERSHEY_SIMPLEX,0.43,(0,255,0),1,cv2.LINE_AA)
                cv2.imshow(WINDOW,np.vstack((panel,preview)))
                key=cv2.waitKey(1)&0xFF
                if key in (ord('q'),27): break
                if key==ord('h'): visible=not visible
                if key==ord(' '):
                    if handover.phase!='APPROACH':
                        print('LAND handover already requested. Use Mission Planner for flight-mode control.')
                        continue
                    if enabled:
                        enabled=False
                        controller.reset_tracking()
                        handover.stable_since=None
                    elif (fresh and ready and result is not None and not controller.aborted
                          and (MIN_HEIGHT if owned else 1.5)<=height<=MAX_HEIGHT):
                        enabled=True
                        controller.reset_tracking()
                    else:
                        print('Not enabled: need fresh target, ARMED GUIDED, initial height 1.5-2.5 m. Guarded runs require landing and restart.')
                if cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1: break
    finally:
        try:
            if handover.phase=='APPROACH' and owned and guided_and_armed(time.monotonic()):
                send_velocity(link,np.zeros(3),started)
        finally:
            link.close()
            cv2.destroyAllWindows()
        summary = dict(success=handover.phase=='SUCCESS',phase=handover.phase,
                       reason=handover.reason or ('Confirmed landing and disarming' if handover.phase=='SUCCESS' else 'Script exited before confirmed completion'),
                       land_requested_s=handover.request_s,land_mode_confirmed_s=handover.mode_s,
                       on_ground_confirmed_s=handover.ground_s,disarmed_confirmed_s=handover.disarmed_s,
                       success_s=handover.success_s,final_horizontal_offset_m=handover.final_offset_m,
                       offset_definition='Last fresh local-position horizontal distance to the fixed simulated marker at success confirmation',
                       landing_method='Vision-guided approach, then ArduPilot LAND; no vision correction after handover',
                       events=handover.events,csv_file=str(path))
        summary_path=path.with_suffix('.json')
        summary_path.write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print('Summary saved:',summary_path)
        print('Log saved:',path)
        if handover.phase=='SUCCESS':
            print('Confirmed flight complete. No manual landing is needed.')
        else:
            print('Script closed without confirmed success. Check Mission Planner and land any ongoing flight.')


if __name__=='__main__':
    try:
        if '--self-test' in sys.argv: self_test()
        else: main()
    except KeyboardInterrupt: print('Stopped by user.')
    except (RuntimeError,OSError) as error:
        print('STOPPED:',error)
        raise SystemExit(1)
