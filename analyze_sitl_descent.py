"""Offline review of SITL descent logs; never connects to a vehicle."""
import argparse
import csv
import math
from pathlib import Path
from statistics import mean


def fresh_samples(rows):
    """Use each fresh position timestamp once, avoiding repeated cached samples."""
    seen = set()
    result = []
    for r in rows:
        stamp = r['position_boot_ms']
        if r['telemetry_fresh'] != 'True' or not stamp or stamp in seen:
            continue
        seen.add(stamp)
        result.append(r)
    return result


def motion(rows):
    if not rows:
        return ['No fresh position samples.']
    n = [float(r['north_m']) for r in rows]
    e = [float(r['east_m']) for r in rows]
    h = [float(r['height_m']) for r in rows]
    speed = [math.hypot(float(r['measured_vn_m_s']), float(r['measured_ve_m_s'])) for r in rows]
    vd = [float(r['measured_vd_m_s']) for r in rows]
    offset = [math.hypot(x-n[0], y-e[0]) for x, y in zip(n,e)]
    path = sum(math.hypot(b-a,d-c) for a,b,c,d in zip(n,n[1:],e,e[1:]))
    return [f'Fresh unique position samples: {len(rows)}',
            f'North change: {(n[-1]-n[0])*100:+.3f} cm; East change: {(e[-1]-e[0])*100:+.3f} cm',
            f'Horizontal net displacement: {offset[-1]*100:.3f} cm',
            f'Maximum sampled horizontal displacement from window start: {max(offset)*100:.3f} cm',
            f'Sampled horizontal path length: {path*100:.3f} cm',
            f'Horizontal speed first / last / mean / maximum: {speed[0]:.4f} / {speed[-1]:.4f} / {mean(speed):.4f} / {max(speed):.4f} m/s',
            f'Down velocity first / last / min / max: {vd[0]:+.4f} / {vd[-1]:+.4f} / {min(vd):+.4f} / {max(vd):+.4f} m/s',
            f'Height first / last / min / max: {h[0]:.4f} / {h[-1]:.4f} / {min(h):.4f} / {max(h):.4f} m',
            f'Height change: {(h[-1]-h[0])*100:+.3f} cm; height span: {(max(h)-min(h))*100:.3f} cm']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_file',type=Path)
    parser.add_argument('--output-dir',type=Path)
    args = parser.parse_args()
    with args.csv_file.open(newline='',encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit('Empty CSV.')
    times = [float(r['elapsed_s']) for r in rows]
    if any(b<=a for a,b in zip(times,times[1:])):
        raise SystemExit('Non-increasing log timestamps; inspect before analysing.')
    lines = [f'Source: {args.csv_file.resolve()}', f'Rows: {len(rows)}']
    groups=[]
    start=None
    for i,r in enumerate(rows):
        lost=r['state']=='TARGET LOST - ZERO VELOCITY'
        if lost and start is None:
            start=i
        if start is not None and (not lost or i==len(rows)-1):
            end=i if not lost else i+1
            groups.append((start,end))
            start=None
    lines.append(f'Target-loss intervals: {len(groups)}')
    for k,(start,end) in enumerate(groups,1):
        group=rows[start:end]
        t0=times[start]
        t1=times[end] if end<len(rows) else times[end-1]
        sent=[r for r in group if r['command_sent']=='True']
        nonzero=sum(any(abs(float(r[c]))>1e-9 for c in
                       ['command_north_m_s','command_east_m_s','command_down_m_s']) for r in sent)
        fresh=fresh_samples(group)
        lines += ['',f'HIDDEN INTERVAL {k}: {t0:.3f} to {t1:.3f} s; duration {t1-t0:.3f} s',
                  f'Rows: {len(group)}; commands sent: {len(sent)}; nonzero commands: {nonzero}',
                  f'Visible-marker rows: {sum(r["marker_visible"]=="True" for r in group)}; detections: {sum(r["detected"]=="True" for r in group)}',
                  f'Stale rows excluded: {sum(r["telemetry_fresh"]!="True" for r in group)}',
                  'WHOLE HIDDEN WINDOW:']
        lines += motion(fresh)
        late=fresh_samples([r for r in group if float(r['elapsed_s'])>=t0+2])
        lines += ['AFTER FIRST 2 SECONDS (diagnostic window, not proof of settling):']
        lines += motion(late)
        if end<len(rows):
            lines.append(f'Next state: {rows[end]["state"]}')
    completed=fresh_samples([r for r in rows if r['state']=='DESCENT COMPLETE - HOLDING'])
    lines += ['', 'COMPLETION HOLD:'] + motion(completed)
    lines += ['', 'LIMITATIONS:',
              'Intervals follow logged controller states; actual key-press times are not recorded.',
              'Motion starts at the first logged hidden sample, not the exact instant the marker disappeared.',
              'Only fresh telemetry is used, with repeated position timestamps removed within each window.',
              'Net displacement is endpoint separation; sampled path length sums intermediate movements.',
              'Sampled path length is affected by estimator noise and does not capture between-sample motion.',
              'Positive Down velocity means descent; negative height change means a lower final height.',
              'Commands are requests. Measured velocities are SITL estimates, not physical measurements.',
              'A zero-velocity request can be followed by braking movement; no automatic pass threshold is assumed.',
              'Completion hold is above ground, not a completed landing.']
    report='\n'.join(lines)
    print(report)
    out=args.output_dir or args.csv_file.parent.parent/'sitl_descent_reports'
    out.mkdir(parents=True,exist_ok=True)
    destination=out/(args.csv_file.stem+'_review.txt')
    destination.write_text(report+'\n',encoding='utf-8')
    print(f'\nReport saved: {destination.resolve()}')


if __name__=='__main__':
    main()
