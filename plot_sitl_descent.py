"""Plot a saved SITL flight. No vehicle connection or flight commands."""
import argparse
import csv
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_file', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with args.csv_file.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit('No readings in this file.')
    t = [float(r['elapsed_s']) for r in rows]
    if any(b <= a for a, b in zip(t, t[1:])):
        raise SystemExit('Log times are not increasing.')

    # Find contiguous target-loss intervals, ending at the next state.
    intervals = []
    start = None
    for i, row in enumerate(rows):
        lost = row['state'] == 'TARGET LOST - ZERO VELOCITY'
        if lost and start is None:
            start = i
        if start is not None and (not lost or i == len(rows)-1):
            end = i if not lost else i+1
            intervals.append((start, end))
            start = None

    def values(column, command=False):
        output = []
        seen = set()
        for r in rows:
            valid = r['command_sent'] == 'True' if command else r['telemetry_fresh'] == 'True'
            stamp = r['position_boot_ms']
            if not command:
                valid = valid and bool(stamp) and stamp not in seen
                if valid:
                    seen.add(stamp)
            output.append(float(r[column]) if valid and r[column] else math.nan)
        return output

    # Focus the time plots on control, plus a short margin at each end.
    active = [i for i,r in enumerate(rows) if r['enabled']=='True' and r['command_sent']=='True']
    if not active:
        raise SystemExit('No active control found.')
    complete = next((i for i,r in enumerate(rows) if r['state']=='DESCENT COMPLETE - HOLDING'), None)
    left = max(t[0], t[active[0]]-2)
    right = min(t[-1], t[complete]+5) if complete is not None else min(t[-1], t[active[-1]]+2)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    height_ax, speed_ax, track_ax, shift_ax = axes.flat
    fig.suptitle('SITL controlled descent — recorded flight', fontsize=16)
    height_ax.plot(t, values('height_m'), color='royalblue', label='Reported height')
    height_ax.axhline(0.8, color='gray', linestyle='--', label='Nominal 0.8 m target')
    height_ax.set(title='Height during alignment and descent', ylabel='Height above anchored plane (m)')
    speed_ax.step(t, values('command_down_m_s', True), where='post', color='darkorange', label='Requested Down velocity')
    speed_ax.plot(t, values('measured_vd_m_s'), color='royalblue', label='Reported Down velocity')
    speed_ax.axhline(0, color='gray', linewidth=0.7)
    speed_ax.set(title='Downward velocity — positive means descending', ylabel='Velocity (m/s)')

    colors = ['#b35b00', '#6b46a5', '#16806a']
    for k, (start, end) in enumerate(intervals):
        color = colors[k % len(colors)]
        finish = t[end] if end < len(rows) else t[end-1]
        for ax in (height_ax, speed_ax):
            ax.axvspan(t[start], finish, color=color, alpha=0.15, label=f'Hidden {k+1}')
        # Retain fresh, unique position samples inside this interval only.
        group = []
        seen = set()
        for r in rows[start:end]:
            stamp = r['position_boot_ms']
            if r['telemetry_fresh']=='True' and stamp and stamp not in seen:
                group.append(r)
                seen.add(stamp)
        if not group:
            continue
        n0, e0 = float(group[0]['north_m']), float(group[0]['east_m'])
        north = [(float(r['north_m'])-n0)*100 for r in group]
        east = [(float(r['east_m'])-e0)*100 for r in group]
        elapsed = [float(r['elapsed_s'])-t[start] for r in group]
        shift = [math.hypot(n,e) for n,e in zip(north,east)]
        track_ax.plot(east, north, color=color, label=f'Hidden {k+1}')
        track_ax.scatter(east[-1], north[-1], marker='x', s=65, color=color)
        shift_ax.plot(elapsed, shift, color=color, label=f'Hidden {k+1}')
    track_ax.scatter(0, 0, color='green', s=45, label='Each interval starts here', zorder=5)
    track_ax.set(title='Horizontal movement while hidden (× = end)', xlabel='East change (cm)', ylabel='North change (cm)')
    track_ax.set_aspect('equal', adjustable='datalim')
    shift_ax.set(title='Horizontal displacement from each hidden start', xlabel='Seconds since first hidden row', ylabel='Displacement (cm)')
    for ax in (height_ax, speed_ax):
        ax.set_xlim(left, right)
        ax.set_xlabel('Seconds since recording started')
    # Scale the vertical axes to the displayed window, not off-screen takeoff/landing.
    for ax, columns in ((height_ax, [('height_m', False)]),
                        (speed_ax, [('measured_vd_m_s', False), ('command_down_m_s', True)])):
        visible_values = [v for column, command in columns for ti, v in zip(t, values(column, command))
                          if left <= ti <= right and math.isfinite(v)]
        visible_values.append(0.8 if ax is height_ax else 0.0)
        lo, hi = min(visible_values), max(visible_values)
        margin = max((hi-lo)*0.12, 0.01)
        ax.set_ylim(lo-margin, hi+margin)
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.supxlabel('Synthetic camera and SITL telemetry; sampled estimates, not physical landing accuracy.', fontsize=10)
    output_dir = args.output_dir or args.csv_file.parent.parent/'sitl_descent_plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir/(args.csv_file.stem+'_overview.png')
    fig.savefig(output, dpi=180)
    print(f'Hidden intervals plotted: {len(intervals)}')
    print(f'Time panels show {left:.1f}–{right:.1f} s; full recording is {t[0]:.1f}–{t[-1]:.1f} s.')
    print(f'Graph saved: {output.resolve()}')
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
