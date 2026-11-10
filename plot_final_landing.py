"""Plot saved SITL landing telemetry. Does not connect to a drone."""
import argparse
import csv
import json
import math
from pathlib import Path


def number(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return math.nan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv_file', type=Path)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with args.csv_file.open(newline='', encoding='utf-8-sig') as file:
        rows = list(csv.DictReader(file))
    summary = json.loads(args.csv_file.with_suffix('.json').read_text(encoding='utf-8-sig'))
    if not rows:
        raise SystemExit('The CSV is empty.')
    times = [number(r, 'elapsed_s') for r in rows]
    if not all(math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise SystemExit('Log times must be finite and increasing.')

    # Keep only fresh, distinct position samples; gaps remain blank on the graph.
    seen = set()
    fresh = []
    for row in rows:
        stamp = row.get('position_boot_ms', '')
        valid = row.get('telemetry_fresh') == 'True' and stamp and stamp not in seen
        fresh.append(bool(valid))
        if valid:
            seen.add(stamp)

    def telemetry(key, scale=1):
        return [number(r, key) * scale if ok else math.nan for r, ok in zip(rows, fresh)]

    # Find each contiguous target-loss interval, ending at the next logged state.
    hidden = []
    start = None
    for i, row in enumerate(rows):
        lost = row['state'] == 'TARGET LOST - ZERO VELOCITY'
        if lost and start is None:
            start = times[i]
        if start is not None and not lost:
            hidden.append((start, times[i]))
            start = None
    if start is not None:
        hidden.append((start, times[-1]))

    # Start the view just before visual alignment; manual takeoff is excluded.
    active = [t for t, r in zip(times, rows) if r.get('enabled') == 'True']
    left = max(times[0], active[0] - 2) if active else times[0]
    right = times[-1] + 1
    events = [
        ('land_requested_s', 'LAND requested', '#d97706'),
        ('land_mode_confirmed_s', 'LAND confirmed', '#9333ea'),
        ('on_ground_confirmed_s', 'On ground', '#0891b2'),
        ('disarmed_confirmed_s', 'Disarmed', '#15803d'),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    height, velocity, offset = axes
    height.plot(times, telemetry('height_m'), color='royalblue', label='Reported height')
    height.axhline(0.8, color='gray', linestyle='--', label='Nominal handover height (0.8 m)')
    height.set_ylabel('Height above anchored plane (m)')
    height.set_title('Visual alignment, descent and automatic LAND handover')

    velocity.plot(times, telemetry('measured_vd_m_s'), color='royalblue', label='Reported Down velocity')
    requested = [number(r, 'command_down_m_s') if r.get('command_sent') == 'True' else math.nan for r in rows]
    velocity.step(times, requested, where='post', color='darkorange', label='Requested Down velocity (Guided only)')
    velocity.set_ylabel('Down velocity (m/s)')
    velocity.set_title('Positive velocity means descending; Guided commands end at handover')

    reference = telemetry('render_reference_horizontal_error_m', 100)
    vision = [number(r, 'estimated_horizontal_error_m') * 100
              if r.get('detected') == 'True' and r.get('telemetry_fresh') == 'True'
              else math.nan for r in rows]
    offset.plot(times, reference, color='royalblue', label='Telemetry reference offset')
    offset.plot(times, vision, color='darkorange', linestyle='--', label='Image-derived offset (when available)')
    offset.set_ylabel('Horizontal distance to marker (cm)')
    offset.set_xlabel('Seconds since recording started')
    final = summary.get('final_horizontal_offset_m')
    if final is not None:
        offset.scatter([summary['success_s']], [final * 100], color='black', zorder=5)
        offset.annotate(f'Final recorded offset: {final * 100:.2f} cm',
                        xy=(summary['success_s'], final * 100), xytext=(-220, 35),
                        textcoords='offset points', arrowprops={'arrowstyle': '->'})

    for ax in axes:
        for i, (a, b) in enumerate(hidden):
            ax.axvspan(a, b, color='gray', alpha=0.18, label='Target hidden' if i == 0 else None)
        for key, label, color in events:
            event_time = summary.get(key)
            if event_time is not None:
                ax.axvline(event_time, color=color, linestyle=':', linewidth=1.5)
        ax.set_xlim(left, right)
        visible_values = [y for line in ax.lines for x, y in zip(line.get_xdata(), line.get_ydata())
                          if left <= x <= right and math.isfinite(y)]
        if visible_values:
            low, high = min(visible_values), max(visible_values)
            padding = max((high - low) * 0.12, 0.02)
            ax.set_ylim(low - padding, high + padding)
        ax.grid(alpha=0.25)
        ax.legend(loc='upper left', fontsize=8)

    # Event times are listed separately because request and confirmation are close.
    event_text = '   |   '.join(f'{label}: {summary[key]:.2f} s'
                              for key, label, _ in events if summary.get(key) is not None)
    fig.suptitle('SITL visual approach and landing — recorded flight', fontsize=16)
    fig.text(0.5, 0.935, event_text, ha='center', fontsize=9)
    fig.text(0.5, 0.025,
             'Synthetic camera and SITL estimates. Final offset is after disarming, not physical touchdown accuracy.\n'
             'After handover, ArduPilot LAND operates without further vision correction.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.065, 1, 0.94))
    out = args.output_dir or args.csv_file.parent.parent / 'sitl_landing_plots'
    out.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        destination = out / (args.csv_file.stem + '_final.' + extension)
        fig.savefig(destination, dpi=180)
        print(f'Saved: {destination.resolve()}')
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
