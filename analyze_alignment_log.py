"""Read a SITL alignment CSV without connecting to or controlling a drone."""
import argparse
import csv
import math
from pathlib import Path
from statistics import mean


def number(row, column):
    value = row.get(column, "")
    return float(value) if value else None


def summary(rows, column, scale=1):
    values = [number(row, column) for row in rows]
    values = [value * scale for value in values if value is not None]
    if not values:
        return "no data"
    return (f"first={values[0]:.3f}, last={values[-1]:.3f}, "
            f"min={min(values):.3f}, max={max(values):.3f}, "
            f"mean={mean(values):.3f}, span={max(values)-min(values):.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    with args.csv_file.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit("The CSV has no readings.")
    required = {"elapsed_s", "state", "enabled", "detected", "height_m",
                "render_reference_horizontal_error_m", "command_sent",
                "command_north_m_s", "command_east_m_s", "command_down_m_s"}
    if not required.issubset(rows[0]):
        raise SystemExit(f"Missing columns: {sorted(required-set(rows[0]))}")
    times = [float(row["elapsed_s"]) for row in rows]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise SystemExit("Times are not strictly increasing; inspect the CSV.")
    lines = [f"Source: {args.csv_file.resolve()}", f"Recorded rows: {len(rows)}"]
    active = [r for r in rows if r["enabled"] == "True" and r["command_sent"] == "True"]
    lines += ["", "ACTIVE CONTROL (excludes manual takeoff and landing):",
              "Height (m): " + summary(active, "height_m")]
    heights = [number(r, "height_m") for r in active if r["height_m"]]
    if heights:
        lines.append(f"Largest sampled difference from 2 m: {max(abs(h-2) for h in heights)*100:.3f} cm")
    groups = []
    start = None
    for i, row in enumerate(rows):
        lost = row["state"] == "TARGET LOST - ZERO VELOCITY"
        if lost and start is None:
            start = i
        if start is not None and (not lost or i == len(rows)-1):
            end = i if not lost else i+1
            groups.append((start, end))
            start = None
    lines.append(f"Target-loss intervals: {len(groups)}")
    for index, (start, end) in enumerate(groups, 1):
        group = rows[start:end]
        t0 = times[start]
        next_time = times[end] if end < len(rows) else times[end-1]
        late = [r for r in group if float(r["elapsed_s"])-t0 >= 5]
        sent = sum(r["command_sent"] == "True" for r in group)
        nonzero = sum(any(abs(float(r[c])) > 1e-9 for c in
                         ("command_north_m_s", "command_east_m_s", "command_down_m_s"))
                      for r in group if r["command_sent"] == "True")
        gaps = [b-a for a, b in zip(times[start:end-1], times[start+1:end])]
        lines += ["", f"TARGET LOSS {index}: {t0:.3f} to {next_time:.3f} s",
                  f"Duration to next state (or final row): {next_time-t0:.3f} s",
                  f"Rows: {len(group)}; command sent: {sent}; nonzero sent commands: {nonzero}",
                  f"Rows reporting detection: {sum(r['detected']=='True' for r in group)}",
                  f"Rows missing height: {sum(not r['height_m'] for r in group)}",
                  f"Largest within-interval logging gap: {max(gaps, default=0):.3f} s",
                  "Distance to marker (cm): " + summary(group, "render_reference_horizontal_error_m", 100),
                  "Height (m): " + summary(group, "height_m"),
                  "After first 5 seconds (diagnostic window, not a settling criterion):",
                  "Distance to marker (cm): " + summary(late, "render_reference_horizontal_error_m", 100),
                  "Height (m): " + summary(late, "height_m")]
        for seconds in (1, 2, 3, 5):
            candidates = [r for r in group if float(r["elapsed_s"])-t0 >= seconds]
            if candidates:
                r = candidates[0]
                offset = number(r, "render_reference_horizontal_error_m")
                if offset is not None:
                    lines.append(f"At +{float(r['elapsed_s'])-t0:.3f} s: distance to marker={offset*100:.3f} cm")
        if end < len(rows):
            lines.append(f"Next state: {rows[end]['state']}")
    lines += ["", "LIMITS:",
              "Commands are requested velocities, not measured velocities or acknowledgements.",
              "Distance to marker is a scalar from SITL telemetry. Its change is not path length.",
              "A constant distance does not prove no sideways movement around the marker.",
              "Height is relative to the anchored simulated ground plane.",
              "These are sampled results from one simulation, not physical landing validation."]
    report = "\n".join(lines)
    print(report)
    output_dir = args.output_dir or args.csv_file.parent.parent / "alignment_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / (args.csv_file.stem + "_review.txt")
    destination.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved: {destination.resolve()}")


if __name__ == "__main__":
    main()
