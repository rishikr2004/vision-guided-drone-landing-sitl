"""Analyse this confirmed measurement session; do not reuse its row labels for other runs."""
from pathlib import Path
import argparse
import csv
import math
import statistics

NAME = 'position_20261109_180114_708409.csv'
parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=Path)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parent
source = args.csv or root / 'position_checks' / NAME
out = args.output or root / 'position_checks' / 'reports'
with source.open(newline='', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
if source.name != NAME or len(rows) != 20:
    raise SystemExit('This report expects the confirmed 20-reading session: ' + NAME)
if any(r['calibration_session'] != 'session_20261109_175221_567046' for r in rows):
    raise SystemExit('Unexpected calibration session.')
# User-confirmed capture order: 40 cm, 30 cm, 50 cm.
groups = [(30, 5, 10), (40, 0, 5), (50, 10, 20)]
lines = ['Webcam marker distance validation', '', f'Source: {source.name}',
         'Reference distances were measured manually by the user.',
         'Confirmed labels: rows 1-5 = 40 cm; 6-10 = 30 cm; 11-20 = 50 cm.',
         'Marker width configured: 5.25 cm. Calibration: session_20261109_175221_567046.', '',
         'Ruler | Count | Mean estimate | Mean signed error | Error % | Min-max | Sample SD | RMSE']
for actual, start, end in groups:
    values = [float(r['range_m']) * 100 for r in rows[start:end]]
    mean = statistics.mean(values)
    bias = mean - actual
    sd = statistics.stdev(values)
    rmse = math.sqrt(statistics.mean((v - actual)**2 for v in values))
    lines.append(f'{actual:.0f} cm | {len(values)} | {mean:.2f} cm | {bias:+.2f} cm | '
                 f'{100*bias/actual:+.2f}% | {min(values):.2f}-{max(values):.2f} cm | '
                 f'{sd:.3f} cm | {rmse:.2f} cm')
lines += ['', 'Interpretation:',
          'Positive signed error means the estimate is higher than the ruler reference.',
          'SD describes variation within each short capture group; it is not absolute accuracy.',
          'RMSE is calculated against the user-supplied ruler distance.',
          'The unequal group sizes are reported separately; no pooled accuracy claim is made.',
          'These were three short, near-centred checks, not independent repositioning trials.',
          'Ruler placement, marker-size uncertainty, focus, glare and calibration can affect error.',
          'No correction multiplier has been fitted or applied.',
          'This does not validate drone landing, lateral position, orientation or other distances.']
out.mkdir(parents=True, exist_ok=True)
report = out / 'distance_validation_20261109_180114.txt'
report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
print(f'\nReport saved to: {report}')
