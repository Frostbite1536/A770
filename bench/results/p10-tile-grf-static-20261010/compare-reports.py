"""Extract actual TILE bodies, excluding launch_fattn helper kernels."""
from collections import Counter
from pathlib import Path
import re

out = Path(__file__).resolve().parent
baseline = out.parent / 'llama-sycl-fa-static-20261010'
fields = re.compile(r'\s+(\d+(?:\s+\d+){7})(?:\s+\D.*)?\s*$')
instance = re.compile(r'flash_attn_tile<(\d+), (\d+), (\d+), (\d+), (false|true), (\d+)>')

def extract(directory):
    rows = {}
    for d in (128, 256):
        report = directory / f'fattn-tile-instance-dkq{d}-dv{d}.txt'
        text = report.read_text()
        assert text.rstrip().endswith('note: spill/private/slm in bytes as reported by IGC; slm is static only, dynamic SLM is budgeted in the design doc.'), report
        count = 0
        for line in text.splitlines():
            if not line.lstrip().startswith('lauch_kernel<') or 'flash_attn_tile<' not in line:
                continue
            match = instance.search(line)
            assert match, line
            dkq, dv, n1, n2, softcap, simd = match.groups()
            key = (int(dkq), int(dv), int(n1), int(n2), softcap, int(simd))
            assert key[0] == key[1] == d and key[5] == 32
            assert key not in rows, ('duplicate body', key)
            match = fields.search(line)
            assert match, line
            rows[key] = tuple(map(int, match.group(1).split()))
            assert rows[key][0] == 32, ('changed SIMD', key, rows[key])
            assert rows[key][6:] == (0, 0), ('unexpected DPAS/FP64', key, rows[key])
            count += 1
        assert count == 34, (report, count)
    return rows

def p6(key):
    return key[3] in (4, 8) and key[2] * key[3] in (8, 16, 32)

def label(key):
    d, _, n1, n2, softcap, _ = key
    value = f'{d}:{n1}x{n2}:{"T" if softcap == "true" else "F"}'
    return f'**{value}**' if p6(key) else value

variants = {'A': extract(baseline), 'B': extract(out / 'B'), 'C': extract(out / 'C')}
assert variants['A'].keys() == variants['B'].keys() == variants['C'].keys()
keys = sorted(variants['A'], key=lambda k: (k[0], k[4], -k[3], -k[2]))
assert sum(map(p6, keys)) == 24
assert all(r[:3] == (32, 128, 8) for r in variants['A'].values())
intro = ('Instance `D:ncols1xncols2:F/T` means '
         '`flash_attn_tile<D, D, ncols1, ncols2, softcap, 32>`; F=false, T=true. '
         '**Bold** marks P-6: ncols2=4/8 and cols_per_block=ncols1*ncols2=8/16/32. '
         'Spill and SLM are bytes reported by IGC. Actual `lauch_kernel` TILE bodies only; '
         'launch helpers are excluded.\n')
lines = [intro, '| instance | spill A | spill B | spill C | GRF B | threads/XVE B |',
         '|---|---:|---:|---:|---:|---:|']
for key in keys:
    a, b, c = (variants[v][key] for v in ('A', 'B', 'C'))
    lines.append(f'| {label(key)} | {a[3]} | {b[3]} | {c[3]} | {b[1]} | {b[2]} |')
a, b, c = (variants[v] for v in ('A', 'B', 'C'))
assert all(b[k][:3] == (32, 256, 4) for k in keys)
assert all(a[k][5] == b[k][5] == c[k][5] for k in keys)
assert all(c[k][:3] == (32, 256, 4) for k in keys)
assert all(b[k][3] > 0 and c[k][3] > 0 for k in keys)
lower = sum(b[k][3] < a[k][3] for k in keys)
higher = sum(b[k][3] > a[k][3] for k in keys)
p6higher = [k for k in keys if p6(k) and b[k][3] > a[k][3]]
different = sum(b[k][:6] != c[k][:6] for k in keys)
assert all(b[k][:3] + b[k][4:] == c[k][:3] + c[k][4:] for k in keys)
c_lower = sum(c[k][3] < b[k][3] for k in keys)
c_higher = sum(c[k][3] > b[k][3] for k in keys)
reading = [
    '- B removes spills nowhere: 0/34 at D128 and 0/34 at D256, including 0/24 P-6 instances.',
    '- B chooses 256 GRF for all 68 instances: IGC reports 4 threads/XVE versus 8 for A; SIMD stays 32 and static SLM is unchanged.',
    f'- B reduces spill bytes in {lower}/68 instances and increases them in {higher}/68; the {len(p6higher)} P-6 increases are D256, cols_per_block=16, ncols2=4/8, with both softcap settings.',
    f'- C also retains spills in all 68 instances at 256 GRF and 4 threads/XVE; versus B, {c_lower} instances report lower spill bytes, {c_higher} higher, and {68 - different} identical; other resource fields match.',
    '- Static evidence only: fewer spills is not faster; runtime spill cost versus halved threads/XVE remains unverified and requires an A770 A/B measurement.',
]
lines.extend(['', *reading])
(out / 'comparison.md').write_text('\n'.join(lines) + '\n')
for variant in ('B', 'C'):
    lines = [intro, '| instance | SIMD | GRF | threads/XVE | spill | SLM |',
             '|---|---:|---:|---:|---:|---:|']
    for key in keys:
        row = variants[variant][key]
        lines.append(f'| {label(key)} | {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[5]} |')
    (out / variant / 'instances.md').write_text('\n'.join(lines) + '\n')
    print(variant, 'GRF/thread modes:', dict(Counter((r[1], r[2]) for r in variants[variant].values())))
    for d in (128, 256):
        rows = [r for key, r in variants[variant].items() if key[0] == d]
        print(variant, d, 'zero spill:', sum(r[3] == 0 for r in rows), '/', len(rows),
              'spill range:', min(r[3] for r in rows), max(r[3] for r in rows))
    print(variant, 'P-6 zero spill:', sum(variants[variant][key][3] == 0 for key in keys if p6(key)), '/24')
print('B/C resource differences:', [(label(k), variants['B'][k][:6], variants['C'][k][:6]) for k in keys if variants['B'][k][:6] != variants['C'][k][:6]])
print('B still spills:', [(label(k), variants['B'][k][3]) for k in keys if variants['B'][k][3]])
print('C still spills:', [(label(k), variants['C'][k][3]) for k in keys if variants['C'][k][3]])
