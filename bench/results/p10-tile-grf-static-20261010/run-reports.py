"""Run only TILE D128/D256 with at most two AOT TUs concurrently."""
import concurrent.futures
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

out = Path(__file__).resolve().parent
root = out.parents[2]
variant, database = sys.argv[1:]
assert variant in ('B', 'C')
assert os.environ['A770_TOOLS'] == '/home/p51/Github/A770/tools'
flags = [] if variant == 'B' else ['--large-grf']
variant_out = out / variant
variant_out.mkdir(exist_ok=True)

def compile_one(d):
    tu = f'fattn-tile-instance-dkq{d}-dv{d}'
    command = [sys.executable, str(root / 'bin/xe-report.py'),
               str(root / f'third_party/llama.cpp/ggml/src/ggml-sycl/template-instances/{tu}.cpp'),
               '--compile-commands', database, '--wide', *flags]
    print(f'{variant} D{d}: started', flush=True)
    start = time.monotonic()
    with (variant_out / f'{tu}.txt').open('w') as log:
        rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    elapsed = round(time.monotonic() - start, 2)
    print(f'{variant} D{d}: wall {elapsed:.2f}s; xe-report rc={rc}', flush=True)
    return dict(variant=variant, dimension=d, command=shlex.join(command),
                wall_seconds=elapsed, xe_report_rc=rc)

start = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    rows = list(pool.map(compile_one, (128, 256)))
metadata = dict(device='acm-g10', static_only=True, max_parallel_TUs=2,
                source_commit='79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6',
                flags=flags, wall_seconds=round(time.monotonic() - start, 2), reports=rows)
(variant_out / 'compile.json').write_text(json.dumps(metadata, indent=2) + '\n')
assert all(row['xe_report_rc'] in (0, 1) for row in rows), 'compile/report failure'
