p11c: two independent SYCL TILE FP16 candidate fixes, 2026-10-11

Scope
-----
Branch p11c-sycl-fa-tile-fp16-fix-candidates, based on main a78c40c.
llama.cpp pin 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6.
base = pin + numbered patches 0001-0005 + updated test-only CPU patch.
F = base + F-SYCL-FA-tile-flush-FP16-VKQ-to-FP32-per-KV-tile.patch.
S = base + S-SYCL-FA-tile-cap-keys-per-block.patch.
All three also receive the identical archived scratch diagnostic diff.
Device: Intel Xeon E3-1505M v6 @ 3.00GHz, Intel OpenCL 3.0, 8 compute
units, max work-group 8192, driver 2026.21.7.0.24_160000.
Intel OpenCL CPU device only; these are same-source CPU numerics, not A770
numerics. Static target acm-g10 only. Timing: not run.
Wall times include build, configure, CPU JIT and tests or static compilation;
they are not performance measurements. No candidate is selected.
No tolerance changes, existing test changes, installs, network, commits,
pushes, main-checkout writes or prohibited device instructions were used.

Measured summary
----------------
variant  g-a770 biased pass / max TILE NMSE   g-min biased pass / max TILE NMSE
base     12/16 / 2.294307766                 8/16 / 11.730833579
F        16/16 / 0.000000005                 16/16 / 0.000000004
S        16/16 / 0.000008770                 16/16 / 0.000008322
F g-cpu: P 97/97, P6 117/117
F g-a770: P 97/97, P6 117/117
S g-cpu: P 97/97, P6 117/117
S g-a770: P 97/97, P6 117/117
The 97 P cases include all original 85 plus 12 additions; P6 includes all
112 patch-0002 cases plus five pre-existing matches.
Static active TILE bodies (spill delta min..max in bytes, maximum spill base->F):
  D=64: 17 active instances; delta -576..+1024; max 10784->11296
  D=128: 34 active instances; delta +96..+2144; max 13984->16032
  D=256: 34 active instances; delta -384..+2592; max 17248->18496
SIMD/GRF changes: none
No A770 runtime or performance verification; no candidate chosen.

Implementation and sizing
-------------------------
F ported from the P100 project, patch 0004. Only fattn-tile.hpp changes.
A float2 VKQ_f32 sits next to half2 VKQ, with equal element counts. Each KV
iteration rescales VKQ_f32 in FP32, accumulates its tile in half2, adds that
tile into VKQ_f32 and zeros VKQ. The np>1 combine reuses KV_tmp as floats;
a byte-size static_assert proves nwarps*DVp floats fit. Copies use float2
and 8 bytes per element. Sink rescale and final output also use VKQ_f32.
The entire source projection with SYCL_FAST_FP16 absent is byte-identical
(assertion in preparation script); actual git diff hunks are archived in
patch-validation.log.gz. This is a source audit, not a non-fast runtime test.

S changes only host launch plumbing: defaulted max_keys=0, set to 1024 by
all six TILE launches. The measured p11b bound is cited beside the define.
In the non-stream-k path, before the unchanged efficiency search, use
min(ntiles_KQ, max(previous_bound, ceil(ntiles_KQ*nbatch_fa/max_keys))).
The product is calculated in int64_t. VEC and all other callers omit the
parameter, retaining 0 and the original split. Stream-k is unchanged.
After the search, existing dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV))
and dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV)) use the final block
count. blocks_num.y and the combine's parallel_blocks argument use the same
count; its local storage is parallel_blocks*sizeof(float2). Thus all partial
outputs, metadata and combine storage grow with S's larger split. Phantom
heads from 0001 do not write output and combine uses actual Q dimensions.

Reproduction
------------
Start every shell command with:
  export PATH=/home/p51/Github/A770/.venv/bin:$PATH;
  export A770_TOOLS=/home/p51/Github/A770/tools
Local scratch setup (no network):
  git clone --no-hardlinks --no-checkout third_party/llama.cpp third_party/p11c/<variant>
  git -C third_party/p11c/<variant> checkout --detach 79e2e74
  for patch in "$PWD"/third_party/patches/000[1-5]-*.patch; do
    git -C third_party/p11c/<variant> apply "$patch"
  done
  git -C third_party/p11c/<variant> apply "$PWD/third_party/patches/test-only/0001-sycl-allow-cpu-device-for-tests.patch"
Paths passed to git apply are absolute in the archived preparation script.
For F/S apply the respective candidate, then decompress and apply the archived <variant>-scratch-diagnostics.diff.gz.
prepare.py.gz records the exact candidate derivation and checks; runs.py.gz
records the exact serial run orchestration. commands.json.gz lists commands,
geometry, LLAMA, actual exit codes and wall times for every completed run.

Numerical environment:
  export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=/tmp/p11c-sycl-cache
  export GGML_SYCL_MKL_FA_DEBUG=1
  unset SYCL_PROGRAM_COMPILE_OPTIONS SYCL_PROGRAM_BUILD_OPTIONS SYCL_CPU_F16
  export LLAMA=$PWD/third_party/p11c/<base|F|S>
Geometries:
  g-cpu: unset GGML_SYCL_TEST_NSM GGML_SYCL_TEST_MAX_WG_PER_CU
  g-a770: export GGML_SYCL_TEST_NSM=32 GGML_SYCL_TEST_MAX_WG_PER_CU=2
  g-min: export GGML_SYCL_TEST_NSM=32 GGML_SYCL_TEST_MAX_WG_PER_CU=1
Run each base/F/S at g-a770 and g-min:
  bin/sycl-cpu-tbo.sh -o FLASH_ATTN_EXT -p 'v_bias=2'
Run each F/S at g-cpu and g-a770:
  bin/sycl-cpu-tbo.sh -o FLASH_ATTN_EXT -p "$P"
  bin/sycl-cpu-tbo.sh -o FLASH_ATTN_EXT -p 'nr23=\[[3567],1\]'
P is copied verbatim from the p6a README into filters.txt.gz (included below).
On the mandated 0001-0005 stack this selects 97 cases: the original 85 plus
12 added by 0002 (GQA=6, two dimensions, five selected non-sink F16 batches
and one selected Q8 batch per dimension). No cases or filters were changed.
The complete historical 85-case multiset, including its duplicate, is checked
against every new P log (historical-P-cases.log.gz, sourced from p11b A g-cpu).
All tests use F16 ON, Release optimized spir64 JIT, per-kernel images,
oneDNN and Level Zero OFF, target test-backend-ops, -j 8. The harness validates
SYCL0 is Intel OpenCL CPU in an empty preflight, then runs the selected tests.
Preflight 0/0 summaries and the skipped separate CPU backend are not successes
in the reported case counts. Input tensors are randomly initialized across
runs; these are not paired experiments on identical inputs.
219 prior p11b CPU JIT cache files seeded without overwrite, each hash verified;
268 additional p6b CPU JIT cache files were later seeded the same way.
cache-seed.log.gz and cache-seed-p6b.log.gz give provenance. Cache keys
select matching source/options; unmatched entries do not bypass compilation.

base-g-a770-bias
D    kv     nb   route p    T    tile keys/acc result NMSE
128  16384  1    VEC   16   4    128  64       PASS   0.000000003
128  16384  4    TILE  16   4    64   1024     PASS   0.000008163
128  16384  16   TILE  16   4    64   1024     PASS   0.000007941
128  16384  512  TILE  2    512  64   8192     FAIL   0.444641228
128  32768  1    VEC   16   4    128  128      PASS   0.000000008
128  32768  4    TILE  16   4    64   2048     PASS   0.000182086
128  32768  16   TILE  16   4    64   2048     PASS   0.000150796
128  32768  512  TILE  2    512  64   16384    FAIL   2.294307766
256  16384  1    VEC   16   4    256  64       PASS   0.000000003
256  16384  4    TILE  16   4    64   1024     PASS   0.000007518
256  16384  16   TILE  16   4    64   1024     PASS   0.000007442
256  16384  512  TILE  2    512  64   8192     FAIL   0.455025873
256  32768  1    VEC   16   4    256  128      PASS   0.000000007
256  32768  4    TILE  16   4    64   2048     PASS   0.000137592
256  32768  16   TILE  16   4    64   2048     PASS   0.000143252
256  32768  512  TILE  2    512  64   16384    FAIL   2.284577316
  0/0 tests passed
2/2 backends passed
  12/16 tests passed
1/2 backends passed
wall_seconds=821.60 actual_exit=1

base-g-min-bias
D    kv     nb   route p    T    tile keys/acc result NMSE
128  16384  1    VEC   8    4    128  128      PASS   0.000000011
128  16384  4    TILE  8    4    64   2048     PASS   0.000157574
128  16384  16   TILE  8    4    64   2048     PASS   0.000168217
128  16384  512  TILE  1    512  64   16384    FAIL   2.274493925
128  32768  1    VEC   8    4    128  256      PASS   0.000000029
128  32768  4    TILE  8    4    64   4096     FAIL   0.023526877
128  32768  16   TILE  8    4    64   4096     FAIL   0.022069249
128  32768  512  TILE  1    512  64   32768    FAIL   11.730833579
256  16384  1    VEC   8    4    256  128      PASS   0.000000013
256  16384  4    TILE  8    4    64   2048     PASS   0.000134163
256  16384  16   TILE  8    4    64   2048     PASS   0.000148078
256  16384  512  TILE  1    512  64   16384    FAIL   2.278238218
256  32768  1    VEC   8    4    256  256      PASS   0.000000033
256  32768  4    TILE  8    4    64   4096     FAIL   0.021544372
256  32768  16   TILE  8    4    64   4096     FAIL   0.022576704
256  32768  512  TILE  1    512  64   32768    FAIL   11.598527341
  0/0 tests passed
2/2 backends passed
  8/16 tests passed
1/2 backends passed
wall_seconds=211.96 actual_exit=1

F-g-a770-bias
D    kv     nb   route p    T    tile keys/acc result NMSE
128  16384  1    VEC   16   4    128  64       PASS   0.000000005
128  16384  4    TILE  16   4    64   1024     PASS   0.000000005
128  16384  16   TILE  16   4    64   1024     PASS   0.000000004
128  16384  512  TILE  2    512  64   8192     PASS   0.000000004
128  32768  1    VEC   16   4    128  128      PASS   0.000000007
128  32768  4    TILE  16   4    64   2048     PASS   0.000000002
128  32768  16   TILE  16   4    64   2048     PASS   0.000000002
128  32768  512  TILE  2    512  64   16384    PASS   0.000000002
256  16384  1    VEC   16   4    256  64       PASS   0.000000003
256  16384  4    TILE  16   4    64   1024     PASS   0.000000003
256  16384  16   TILE  16   4    64   1024     PASS   0.000000004
256  16384  512  TILE  2    512  64   8192     PASS   0.000000004
256  32768  1    VEC   16   4    256  128      PASS   0.000000009
256  32768  4    TILE  16   4    64   2048     PASS   0.000000002
256  32768  16   TILE  16   4    64   2048     PASS   0.000000002
256  32768  512  TILE  2    512  64   16384    PASS   0.000000002
  0/0 tests passed
2/2 backends passed
  16/16 tests passed
2/2 backends passed
wall_seconds=1178.77 actual_exit=0

F-g-min-bias
D    kv     nb   route p    T    tile keys/acc result NMSE
128  16384  1    VEC   8    4    128  128      PASS   0.000000013
128  16384  4    TILE  8    4    64   2048     PASS   0.000000004
128  16384  16   TILE  8    4    64   2048     PASS   0.000000003
128  16384  512  TILE  1    512  64   16384    PASS   0.000000004
128  32768  1    VEC   8    4    128  256      PASS   0.000000029
128  32768  4    TILE  8    4    64   4096     PASS   0.000000002
128  32768  16   TILE  8    4    64   4096     PASS   0.000000002
128  32768  512  TILE  1    512  64   32768    PASS   0.000000002
256  16384  1    VEC   8    4    256  128      PASS   0.000000015
256  16384  4    TILE  8    4    64   2048     PASS   0.000000004
256  16384  16   TILE  8    4    64   2048     PASS   0.000000004
256  16384  512  TILE  1    512  64   16384    PASS   0.000000004
256  32768  1    VEC   8    4    256  256      PASS   0.000000036
256  32768  4    TILE  8    4    64   4096     PASS   0.000000002
256  32768  16   TILE  8    4    64   4096     PASS   0.000000002
256  32768  512  TILE  1    512  64   32768    PASS   0.000000002
  0/0 tests passed
2/2 backends passed
  16/16 tests passed
2/2 backends passed
wall_seconds=208.03 actual_exit=0

S-g-a770-bias
D    kv     nb   route p    T    tile keys/acc result NMSE
128  16384  1    VEC   16   4    128  64       PASS   0.000000005
128  16384  4    TILE  16   4    64   1024     PASS   0.000008013
128  16384  16   TILE  16   4    64   1024     PASS   0.000008183
128  16384  512  TILE  16   512  64   1024     PASS   0.000007688
128  32768  1    VEC   16   4    128  128      PASS   0.000000007
128  32768  4    TILE  32   4    64   1024     PASS   0.000007961
128  32768  16   TILE  32   4    64   1024     PASS   0.000007512
128  32768  512  TILE  32   512  64   1024     PASS   0.000007390
256  16384  1    VEC   16   4    256  64       PASS   0.000000005
256  16384  4    TILE  16   4    64   1024     PASS   0.000008770
256  16384  16   TILE  16   4    64   1024     PASS   0.000008149
256  16384  512  TILE  16   512  64   1024     PASS   0.000007851
256  32768  1    VEC   16   4    256  128      PASS   0.000000006
256  32768  4    TILE  32   4    64   1024     PASS   0.000007086
256  32768  16   TILE  32   4    64   1024     PASS   0.000008000
256  32768  512  TILE  32   512  64   1024     PASS   0.000007928
  0/0 tests passed
2/2 backends passed
  16/16 tests passed
2/2 backends passed
wall_seconds=678.20 actual_exit=0

S-g-min-bias
D    kv     nb   route p    T    tile keys/acc result NMSE
128  16384  1    VEC   8    4    128  128      PASS   0.000000021
128  16384  4    TILE  16   4    64   1024     PASS   0.000007558
128  16384  16   TILE  16   4    64   1024     PASS   0.000008057
128  16384  512  TILE  16   512  64   1024     PASS   0.000007824
128  32768  1    VEC   8    4    128  256      PASS   0.000000029
128  32768  4    TILE  32   4    64   1024     PASS   0.000006900
128  32768  16   TILE  32   4    64   1024     PASS   0.000006726
128  32768  512  TILE  32   512  64   1024     PASS   0.000007400
256  16384  1    VEC   8    4    256  128      PASS   0.000000015
256  16384  4    TILE  16   4    64   1024     PASS   0.000008318
256  16384  16   TILE  16   4    64   1024     PASS   0.000007930
256  16384  512  TILE  16   512  64   1024     PASS   0.000008322
256  32768  1    VEC   8    4    256  256      PASS   0.000000029
256  32768  4    TILE  32   4    64   1024     PASS   0.000006818
256  32768  16   TILE  32   4    64   1024     PASS   0.000007267
256  32768  512  TILE  32   512  64   1024     PASS   0.000007555
  0/0 tests passed
2/2 backends passed
  16/16 tests passed
2/2 backends passed
wall_seconds=211.96 actual_exit=0

Biased-V tables: D=128/256, KV=16384/32768, nb=1/4/16/512, GQA=4,
no mask, F16 K/V, KV views, V+2 on channels 0..7. Unchanged NMSE tolerance
0.000500000. Route/p/T/tile are measured diagnostic prints. keys/acc is
source-derived: TILE kv/p; VEC kv/(16*p), since VEC has 16 warp/lane partial
sums per block. All selected shapes divide evenly. F's keys/acc column means
keys processed by each block's carried accumulator; its half2 accumulator
is reset every tile (64 keys here). NMSE prints have nine decimal places;
printed zero does not imply mathematical equality.

Existing coverage result lines
------------------------------
Patch 0002 adds 112 cases (2 dimensions x 4 GQA ratios x (6 batches x
2 sink settings + 2 quantized batches)); the requested filter also selects
five pre-existing matching cases, for 117 total. All 112 patch-0002 case
strings are checked with multiplicity against every P6 log. Source-derived
case strings and the five additional cases are archived in
patch-0002-cases.log.gz and P6-coverage-audit.log.gz. Counts below are measured
from the runner.
F-g-cpu-P: 97/97 tests passed, 2/2 backends passed; counted {'OK': 97}
wall_seconds=2213.39 actual_exit=0
F-g-cpu-P6: 117/117 tests passed, 2/2 backends passed; counted {'OK': 117}
wall_seconds=1202.26 actual_exit=0
F-g-a770-P: 97/97 tests passed, 2/2 backends passed; counted {'OK': 97}
wall_seconds=189.25 actual_exit=0
F-g-a770-P6: 117/117 tests passed, 2/2 backends passed; counted {'OK': 117}
wall_seconds=216.66 actual_exit=0
S-g-cpu-P: 97/97 tests passed, 2/2 backends passed; counted {'OK': 97}
wall_seconds=193.12 actual_exit=0
S-g-cpu-P6: 117/117 tests passed, 2/2 backends passed; counted {'OK': 117}
wall_seconds=213.44 actual_exit=0
S-g-a770-P: 97/97 tests passed, 2/2 backends passed; counted {'OK': 97}
wall_seconds=190.31 actual_exit=0
S-g-a770-P6: 117/117 tests passed, 2/2 backends passed; counted {'OK': 117}
wall_seconds=214.77 actual_exit=0

Static comparison (new base report, no prior-report reuse)
----------------------------------------------------------
  LLAMA=$PWD/third_party/p11c/base bin/llama-sycl-fa-static.sh bench/results/p11c-sycl-fa-tile-fp16-fix-20261011/static-base
  LLAMA=$PWD/third_party/p11c/F bin/llama-sycl-fa-static.sh bench/results/p11c-sycl-fa-tile-fp16-fix-20261011/static-F
Both use A770_TOOLS=/home/p51/Github/A770/tools; IGC acm-g10, AOT -O2,
default GRF mode. No CPU override/device acceptance gate for static reports.
The script reports TILE dkq64/128/256 plus two VEC translation units.
TU xe-report rc=1 flags resource violations such as upstream spills; the
script exits 0 when compilation and row extraction succeed (rc>=2 fails).
Below only actual lauch_kernel TILE device bodies, excluding host launch,
copy and fixup wrappers. Every emitted ncols1/ncols2/logit-softcap instance
is compared; all retain the requested subgroup 32.
D64 softcap=true is an unused return-only variant (zero SLM/spill); active
instance summaries exclude those 17 empty bodies. All TILE body rows in
both reports have zero DPAS and FP64 instruction counts.
D   cols1 cols2 softcap GRF base->F SIMD base->F spill bytes base->F delta
64  1     2     false   128->128    32->32        9920->10048  +128
64  1     2     true    128->128    32->32        0->0      +0
64  1     4     false   128->128    32->32        3072->3200   +128
64  1     4     true    128->128    32->32        0->0      +0
64  1     8     false   128->128    32->32        1984->2112   +128
64  1     8     true    128->128    32->32        0->0      +0
64  2     1     false   128->128    32->32        5504->5568   +64
64  2     1     true    128->128    32->32        0->0      +0
64  2     2     false   128->128    32->32        3200->3328   +128
64  2     2     true    128->128    32->32        0->0      +0
64  2     4     false   128->128    32->32        2112->2240   +128
64  2     4     true    128->128    32->32        0->0      +0
64  2     8     false   128->128    32->32        4480->4736   +256
64  2     8     true    128->128    32->32        0->0      +0
64  4     1     false   128->128    32->32        10784->11296  +512
64  4     1     true    128->128    32->32        0->0      +0
64  4     2     false   128->128    32->32        2112->2240   +128
64  4     2     true    128->128    32->32        0->0      +0
64  4     4     false   128->128    32->32        4480->4736   +256
64  4     4     true    128->128    32->32        0->0      +0
64  4     8     false   128->128    32->32        7264->6880   -384
64  4     8     true    128->128    32->32        0->0      +0
64  8     1     false   128->128    32->32        10528->10656  +128
64  8     1     true    128->128    32->32        0->0      +0
64  8     2     false   128->128    32->32        3872->4160   +288
64  8     2     true    128->128    32->32        0->0      +0
64  8     4     false   128->128    32->32        6784->6208   -576
64  8     4     true    128->128    32->32        0->0      +0
64  16    1     false   128->128    32->32        5536->6048   +512
64  16    1     true    128->128    32->32        0->0      +0
64  16    2     false   128->128    32->32        7104->6528   -576
64  16    2     true    128->128    32->32        0->0      +0
64  32    1     false   128->128    32->32        9312->10336  +1024
64  32    1     true    128->128    32->32        0->0      +0
128 1     2     false   128->128    32->32        8768->9088   +320
128 1     2     true    128->128    32->32        8896->9184   +288
128 1     4     false   128->128    32->32        8608->8928   +320
128 1     4     true    128->128    32->32        8672->8832   +160
128 1     8     false   128->128    32->32        7840->8032   +192
128 1     8     true    128->128    32->32        8000->8224   +224
128 2     1     false   128->128    32->32        9728->10304  +576
128 2     1     true    128->128    32->32        9888->10336  +448
128 2     2     false   128->128    32->32        8736->9056   +320
128 2     2     true    128->128    32->32        8800->8896   +96
128 2     4     false   128->128    32->32        7968->8160   +192
128 2     4     true    128->128    32->32        8128->8352   +224
128 2     8     false   128->128    32->32        7872->8928   +1056
128 2     8     true    128->128    32->32        8096->8992   +896
128 4     1     false   128->128    32->32        7776->8224   +448
128 4     1     true    128->128    32->32        7904->8224   +320
128 4     2     false   128->128    32->32        7968->8160   +192
128 4     2     true    128->128    32->32        8128->8352   +224
128 4     4     false   128->128    32->32        7872->8928   +1056
128 4     4     true    128->128    32->32        8096->8992   +896
128 4     8     false   128->128    32->32        11520->13600  +2080
128 4     8     true    128->128    32->32        11552->13568  +2016
128 8     1     false   128->128    32->32        6688->7072   +384
128 8     1     true    128->128    32->32        6784->7328   +544
128 8     2     false   128->128    32->32        7360->8320   +960
128 8     2     true    128->128    32->32        7648->8480   +832
128 8     4     false   128->128    32->32        10816->12768  +1952
128 8     4     true    128->128    32->32        10976->12800  +1824
128 16    1     false   128->128    32->32        9056->10400  +1344
128 16    1     true    128->128    32->32        9280->10560  +1280
128 16    2     false   128->128    32->32        11136->13088  +1952
128 16    2     true    128->128    32->32        11296->13120  +1824
128 32    1     false   128->128    32->32        13664->15808  +2144
128 32    1     true    128->128    32->32        13984->16032  +2048
256 1     2     false   128->128    32->32        10432->11456  +1024
256 1     2     true    128->128    32->32        10432->11424  +992
256 1     4     false   128->128    32->32        8736->9216   +480
256 1     4     true    128->128    32->32        8576->9344   +768
256 1     8     false   128->128    32->32        7552->8320   +768
256 1     8     true    128->128    32->32        7584->8544   +960
256 2     1     false   128->128    32->32        11296->12256  +960
256 2     1     true    128->128    32->32        11488->12320  +832
256 2     2     false   128->128    32->32        8896->9440   +544
256 2     2     true    128->128    32->32        8736->9568   +832
256 2     4     false   128->128    32->32        7776->8640   +864
256 2     4     true    128->128    32->32        7808->8736   +928
256 2     8     false   128->128    32->32        10432->12640  +2208
256 2     8     true    128->128    32->32        10784->12992  +2208
256 4     1     false   128->128    32->32        9184->10176  +992
256 4     1     true    128->128    32->32        9280->10240  +960
256 4     2     false   128->128    32->32        7776->8640   +864
256 4     2     true    128->128    32->32        7808->8736   +928
256 4     4     false   128->128    32->32        10432->12640  +2208
256 4     4     true    128->128    32->32        10784->12992  +2208
256 4     8     false   128->128    32->32        14656->14464  -192
256 4     8     true    128->128    32->32        14784->15488  +704
256 8     1     false   128->128    32->32        8224->9216   +992
256 8     1     true    128->128    32->32        8256->9280   +1024
256 8     2     false   128->128    32->32        9888->12448  +2560
256 8     2     true    128->128    32->32        10016->12352  +2336
256 8     4     false   128->128    32->32        13696->13824  +128
256 8     4     true    128->128    32->32        14208->14368  +160
256 16    1     false   128->128    32->32        11584->13664  +2080
256 16    1     true    128->128    32->32        11456->14048  +2592
256 16    2     false   128->128    32->32        14784->14400  -384
256 16    2     true    128->128    32->32        15008->15168  +160
256 32    1     false   128->128    32->32        15936->17824  +1888
256 32    1     true    128->128    32->32        17248->18496  +1248
SIMD/GRF changes: none

Static IGC report: device acm-g10 (Intel Arc A770); no runtime measurements.
llama.cpp commit: 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
Flags: compile database -I/-isystem/-D/-std=; xe-report AOT -O2; extra args: (none; default GRF).
Spill/private/SLM values in TU reports are bytes reported by IGC.
TU | kernels | max GRF | any SPILL | SIMD widths | min threads/XVE | compile wall seconds | xe-report rc
fattn-tile-instance-dkq64-dv64 | 140 | 128 | yes | 16,32 | 8 | 46.62 | 1
fattn-tile-instance-dkq128-dv128 | 140 | 128 | yes | 16,32 | 8 | 108.12 | 1
fattn-tile-instance-dkq256-dv256 | 140 | 128 | yes | 16,32 | 8 | 235.13 | 1
fattn-vec-instance-f16-f16 | 116 | 256 | yes | 16,32 | 4 | 53.60 | 1
fattn-vec-instance-q8_0-q8_0 | 116 | 256 | yes | 16,32 | 4 | 66.65 | 1
wall_seconds=513.17 actual_exit=0

Static IGC report: device acm-g10 (Intel Arc A770); no runtime measurements.
llama.cpp commit: 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
Flags: compile database -I/-isystem/-D/-std=; xe-report AOT -O2; extra args: (none; default GRF).
Spill/private/SLM values in TU reports are bytes reported by IGC.
TU | kernels | max GRF | any SPILL | SIMD widths | min threads/XVE | compile wall seconds | xe-report rc
fattn-tile-instance-dkq64-dv64 | 140 | 128 | yes | 16,32 | 8 | 45.34 | 1
fattn-tile-instance-dkq128-dv128 | 140 | 128 | yes | 16,32 | 8 | 111.05 | 1
fattn-tile-instance-dkq256-dv256 | 140 | 128 | yes | 16,32 | 8 | 244.51 | 1
fattn-vec-instance-f16-f16 | 116 | 256 | yes | 16,32 | 4 | 53.30 | 1
fattn-vec-instance-q8_0-q8_0 | 116 | 256 | yes | 16,32 | 4 | 66.35 | 1
wall_seconds=523.64 actual_exit=0

Required checks and applicability
---------------------------------
Fresh independent pin clone + 0001-0005, without CPU patch/diagnostics:
  git apply --check <F candidate>: exit 0
  git apply --check <S candidate>: exit 0
Source audit, fresh-stack patch checks, actual F git diff hunks and whitespace
checks are archived in patch-validation.log.gz; preparation asserts non-fast
projection identity. Scratch-only diagnostic diffs are archived for all clones.
bin/check-forbidden.sh: ok
exit=0
shellcheck -x bin/*.sh: exit=0, no diagnostics
Repo checks (host compiler; A770_TOOLS unset and local tools/ absent;
nine xe-report fixture tests skipped):
  cmake -S . -B /tmp/p11c-a770-repo-build -G Ninja
  cmake --build /tmp/p11c-a770-repo-build -j 2
  ctest --test-dir /tmp/p11c-a770-repo-build --output-on-failure
100% tests passed out of 28
Total Test time (real) =  53.91 sec
wall_seconds=61.20 exit=0

Not verified
------------
A770 numerical correctness, driver work-group limits, runtime memory behavior
or speed; real-model V statistics; full test-backend-ops coverage; oneDNN;
non-FAST_FP16 runtime; skipped repo static tests. Biased V remains a stress
case. No candidate enters the numbered series and neither is chosen here.

Exact filters
-------------
P='((^hsk=(64|128|256),hsv=(64|128|256),nh=[124],nr23=\[(1|2|4|6|8|12),1\],kv=(113|128|256|512|1024|4096),nb=(1|2|3|4|16|32),mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=(f16|q8_0|q4_0),type_V=(f16|q8_0|q4_0),permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$)|(^hsk=(64|128|256),hsv=(64|128|256),nh=(1|8),nr23=\[8,1\],kv=4096,nb=(1|4),mask=1,sinks=0,max_bias=(0|8)\.000000,logit_softcap=0\.000000,prec=f32,type_K=(f16|q8_0),type_V=(f16|q8_0),permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=(0|512)$))|(^hsk=64,hsv=64,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=64,hsv=64,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=64,hsv=64,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q4_0,type_V=q4_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=128,hsv=128,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=128,hsv=128,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=128,hsv=128,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q4_0,type_V=q4_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=256,hsv=256,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=256,hsv=256,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=256,hsv=256,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q4_0,type_V=q4_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=128,hsv=128,nh=4,nr23=\[12,1\],kv=512,nb=3,mask=1,sinks=0,max_bias=8\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=576,hsv=512,nh=1,nr23=\[20,1\],kv=512,nb=1,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=1,n_kv_max=0$|^hsk=256,hsv=256,nh=2,nr23=\[16,1\],kv=113,nb=1,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$)'
P6='nr23=\[[3567],1\]'
BIAS='v_bias=2'

Artifacts
---------
*.log.gz: complete stdout/stderr and audit records.
commands.json.gz: exact run argv, geometry, LLAMA, exit code and wall time.
wall-times.log.gz: original timing records for all runs and repo checks.
filters.txt.gz: exact quoted filters (also included above).
prepare.py.gz, runs.py.gz, report.py.gz: preparation, orchestration and report extraction.
F/S-candidate.diff.gz: candidate source differences before scratch diagnostics.
base/F/S-scratch-diagnostics.diff.gz: identical print-only scratch changes.
patch-validation.log.gz: fresh-stack apply checks and actual F git diff hunks.
source-audit.log.gz and test-source-audit.log.gz: source identity and provenance.
P/P6-coverage-audit.log.gz: membership of historical and patch-0002 case sets.
split-audit.log.gz: F/base split identity, unchanged S VEC and measured S TILE cap.
static-base/ and static-F/: raw per-TU IGC reports, configure output and summaries.

Manager verification (Claude Opus 5.5, 2026-10-11)
--------------------------------------------------
Rechecked: every printed result line in the base/F/S logs (base 12/16 and
8/16 control; F and S 16/16 at g-a770 and g-min; P 97/97 and P-6 117/117
for F and S at g-cpu and g-a770). Read both candidate diffs. F: the
running-max rescale now acts on VKQ_f32 while VKQ is zero (it is flushed
and zeroed at the end of every flash_attn_tile_iter call), so half2 only
ever holds one KV tile (nbatch_fa = 64 keys). S: only raises the lower
bound of parallel_blocks, still capped by ntiles_KQ; VEC launches pass 0.
Decode-relevant instance for Llama-3.1-8B (D128, ncols1 1, ncols2 4):
F spill 8608 -> 8928 B (+320). Speed cost of either candidate: not run.
