p11b-sycl-fa-fp16-a770-geometry: CPU SYCL numerics, 2026-10-11

Scope and configuration
-----------------------
Worktree branch p11b-sycl-fa-fp16-a770-geometry, base main 0feee30.
llama.cpp pin 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6.
CPU device: Intel(R) Xeon(R) CPU E3-1505M v6 @ 3.00GHz; Intel OpenCL 3.0;
driver 2026.21.7.0.24_160000; 8 compute units; max work-group size 8192.
These are CPU-device numerics using the shipped SYCL FA kernel source and
the prescribed launch geometry. No A770 execution. Timing: not run.
Wall times below include configure/build/preflight/CPU JIT/test execution;
the harness reconfigures each time, triggering substantial relink overhead here.
These are not kernel or GPU performance measurements.
No tolerances, existing cases, or FA arithmetic were changed. No kernel fix.
No installs, network, commits, pushes, or main-checkout writes were used.

A/B scratch clones and diagnostics
----------------------------------
For each variant A and B, run in this worktree:
  export PATH=/home/p51/Github/A770/.venv/bin:$PATH;
  git clone --no-hardlinks --no-checkout third_party/llama.cpp third_party/p11b/<variant>
  git -C third_party/p11b/<variant> checkout --detach 79e2e74
  git -C third_party/p11b/<variant> apply "$PWD/third_party/patches/test-only/0001-sycl-allow-cpu-device-for-tests.patch"
  git -C third_party/p11b/<variant> apply "$PWD/third_party/patches/0005-tests-biased-V-FA-cases-with-FP32-CPU-reference.patch"
Apply <variant>-scratch-diagnostics.diff to that patched clone. The two diffs
are byte-identical and contain only the prescribed host [FA-PB] print and
the passing NMSE print in test-backend-ops. They are never a shipped patch.
A: default SYCL_CPU_F16 (ON). B: SYCL_CPU_F16=OFF, VEC FP32; TILE still half2.
0001-0004 are absent from numerical clones. 0005 enables the FP32 CPU reference.
VEC/TILE headers are byte-identical to the pin (source-audit.log.gz).
Inputs use upstream random initialization and differ across runs; A/B
and geometry comparisons are not paired experiments on identical tensors.
Both configurations use Release -O3 -DNDEBUG, optimized spir64 JIT,
per-kernel images, oneDNN and Level Zero OFF, target test-backend-ops, -j 8.
B was first configured ON without building, then the harness reconfigured
the existing cache OFF. B-cache-ON.log.gz and configuration-audit.log.gz
verify the cache change. SYCL_CPU_F16=invalid exits 2 before patch/build.
The test-only patch was regenerated with git diff from A before 0005 and
diagnostics; the final ggml-sycl.cpp hunk uses git diff -U1 context.
Its only extension is CPU geometry parsing/printing, guarded
by GGML_SYCL_TEST_CPU=1 and device.is_cpu(), after max_wg_per_cu assignment.
Each override accepts positive decimal integers representable as int; invalid,
empty, zero, negative or overflowing values preserve that field default.
GPU devices retain the original initialization path.

Commands and cache
------------------
Start every shell command with:
  export PATH=/home/p51/Github/A770/.venv/bin:$PATH;
  export A770_TOOLS=/home/p51/Github/A770/tools
  export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=/tmp/p11b-sycl-cache
  export GGML_SYCL_MKL_FA_DEBUG=1
  unset SYCL_PROGRAM_COMPILE_OPTIONS SYCL_PROGRAM_BUILD_OPTIONS
The harness sets Intel OpenCL CPU selection, GGML_SYCL_TEST_CPU=1, and
OCL_ICD_VENDORS/LD_LIBRARY_PATH under $A770_TOOLS/oneapi-2026.1. Its preflight
requires the validated CPU marker and Backend <n>/<n>: SYCL0.
Set LLAMA=$PWD/third_party/p11b/A or B; unset SYCL_CPU_F16 for A, OFF for B.
Geometry environment:
  g-cpu: unset GGML_SYCL_TEST_NSM GGML_SYCL_TEST_MAX_WG_PER_CU
  g-a770: export GGML_SYCL_TEST_NSM=32 GGML_SYCL_TEST_MAX_WG_PER_CU=2
  g-min: export GGML_SYCL_TEST_NSM=32 GGML_SYCL_TEST_MAX_WG_PER_CU=1
For every A/B geometry:
  /usr/bin/time -f 'wall_seconds=%e exit=%x' -o <run>.time \
    bin/sycl-cpu-tbo.sh -o FLASH_ATTN_EXT -p 'v_bias=2' > <run>.log 2>&1
For A at g-cpu (default F16, no geometry env) and g-a770:
  /usr/bin/time -f 'wall_seconds=%e exit=%x' -o <run>.time \
    bin/sycl-cpu-tbo.sh -o FLASH_ATTN_EXT -p "$P" > <run>.log 2>&1
P copied verbatim from bench/results/sycl-cpu-fa-baseline-20261010/README.txt:
P='((^hsk=(64|128|256),hsv=(64|128|256),nh=[124],nr23=\[(1|2|4|6|8|12),1\],kv=(113|128|256|512|1024|4096),nb=(1|2|3|4|16|32),mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=(f16|q8_0|q4_0),type_V=(f16|q8_0|q4_0),permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$)|(^hsk=(64|128|256),hsv=(64|128|256),nh=(1|8),nr23=\[8,1\],kv=4096,nb=(1|4),mask=1,sinks=0,max_bias=(0|8)\.000000,logit_softcap=0\.000000,prec=f32,type_K=(f16|q8_0),type_V=(f16|q8_0),permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=(0|512)$))|(^hsk=64,hsv=64,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=64,hsv=64,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=64,hsv=64,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q4_0,type_V=q4_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=128,hsv=128,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=128,hsv=128,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=128,hsv=128,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q4_0,type_V=q4_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=256,hsv=256,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=256,hsv=256,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=256,hsv=256,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q4_0,type_V=q4_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=128,hsv=128,nh=4,nr23=\[12,1\],kv=512,nb=3,mask=1,sinks=0,max_bias=8\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=576,hsv=512,nh=1,nr23=\[20,1\],kv=512,nb=1,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=1,n_kv_max=0$|^hsk=256,hsv=256,nh=2,nr23=\[16,1\],kv=113,nb=1,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$)'
Exact run orchestration and P are archived in commands.log.gz.
Before the first numerical run, 219 files were copied read-only from
/tmp/p11-sycl-cache into the new /tmp/p11b-sycl-cache; SHA256 verified.
No existing entry was overwritten. SYCL cache keys select matching program
inputs/options; this reuses CPU JIT work without changing source or math.
Provenance, per-file hashes and sizes: cache-seed.log.gz.

Per-case biased-V results
-------------------------
D=128/256, kv=16384/32768, nb=1/4/16/512; nh=1, G=4, no mask, F16 K/V,
KV views, V+2 on channels 0..7. Unchanged NMSE tolerance = 0.000500000.
Route and p/T/tile are measured [FA-DISP]/[FA-PB] values, not inferred routing.
keys/acc is source-derived: TILE kv/p, VEC kv/(16*p), as explained in p11.
VEC has 16 warp/lane partial sums per block; kv/p is only keys per VEC block.
All selected shapes are divisible. T=ntiles_total; tile=nbatch_fa.
NMSE is printed to nine decimal places; printed zero is not proof of exact zero.

A g-cpu: effective nsm=8, max_wg_per_cu=1024
D    kv     nb   route p    T    tile keys/acc result NMSE
128  16384  1    VEC   128  4    128  8        PASS   0.000000000
128  16384  4    TILE  256  4    64   64       PASS   0.000000003
128  16384  16   TILE  256  4    64   64       PASS   0.000000003
128  16384  512  TILE  256  512  64   64       PASS   0.000000004
128  32768  1    VEC   256  4    128  8        PASS   0.000000000
128  32768  4    TILE  512  4    64   64       PASS   0.000000002
128  32768  16   TILE  512  4    64   64       PASS   0.000000002
128  32768  512  TILE  512  512  64   64       PASS   0.000000002
256  16384  1    VEC   64   4    256  16       PASS   0.000000000
256  16384  4    TILE  256  4    64   64       PASS   0.000000003
256  16384  16   TILE  256  4    64   64       PASS   0.000000004
256  16384  512  TILE  256  512  64   64       PASS   0.000000004
256  32768  1    VEC   128  4    256  16       PASS   0.000000000
256  32768  4    TILE  512  4    64   64       PASS   0.000000002
256  32768  16   TILE  512  4    64   64       PASS   0.000000002
256  32768  512  TILE  512  512  64   64       PASS   0.000000002
Printed result: 16/16 tests passed.
Counted: 16 passed / 0 failed / 0 unsupported.
VEC: 4/4; TILE: 12/12.
wall time: wall_seconds=690.65 exit=0; actual_exit=0

A g-a770: effective nsm=32, max_wg_per_cu=2
D    kv     nb   route p    T    tile keys/acc result NMSE
128  16384  1    VEC   16   4    128  64       PASS   0.000000005
128  16384  4    TILE  16   4    64   1024     PASS   0.000008029
128  16384  16   TILE  16   4    64   1024     PASS   0.000007889
128  16384  512  TILE  2    512  64   8192     FAIL   0.442040216
128  32768  1    VEC   16   4    128  128      PASS   0.000000010
128  32768  4    TILE  16   4    64   2048     PASS   0.000156295
128  32768  16   TILE  16   4    64   2048     PASS   0.000174875
128  32768  512  TILE  2    512  64   16384    FAIL   2.306230598
256  16384  1    VEC   16   4    256  64       PASS   0.000000005
256  16384  4    TILE  16   4    64   1024     PASS   0.000008530
256  16384  16   TILE  16   4    64   1024     PASS   0.000008250
256  16384  512  TILE  2    512  64   8192     FAIL   0.450525039
256  32768  1    VEC   16   4    256  128      PASS   0.000000008
256  32768  4    TILE  16   4    64   2048     PASS   0.000146585
256  32768  16   TILE  16   4    64   2048     PASS   0.000143211
256  32768  512  TILE  2    512  64   16384    FAIL   2.290794069
Printed result: 12/16 tests passed.
Counted: 12 passed / 4 failed / 0 unsupported.
VEC: 4/4; TILE: 8/12.
wall time: Command exited with non-zero status 1; wall_seconds=212.14 exit=1; actual_exit=1

A g-min: effective nsm=32, max_wg_per_cu=1
D    kv     nb   route p    T    tile keys/acc result NMSE
128  16384  1    VEC   8    4    128  128      PASS   0.000000014
128  16384  4    TILE  8    4    64   2048     PASS   0.000199123
128  16384  16   TILE  8    4    64   2048     PASS   0.000173494
128  16384  512  TILE  1    512  64   16384    FAIL   2.305263048
128  32768  1    VEC   8    4    128  256      PASS   0.000000032
128  32768  4    TILE  8    4    64   4096     FAIL   0.025303785
128  32768  16   TILE  8    4    64   4096     FAIL   0.022011706
128  32768  512  TILE  1    512  64   32768    FAIL   11.663531070
256  16384  1    VEC   8    4    256  128      PASS   0.000000017
256  16384  4    TILE  8    4    64   2048     PASS   0.000157343
256  16384  16   TILE  8    4    64   2048     PASS   0.000157762
256  16384  512  TILE  1    512  64   16384    FAIL   2.286076295
256  32768  1    VEC   8    4    256  256      PASS   0.000000026
256  32768  4    TILE  8    4    64   4096     FAIL   0.022588584
256  32768  16   TILE  8    4    64   4096     FAIL   0.023194594
256  32768  512  TILE  1    512  64   32768    FAIL   11.605829454
Printed result: 8/16 tests passed.
Counted: 8 passed / 8 failed / 0 unsupported.
VEC: 4/4; TILE: 4/12.
wall time: Command exited with non-zero status 1; wall_seconds=208.70 exit=1; actual_exit=1

B g-cpu: effective nsm=8, max_wg_per_cu=1024
D    kv     nb   route p    T    tile keys/acc result NMSE
128  16384  1    VEC   128  4    128  8        PASS   0.000000000
128  16384  4    TILE  256  4    64   64       PASS   0.000000004
128  16384  16   TILE  256  4    64   64       PASS   0.000000004
128  16384  512  TILE  256  512  64   64       PASS   0.000000004
128  32768  1    VEC   256  4    128  8        PASS   0.000000000
128  32768  4    TILE  512  4    64   64       PASS   0.000000002
128  32768  16   TILE  512  4    64   64       PASS   0.000000002
128  32768  512  TILE  512  512  64   64       PASS   0.000000002
256  16384  1    VEC   64   4    256  16       PASS   0.000000000
256  16384  4    TILE  256  4    64   64       PASS   0.000000003
256  16384  16   TILE  256  4    64   64       PASS   0.000000004
256  16384  512  TILE  256  512  64   64       PASS   0.000000004
256  32768  1    VEC   128  4    256  16       PASS   0.000000000
256  32768  4    TILE  512  4    64   64       PASS   0.000000002
256  32768  16   TILE  512  4    64   64       PASS   0.000000002
256  32768  512  TILE  512  512  64   64       PASS   0.000000002
Printed result: 16/16 tests passed.
Counted: 16 passed / 0 failed / 0 unsupported.
VEC: 4/4; TILE: 12/12.
wall time: wall_seconds=669.43 exit=0; actual_exit=0

B g-a770: effective nsm=32, max_wg_per_cu=2
D    kv     nb   route p    T    tile keys/acc result NMSE
128  16384  1    VEC   16   4    128  64       PASS   0.000000000
128  16384  4    TILE  16   4    64   1024     PASS   0.000007364
128  16384  16   TILE  16   4    64   1024     PASS   0.000007434
128  16384  512  TILE  2    512  64   8192     FAIL   0.441381825
128  32768  1    VEC   16   4    128  128      PASS   0.000000000
128  32768  4    TILE  16   4    64   2048     PASS   0.000158295
128  32768  16   TILE  16   4    64   2048     PASS   0.000179933
128  32768  512  TILE  2    512  64   16384    FAIL   2.296160712
256  16384  1    VEC   16   4    256  64       PASS   0.000000000
256  16384  4    TILE  16   4    64   1024     PASS   0.000008065
256  16384  16   TILE  16   4    64   1024     PASS   0.000008422
256  16384  512  TILE  2    512  64   8192     FAIL   0.446754796
256  32768  1    VEC   16   4    256  128      PASS   0.000000000
256  32768  4    TILE  16   4    64   2048     PASS   0.000152854
256  32768  16   TILE  16   4    64   2048     PASS   0.000135774
256  32768  512  TILE  2    512  64   16384    FAIL   2.291259336
Printed result: 12/16 tests passed.
Counted: 12 passed / 4 failed / 0 unsupported.
VEC: 4/4; TILE: 8/12.
wall time: Command exited with non-zero status 1; wall_seconds=204.42 exit=1; actual_exit=1

B g-min: effective nsm=32, max_wg_per_cu=1
D    kv     nb   route p    T    tile keys/acc result NMSE
128  16384  1    VEC   8    4    128  128      PASS   0.000000000
128  16384  4    TILE  8    4    64   2048     PASS   0.000179448
128  16384  16   TILE  8    4    64   2048     PASS   0.000175815
128  16384  512  TILE  1    512  64   16384    FAIL   2.314389318
128  32768  1    VEC   8    4    128  256      PASS   0.000000000
128  32768  4    TILE  8    4    64   4096     FAIL   0.021181826
128  32768  16   TILE  8    4    64   4096     FAIL   0.022744529
128  32768  512  TILE  1    512  64   32768    FAIL   11.649662936
256  16384  1    VEC   8    4    256  128      PASS   0.000000000
256  16384  4    TILE  8    4    64   2048     PASS   0.000159709
256  16384  16   TILE  8    4    64   2048     PASS   0.000147140
256  16384  512  TILE  1    512  64   16384    FAIL   2.257099096
256  32768  1    VEC   8    4    256  256      PASS   0.000000000
256  32768  4    TILE  8    4    64   4096     FAIL   0.022242941
256  32768  16   TILE  8    4    64   4096     FAIL   0.022538976
256  32768  512  TILE  1    512  64   32768    FAIL   11.543734483
Printed result: 8/16 tests passed.
Counted: 8 passed / 8 failed / 0 unsupported.
VEC: 4/4; TILE: 4/12.
wall time: Command exited with non-zero status 1; wall_seconds=223.24 exit=1; actual_exit=1

NMSE margin by route (relative to the unchanged 0.000500000 limit)
run                  route passed worst_NMSE  limit_minus_worst
A-g-cpu-biased       VEC   4/4    0.000000000 0.000500000
A-g-cpu-biased       TILE  12/12    0.000000004 0.000499996
A-g-a770-biased      VEC   4/4    0.000000010 0.000499990
A-g-a770-biased      TILE  8/12    2.306230598 -2.305730598
A-g-min-biased       VEC   4/4    0.000000032 0.000499968
A-g-min-biased       TILE  4/12    11.663531070 -11.663031070
B-g-cpu-biased       VEC   4/4    0.000000000 0.000500000
B-g-cpu-biased       TILE  12/12    0.000000004 0.000499996
B-g-a770-biased      VEC   4/4    0.000000000 0.000500000
B-g-a770-biased      TILE  8/12    2.296160712 -2.295660712
B-g-min-biased       VEC   4/4    0.000000000 0.000500000
B-g-min-biased       TILE  4/12    11.649662936 -11.649162936

Geometry verification
---------------------
All 32 g-a770 [FA-PB] rows (A and B) match the biased-V Numeric A770 table
in p11: VEC p=16; TILE nb=4/16 p=16; TILE nb=512 p=2. No formula change.
g-cpu prints nsm=8, m=1024; TILE p=kv/64 and exactly 64 keys/accumulator.
g-a770 prints nsm=32, m=2; TILE sums 1024/2048 keys at nb=4/16 and
8192/16384 at nb=512 for 16K/32K KV. g-min prints nsm=32, m=1;
TILE p=8 at nb=4/16 and p=1 at nb=512: 2048/4096 and 16384/32768 keys.
The source uses W=nsm*m; p starts at min(m,ceil(kv/nbatch_fa)), increases
only for strictly improved wave efficiency, and stops after >=95% efficiency
before the next wave. See pinned fattn-common.hpp:1061-1104 and p11 README.

p6a P subset
------------
A-g-cpu-P: 85/85 tests passed; routes {'MKL': 10, 'TILE': 62, 'VEC': 13}.
Worst printed passing P NMSE: 0.000006714; printed limits: ['0.000500000'].
wall time: wall_seconds=189.45 exit=0; actual_exit=0
0 failures / 0 unsupported.
A-g-a770-P: 85/85 tests passed; routes {'MKL': 10, 'TILE': 62, 'VEC': 13}.
Worst printed passing P NMSE: 0.000012920; printed limits: ['0.000500000'].
wall time: wall_seconds=184.18 exit=0; actual_exit=0
0 failures / 0 unsupported.
CPU is the reference; its separate backend entry is skipped by -b SYCL0.
The printed 1/2 or 2/2 backend count is not evidence of two backends executing.
Harness preflight 0/0 summaries are excluded from all numerical counts.

Validation
----------
Fresh local pin clones third_party/p11b/check and check-final:
  git apply --check <updated-test-only-patch>: exit 0 on the fresh pin.
  Apply 0001-0005 in order, then git apply --check <updated-test-only-patch>: exit 0.
  Apply the test-only patch, git diff --check: exit 0.
A/B reverse applicability and git diff --check: exit 0. Full commands/results
are in patch-validation.log.gz. A fresh replay of the final test-only patch,
0005, and each archived diagnostic diff reconstructs A/B source byte-for-byte
(final-validation.log.gz). No git commands wrote the project worktree.
Runtime empty-filter checks: unset values, each independent override, leading
zeros, empty/zero/negative/non-integer/trailing-text/overflow inputs: PASS.
These validate host metadata without launching numerical kernels.
Runtime parser checks and F16 configuration flags: configuration-audit.log.gz.
  bin/check-forbidden.sh: ok, exit 0.
  shellcheck -x bin/*.sh: exit 0, no diagnostics.
  cmake -S . -B /tmp/p11b-repo-build -G Ninja
  cmake --build /tmp/p11b-repo-build -j 8
  ctest --test-dir /tmp/p11b-repo-build --output-on-failure
    100% tests passed out of 28; 19 passed, 9 skipped; exit 0.
    Total Test time (real) = 53.21 sec.
Skipped: xe_report_dpas_simd8, xe_report_large_grf, xe_report_dpas_simd16_silent,
xe_report_spill, xe_report_fp64, xe_report_block2d, xe_report_slm,
xe_report_sycl_joint_matrix, xe_report_compile_commands. See repo-checks.log.gz.

Wall times (seconds; overhead-inclusive, no timing benchmark)
-----------------------------------------------------------
phase                         wall_seconds exit actual_exit
A-g-a770-P                    184.18       0    0
A-g-a770-biased               212.14       1    1
A-g-cpu-P                     189.45       0    0
A-g-cpu-biased                690.65       0    0
A-g-min-biased                208.70       1    1
B-cache-ON                    7.50         0    0
B-g-a770-biased               204.42       1    1
B-g-cpu-biased                669.43       0    0
B-g-min-biased                223.24       1    1
repo-checks                   58.52        0    0

Interpretation and limits
-------------------------
A-g-cpu-biased VEC: 4/4 pass against FP32; worst printed NMSE across all cases 0.000000000.
A-g-cpu-biased TILE: 12/12 pass against FP32; worst printed NMSE across all cases 0.000000004.
A-g-a770-biased VEC: 4/4 pass against FP32; worst printed NMSE across all cases 0.000000010.
A-g-a770-biased TILE: 8/12 pass against FP32; worst printed NMSE across all cases 2.306230598.
A-g-min-biased VEC: 4/4 pass against FP32; worst printed NMSE across all cases 0.000000032.
A-g-min-biased TILE: 4/12 pass against FP32; worst printed NMSE across all cases 11.663531070.
B-g-cpu-biased VEC: 4/4 pass against FP32; worst printed NMSE across all cases 0.000000000.
B-g-cpu-biased TILE: 12/12 pass against FP32; worst printed NMSE across all cases 0.000000004.
B-g-a770-biased VEC: 4/4 pass against FP32; worst printed NMSE across all cases 0.000000000.
B-g-a770-biased TILE: 8/12 pass against FP32; worst printed NMSE across all cases 2.296160712.
B-g-min-biased VEC: 4/4 pass against FP32; worst printed NMSE across all cases 0.000000000.
B-g-min-biased TILE: 4/12 pass against FP32; worst printed NMSE across all cases 11.649662936.
A g-a770 worst passing TILE NMSE 0.000174875; remaining margin 0.000325125.
B g-a770 worst passing TILE NMSE 0.000179933; remaining margin 0.000320067.
Any failing rows are findings under the prescribed CPU geometry; no arithmetic
or tolerance change was made in response. A/B differ by the F16 configuration
switch; TILE remains half2 in both. No fix is proposed or decided.
Not verified: A770 numerics/runtime behavior; A770 max_work_group_size (1024
is an assumption, bracketed by m=1); real-model V statistics; any fix;
GPU/kernel timing or performance; oneDNN; full upstream test suite; skipped checks.
Upstream MKL debug output includes incidental time_us fields; no timing
benchmark was run or interpreted, and those fields are not performance evidence.
CPU emulation of the launch split does not establish A770 numerical behavior.

Artifacts
---------
README.txt: commands, exact P, all six tables, margins, geometry check, wall times.
A/B-scratch-diagnostics.diff: the two scratch-only diagnostic diffs.
A/B-source-diff.log.gz: complete source differences relative to the pin.
<variant>-<geometry>-biased.log.gz and A-<geometry>-P.log.gz: unabridged harness output.
B-cache-ON.log.gz: preconfiguration proving the subsequent ON-to-OFF cache change.
patch-validation.log.gz, source-audit.log.gz, configuration-audit.log.gz,
cache-seed.log.gz, commands.log.gz, wall-times.log.gz, repo-checks.log.gz,
final-validation.log.gz: evidence.

Manager verification (Claude Opus 5.5, 2026-10-11)
--------------------------------------------------
Rechecked from the logs: the geometry line and [FA-PB] parallel_blocks
per case in all six biased runs (g-a770 matches p11's A770 table: VEC 16,
TILE 16 for nb 4/16, TILE 2 for nb 512); every A g-a770 NMSE; the test-only
patch applying alone and in the stack 0001-0005 + test-only (fresh clone).
TILE NMSE grows with keys per accumulator: 1024 -> ~8e-6, 2048 -> ~1.5e-4
(30-35% of tolerance), 4096 (g-min) -> fail, 8192 -> 0.44, 16384 -> 2.3.
B equals A for TILE as expected (F16 OFF does not change TILE's half2).
Real-model relevance: masked Llama-3.1-8B decode at 32K takes TILE with
4096 keys per accumulator under m=2 (p11 table), the failing regime here;
whether real V statistics reach it is unmeasured.
