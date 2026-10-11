p6b-sycl-tile-padded-gqa: CPU-device correctness for patches 0001/0002 (complete; finished by the manager)

Step 1 (test addition only; kernel unpatched):
  112/112 tests passed; 112 passed / 0 failed / 0 unsupported
  2/2 backends passed (CPU backend entry skipped; used as comparison reference)
  wall_seconds=3710.74 exit=0

N='^hsk=(128|256),hsv=(128|256),nh=2,nr23=\[(3|5|6|7),1\],kv=4096,nb=(1|2|3|4|8|16),mask=1,sinks=(0|1),max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=(f16|q8_0),type_V=(f16|q8_0),permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$'

Common environment:
  export PATH=/home/p51/Github/A770/.venv/bin:$PATH;
  export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=/tmp/p6b-sycl-tile-padded-gqa-sycl-cache

Step 1 command:
  /usr/bin/time -f 'wall_seconds=%e exit=%x' -o /tmp/p6b-step1.time env -u SYCL_PROGRAM_COMPILE_OPTIONS -u SYCL_PROGRAM_BUILD_OPTIONS GGML_SYCL_MKL_FA_DEBUG=1 bin/sycl-cpu-tbo.sh -o FLASH_ATTN_EXT -p "$N" > /tmp/p6b-step1.log 2>&1

CPU correctness evidence only. A770 timing: not run.
AOT/register report for acm-g10: not run. Real-GPU numerics: not run.

Step 2 (indexing and phantom guards; original dispatch):
  197/197 tests passed; 197 passed / 0 failed / 0 unsupported
  Existing cases: 85; new cases: 112.
  wall_seconds=3061.70 exit=0
  The exact p6a P also matches 12 of the added GQA-6 cases; the union runs each test node once.

Step 3 (indexing, phantom guards and padded dispatch):
  197/197 tests passed; 197 passed / 0 failed / 0 unsupported
  Existing cases: 85; new cases: 112.
  wall_seconds=3285.23 exit=0 actual_exit=0
  FA-DISP: TILE 174, VEC 13, MKL 10.
  Negative controls are in progress.

Negative control a (output/meta guard removed):
  0 passed, 1 failed, 7 unfinished of 8 selected.
  One numerical failure, followed by SIGSEGV during the second case.
  wall_seconds=1448.95 actual_exit=139 (GNU time %x printed 0 for the signal).
Recovery a (source restored byte-for-byte):
  8/8 tests passed; wall_seconds=498.65 actual_exit=0.
  Control b is in progress.

Negative control b (sink guard removed):
  8/8 tests passed; wall_seconds=1832.56 actual_exit=0.
  Required FAIL was not observed. Phantom sink changes affect discarded private columns.
  Sink guard restored; recovery b is in progress.

Recovery b:
  8/8 tests passed; wall_seconds=318.96 actual_exit=0.
  Control c is in progress; this quotient is mathematically equal to kv_head under the new indexing.

Negative control c, as ticketed (K/V bases from head0 / gqa_ratio):
  Not observable. Padding implies ncols2 >= G, so ntiles_z_gqa = 1 and
  head0 / gqa_ratio == kv_head exactly. The worker saw 4/4 D=128 cases pass
  (neg-c-original.log.gz). Then its Codex session was ended by the provider's
  content filter ("flagged for possible cybersecurity risk") mid-run, with the
  kernel left mutated. The manager restored it byte-for-byte from the worker's
  saved good copy (/tmp/p6b-good-tile.hpp) and finished the ticket.

Manager steps (Claude Opus 5.5), same environment and filters (line 4 = the
eight nb=3 sinks=1 F16 cases, line 3 = p6a subset + new block):
  Control c' (replaces c): padding on, head0 from the pre-patch formula
    const int head0 = z * ncols2 - sequence * ne02;
  -> first case FAIL: NaN at index 384 (SYCL0=-nan CPU=-0.009326),
     hsk=128 nr23=[3,1] nb=3 sinks=1; then SIGSEGV. exit=139, 644 s.
  Recovery c' (source restored byte-for-byte): 8/8 passed, exit 0, 410 s.
  Full recovery, line 3: 197/197 tests passed, exit 0, 281 s (JIT cache warm).
    FA-DISP: TILE 174, VEC 13, MKL 10.

Summary of controls:
  a (phantom output/meta write)        FAIL (ERR 0.29, then SIGSEGV) -> recovered
  b (phantom sink guard)               not observable numerically: a phantom
                                        column's sink only affects that column,
                                        which is never written; the guard
                                        prevents a read past the end of sinks
                                        for the last KV head. Verified by review.
  c (K/V base head0/G)                 not observable (algebraically equal)
  c' (pre-patch head0 formula)         FAIL (NaN, then SIGSEGV) -> recovered

Patches: 0001 (fattn-tile.hpp only) and 0002 (test block only) apply to a
clean clone at 79e2e74. The series 0001-0004 also applies cleanly, and its
fattn-tile.hpp is byte-identical to the tested kernel. The test-only CPU patch
is in neither.
Static acm-g10 check: ../p6b-static-acm-g10-20261010/comparison.txt.
