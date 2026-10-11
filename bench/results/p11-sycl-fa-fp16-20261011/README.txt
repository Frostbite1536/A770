p11-sycl-fa-fp16-accumulation: Intel OpenCL CPU numerics, 2026-10-11

Scope
-----
llama.cpp pin 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6; A770 worktree
p11-sycl-fa-fp16-accumulation based on main 2d000d7. No A770 execution.
Timing benchmarks: not run. Wall times below measure build/test execution,
including CPU JIT; they are not kernel or A770 performance measurements.
No tolerances or existing cases were changed; no production kernel fix was made. No installs,
network, commits, pushes, or changes to the main checkout were used.

Patch 0005
----------
third_party/patches/0005-tests-biased-V-FA-cases-with-FP32-CPU-reference.patch
is ported from the P100 project, patch 0003. Its complete diff body is
byte-identical to that patch: only the explanatory header differs.
v_bias defaults to zero and appears in case names only when nonzero.
Sixteen cases: D=128/256, kv=16384/32768, nb=1/4/16/512; one KV head,
four Q heads, F16 K/V, KV views, no mask/sinks/bias/softcap, V+2 on channels
0..7. CPU F16 output accumulator branches are gated with !params->use_ref;
the existing conversion/scaling/multiply-add branch supplies FP32 reference
accumulation. Existing case error metric is NMSE and tolerance is 0.000500000.
The patch adds no error printing. Passing cases print OK without numeric ERR.

Variants (scratch only, third_party/p11/)
-----------------------------------------
Every clone was made locally with git clone --no-hardlinks --no-checkout
third_party/llama.cpp third_party/p11/<variant>, then git checkout --detach
79e2e74 inside the clone. Apply only the CPU-device test-only patch and 0005.
No 0001-0004 kernel/test stack is included in numerical builds.
Patch application inside each new pin clone:
  git -C "$clone" apply "$PWD/third_party/patches/test-only/0001-sycl-allow-cpu-device-for-tests.patch"
  git -C "$clone" apply "$PWD/third_party/patches/0005-tests-biased-V-FA-cases-with-FP32-CPU-reference.patch"
R's restoration was the byte output of:
  git -C "$clone" show HEAD:ggml/src/ggml-cpu/ops.cpp
written to "$clone/ggml/src/ggml-cpu/ops.cpp".
C's common.hpp edit deleted only the exact line:
  #define SYCL_FAST_FP16  //don't change. remove it will break fattn-tile.hpp building
The preparation asserted that line occurred exactly once before deleting it.
R: GGML_SYCL_F16=ON; restore ggml-cpu/ops.cpp verbatim from the pin after
   applying 0005, retaining biased-V cases and the old FP16 reference.
A: GGML_SYCL_F16=ON; 0005, including FP32 reference; shipped accumulation.
B: GGML_SYCL_F16=OFF; 0005; VEC takes its FP32 path, TILE still uses half2.
C: GGML_SYCL_F16=ON; 0005; remove exactly the #define SYCL_FAST_FP16 line
   from common.hpp. No fattn-tile.hpp change, even if compilation fails.
Full source diffs are archived per clone in <variant>-source-diff.log.gz.
R is a reference diagnostic, not an accuracy oracle. R/A/B use the same
TILE source and half2 VKQ path; the F16 switch affects other kernel
arithmetic too, so B
is the prescribed configuration comparison, not an isolated instruction swap.
Inputs use upstream random initialization and are not identical across runs.

Measured CPU device: Intel(R) Xeon(R) CPU E3-1505M v6 @ 3.00GHz,
Intel(R) OpenCL 3.0, driver 2026.21.7.0.24_160000, x86_64; fp16;
subgroup sizes 4/8/16/32/64. OpenCL device-info queries confirm 8 compute
units and maximum work-group size 8192 (device-info.log.gz).
Metadata command, with the common CPU environment below:
  "$O/bin/sycl-ls" --verbose
The two launch properties were queried with clGetDeviceInfo via Python
ctypes against $O/lib/libOpenCL.so.1, keys 0x1002 (cl_uint) and 0x1004
(size_t); CPU device type 2 from clGetDeviceIDs. No kernel was launched
by this metadata query.

Commands
--------
Start every shell command with:
  export PATH=/home/p51/Github/A770/.venv/bin:$PATH;
  export A770_TOOLS=/home/p51/Github/A770/tools
  O=$A770_TOOLS/oneapi-2026.1
  export OCL_ICD_VENDORS=$O/etc/OpenCL/vendors LD_LIBRARY_PATH=$O/lib
  export ONEAPI_DEVICE_SELECTOR=opencl:cpu GGML_SYCL_TEST_CPU=1
  export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=/tmp/p11-sycl-cache
  export GGML_SYCL_MKL_FA_DEBUG=1
  unset SYCL_PROGRAM_COMPILE_OPTIONS SYCL_PROGRAM_BUILD_OPTIONS

The repository harness bin/sycl-cpu-tbo.sh forces GGML_SYCL_F16=ON when
reconfiguring, so it cannot preserve B. The same configure/build/preflight
steps were executed directly for every clone, changing only F16 for B:
  clone=$PWD/third_party/p11/<variant>
  f16=ON # OFF for B
  /usr/bin/time -f 'wall_seconds=%e exit=%x' -o <variant>-configure.time \
    cmake -S "$clone" -B "$clone/build-sycl-cpu" -G Ninja \
    -DGGML_SYCL=ON -DGGML_SYCL_F16=$f16 -DGGML_SYCL_DNN=OFF \
    -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=OFF -DGGML_SYCL_DEVICE_ARCH= \
    -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$O/bin/icx" -DCMAKE_CXX_COMPILER="$O/bin/icpx" \
    -DCMAKE_PREFIX_PATH="$O" > <variant>-configure.log 2>&1
  /usr/bin/time -f 'wall_seconds=%e exit=%x' -o <variant>-build.time \
    cmake --build "$clone/build-sycl-cpu" --target test-backend-ops -j 8 \
    > <variant>-build.log 2>&1
  /usr/bin/time -f 'wall_seconds=%e exit=%x' -o <variant>-preflight.time \
    "$clone/build-sycl-cpu/bin/test-backend-ops" \
    -b SYCL0 -o FLASH_ATTN_EXT -p '^$' > <variant>-preflight.log 2>&1
Require preflight exit 0, the marker
  GGML_SYCL_TEST_CPU: SYCL0 is Intel OpenCL CPU:
and Backend <n>/<n>: SYCL0. A backend skipped by the runner is not a pass.
Release -O3 -DNDEBUG, spir64 JIT, per-kernel images, oneDNN and Level Zero OFF;
optimization enabled. All builds target test-backend-ops and its dependencies.
During A's biased run, the requested /tmp/p11-sycl-cache was seeded with
completed cache entries from /tmp/p6a-sycl-cpu-cache (read-only). Existing
cache directories were never overwritten; overlapping .src input entries
were checked byte-identical. The provenance/counts/hashes are archived in
cache-seed.log.gz. SYCL selects cached kernels by their input/option keys;
this reuses prior JIT work without changing source or math. A/R/B/C all
use /tmp/p11-sycl-cache. Wall times are not comparable performance data.

Exact P filter (copied verbatim from p6a's baseline README):
P='((^hsk=(64|128|256),hsv=(64|128|256),nh=[124],nr23=\[(1|2|4|6|8|12),1\],kv=(113|128|256|512|1024|4096),nb=(1|2|3|4|16|32),mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=(f16|q8_0|q4_0),type_V=(f16|q8_0|q4_0),permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$)|(^hsk=(64|128|256),hsv=(64|128|256),nh=(1|8),nr23=\[8,1\],kv=4096,nb=(1|4),mask=1,sinks=0,max_bias=(0|8)\.000000,logit_softcap=0\.000000,prec=f32,type_K=(f16|q8_0),type_V=(f16|q8_0),permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=(0|512)$))|(^hsk=64,hsv=64,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=64,hsv=64,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=64,hsv=64,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q4_0,type_V=q4_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=128,hsv=128,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=128,hsv=128,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=128,hsv=128,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q4_0,type_V=q4_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=256,hsv=256,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=256,hsv=256,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=256,hsv=256,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q4_0,type_V=q4_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=128,hsv=128,nh=4,nr23=\[12,1\],kv=512,nb=3,mask=1,sinks=0,max_bias=8\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=576,hsv=512,nh=1,nr23=\[20,1\],kv=512,nb=1,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=1,n_kv_max=0$|^hsk=256,hsv=256,nh=2,nr23=\[16,1\],kv=113,nb=1,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$)'

For each successfully built clone, run both filters separately:
  /usr/bin/time -f 'wall_seconds=%e exit=%x' -o <variant>-biased.time \
    "$clone/build-sycl-cpu/bin/test-backend-ops" \
    -b SYCL0 -o FLASH_ATTN_EXT -p 'v_bias=2' > <variant>-biased.log 2>&1
  /usr/bin/time -f 'wall_seconds=%e exit=%x' -o <variant>-P.time \
    "$clone/build-sycl-cpu/bin/test-backend-ops" \
    -b SYCL0 -o FLASH_ATTN_EXT -p "$P" > <variant>-P.log 2>&1
All .log files below are archived as .log.gz; .time files are incorporated
in this README and wall-times.log.gz. Full orchestration is in commands.log.gz.

Measured runs
-------------
Clone R: biased-V
D    kv     nb   route  result  ERR (NMSE; failure prints x > y)
128  16384  1    VEC    FAIL   0.358595148 > 0.000500000
128  16384  4    TILE   FAIL   0.363269030 > 0.000500000
128  16384  16   TILE   FAIL   0.364008403 > 0.000500000
128  16384  512  TILE   FAIL   0.363702114 > 0.000500000
128  32768  1    VEC    FAIL   0.598344619 > 0.000500000
128  32768  4    TILE   FAIL   0.599947175 > 0.000500000
128  32768  16   TILE   FAIL   0.603352191 > 0.000500000
128  32768  512  TILE   FAIL   0.600097358 > 0.000500000
256  16384  1    VEC    FAIL   0.386946722 > 0.000500000
256  16384  4    TILE   FAIL   0.360296393 > 0.000500000
256  16384  16   TILE   FAIL   0.357734289 > 0.000500000
256  16384  512  TILE   FAIL   0.362854280 > 0.000500000
256  32768  1    VEC    FAIL   0.624953355 > 0.000500000
256  32768  4    TILE   FAIL   0.595884366 > 0.000500000
256  32768  16   TILE   FAIL   0.601016954 > 0.000500000
256  32768  512  TILE   FAIL   0.597636767 > 0.000500000
Printed summary: 0/16 tests passed.
Counted cases: 0 passed / 16 failed / 0 unsupported.
Routes: {'VEC': 4, 'TILE': 12}.
P: 85/85 tests passed; routes {'MKL': 10, 'TILE': 62, 'VEC': 13}.
The separate CPU backend entry is skipped by -b SYCL0. The printed
1/2 backends count is not evidence that two backends executed.

Clone A: biased-V
D    kv     nb   route  result  ERR (NMSE; failure prints x > y)
128  16384  1    VEC    PASS   not printed
128  16384  4    TILE   PASS   not printed
128  16384  16   TILE   PASS   not printed
128  16384  512  TILE   PASS   not printed
128  32768  1    VEC    PASS   not printed
128  32768  4    TILE   PASS   not printed
128  32768  16   TILE   PASS   not printed
128  32768  512  TILE   PASS   not printed
256  16384  1    VEC    PASS   not printed
256  16384  4    TILE   PASS   not printed
256  16384  16   TILE   PASS   not printed
256  16384  512  TILE   PASS   not printed
256  32768  1    VEC    PASS   not printed
256  32768  4    TILE   PASS   not printed
256  32768  16   TILE   PASS   not printed
256  32768  512  TILE   PASS   not printed
Printed summary: 16/16 tests passed.
Counted cases: 16 passed / 0 failed / 0 unsupported.
Routes: {'VEC': 4, 'TILE': 12}.
P: 85/85 tests passed; routes {'MKL': 10, 'TILE': 62, 'VEC': 13}.
The separate CPU backend entry is skipped by -b SYCL0. The printed
2/2 backends count is not evidence that two backends executed.

Clone B: biased-V
D    kv     nb   route  result  ERR (NMSE; failure prints x > y)
128  16384  1    VEC    PASS   not printed
128  16384  4    TILE   PASS   not printed
128  16384  16   TILE   PASS   not printed
128  16384  512  TILE   PASS   not printed
128  32768  1    VEC    PASS   not printed
128  32768  4    TILE   PASS   not printed
128  32768  16   TILE   PASS   not printed
128  32768  512  TILE   PASS   not printed
256  16384  1    VEC    PASS   not printed
256  16384  4    TILE   PASS   not printed
256  16384  16   TILE   PASS   not printed
256  16384  512  TILE   PASS   not printed
256  32768  1    VEC    PASS   not printed
256  32768  4    TILE   PASS   not printed
256  32768  16   TILE   PASS   not printed
256  32768  512  TILE   PASS   not printed
Printed summary: 16/16 tests passed.
Counted cases: 16 passed / 0 failed / 0 unsupported.
Routes: {'VEC': 4, 'TILE': 12}.
P: 85/85 tests passed; routes {'MKL': 10, 'TILE': 62, 'VEC': 13}.
The separate CPU backend entry is skipped by -b SYCL0. The printed
2/2 backends count is not evidence that two backends executed.

Clone C: did not build. Command exited with non-zero status 1
wall_seconds=225.83 exit=1
First compiler error in emitted build log:
/home/p51/Github/A770/.worktrees/p11-sycl-fa-fp16-accumulation/third_party/p11/C/ggml/src/ggml-sycl/template-instances/../fattn-tile.hpp:714:19: error: static assertion failed due to requirement 'ggml_sycl_fattn_tile_get_config(576, 512, 2 * 16) != 0': kernel config not defined
  714 |     static_assert(ggml_sycl_fattn_tile_get_config(DKQ, DV, ncols1*ncols2) != 0, "kernel config not defined");
      |                   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Biased-V and P: not run; no C numerical result. No TILE edits were made.

Summary
variant biased-V P    VEC/TILE biased routes
R       0/16     85/85  4/12
A       16/16     85/85  4/12
B       16/16     85/85  4/12
C       did not build; numerical runs not performed

A770 parallel_blocks (source calculation only)
----------------------------------------------
Anchors below refer to the unpatched llama.cpp pin, under ggml/src/ggml-sycl/.
No A770 device or kernel was run for this calculation. Routing below
uses the diagnostic configuration (oneDNN OFF); nb=512 has no mask.

nsm = floor(max_compute_units / 16) = 512 / 16 = 32
  ggml-sycl.cpp:239.
m = max_blocks_per_sm = floor(max_work_group_size / max_compute_units)
  ggml-sycl.cpp:248-249; fattn-common.hpp:1061-1062.
The numeric A770 tables assume max_work_group_size=1024, hence m=2.
That driver property has NOT been read from an A770. nsm=32 alone does
not determine the split. To make that dependency explicit, the symbolic
results are given below too; do not substitute the TILE config's occupancy
field for m (launch_fattn uses max_wg_per_cu, not that config field).

T = ceil(nb/ncols1) * ceil(G/ncols2) * KV_heads * sequences
  fattn-common.hpp:1018-1021.
L = ceil(kv / nbatch_fa), W = nsm*m.
Start p=min(m,L). For candidates p..L, calculate
  waves = ceil(T*p/W), efficiency = floor(100*T*p/(waves*W)).
Retain p only on a strict efficiency improvement. Break before considering
another wave once best efficiency >=95% and waves > best_waves.
  fattn-common.hpp:1078-1104. TILE launch has stream_k=false
  (fattn-tile.hpp:1087-1090,1130-1135).
For these shapes the first 100% efficient split is the retained value:
  masked Llama nb=1: p=4*m; nb=16: p=2*m;
  biased-V nb=1/4/16: p=8*m; nb=512: p=m.
These formulas apply here when the key-tile cap L allows that split.

TILE uses nbatch_fa=64 in the FP16 configuration at D=128/256
  fattn-tile.hpp:58-68,177-182; common.hpp:957-959 always returns true
  for fast_fp16_available(cc) used by host configuration.
Masked G=4 uses ncols2=4 (fattn-tile.hpp:1168-1169,1200-1203).
At nb=1: ncols1=1, T=8; at nb=16: ncols1=8, T=16
  fattn-tile.hpp:1080-1091,1128-1137.
With no mask, biased-V TILE uses ncols2=1; ncols1=4/16/4 for
nb=4/16/512, and T=4/4/512. NB=512 remains TILE: no mask excludes
MKL and oneDNN is compiled out. NB=1 takes VEC (fattn.cpp:130,260-267).
No diagnostic forces routing.

TILE keys per accumulator = maximum over split indices b of
  sum(min(nbatch_fa,kv-k) for k=b*nbatch_fa; k<kv;
      k+=p*nbatch_fa).
For these divisible shapes it is exactly kv/p. np=1, so there is no
additional per-column warp split (fattn-tile.hpp:744-748; loops:866-885).
The table is the full-context budget; masked-out keys can contribute zero.

VEC nb=1 scheduling uses nbatch_fa=D (fattn-vec.hpp:608-613), but its
actual key-loop increment is nthreads=128 (fattn-vec.hpp:292-299).
The table lists kv/p keys per block and kv/(16*p) per output accumulator:
warp_size=16, nwarps=8, cpy_nb=16 (common.hpp:794-796), nthreads_V=8,
V_cols_per_iter=2, so 8*2 independent warp/lane sums precede FP32 combine
  fattn-vec.hpp:102-109,133,375-397,532-541. This subdivision matters:
keys per block is not keys per VEC accumulator.

Numeric A770 table, conditional on max_work_group_size=1024 (m=2)
family           D    KV_heads G mask nb   kv    route ncols1 ncols2 T   tile p  keys/block keys/accumulator
Llama-3.1-8B     128  8        4 1    1    4096  TILE  1      4      8   64   8  512        512
Llama-3.1-8B     128  8        4 1    1    16384 TILE  1      4      8   64   8  2048       2048
Llama-3.1-8B     128  8        4 1    1    32768 TILE  1      4      8   64   8  4096       4096
Llama-3.1-8B     128  8        4 1    16   4096  TILE  8      4      16  64   4  1024       1024
Llama-3.1-8B     128  8        4 1    16   16384 TILE  8      4      16  64   4  4096       4096
Llama-3.1-8B     128  8        4 1    16   32768 TILE  8      4      16  64   4  8192       8192
biased-V         128  1        4 0    1    16384 VEC   1      1      4   128  16 1024       64
biased-V         128  1        4 0    4    16384 TILE  4      1      4   64   16 1024       1024
biased-V         128  1        4 0    16   16384 TILE  16     1      4   64   16 1024       1024
biased-V         128  1        4 0    512  16384 TILE  4      1      512 64   2  8192       8192
biased-V         128  1        4 0    1    32768 VEC   1      1      4   128  16 2048       128
biased-V         128  1        4 0    4    32768 TILE  4      1      4   64   16 2048       2048
biased-V         128  1        4 0    16   32768 TILE  16     1      4   64   16 2048       2048
biased-V         128  1        4 0    512  32768 TILE  4      1      512 64   2  16384      16384
biased-V         256  1        4 0    1    16384 VEC   1      1      4   256  16 1024       64
biased-V         256  1        4 0    4    16384 TILE  4      1      4   64   16 1024       1024
biased-V         256  1        4 0    16   16384 TILE  16     1      4   64   16 1024       1024
biased-V         256  1        4 0    512  16384 TILE  4      1      512 64   2  8192       8192
biased-V         256  1        4 0    1    32768 VEC   1      1      4   256  16 2048       128
biased-V         256  1        4 0    4    32768 TILE  4      1      4   64   16 2048       2048
biased-V         256  1        4 0    16   32768 TILE  16     1      4   64   16 2048       2048
biased-V         256  1        4 0    512  32768 TILE  4      1      512 64   2  16384      16384

CPU launch geometry for biased-V in R/A/B (calculated from measured device properties)
D    nb   kv    route p    keys/block keys/accumulator
128  1    16384 VEC   128  128        8
128  4    16384 TILE  256  64         64
128  16   16384 TILE  256  64         64
128  512  16384 TILE  256  64         64
128  1    32768 VEC   256  128        8
128  4    32768 TILE  512  64         64
128  16   32768 TILE  512  64         64
128  512  32768 TILE  512  64         64
256  1    16384 VEC   64   256        16
256  4    16384 TILE  256  64         64
256  16   16384 TILE  256  64         64
256  512  16384 TILE  256  64         64
256  1    32768 VEC   128  256        16
256  4    32768 TILE  512  64         64
256  16   32768 TILE  512  64         64
256  512  32768 TILE  512  64         64
The CPU harness overrides nsm to 8; its m=8192/8=1024, W=8192.
For TILE here p=kv/64, leaving 64 keys per output accumulator; VEC
has 8 (D=128) or 16 (D=256) keys per individual warp/lane accumulator.
These are source-derived counts, not instrumentation of the launch.
The A770 nb=512 biased-V TILE launch would instead have 8192/16384
keys per accumulator at 16K/32K under m=2. CPU passes therefore do not
validate A770's long sums, even though the source is the same.

Wall times (seconds; build/JIT/test overhead, not timing benchmarks)
------------------------------------------------------------------
phase                  wall_seconds exit
A-P                    5.12         0
A-biased               574.69       0
A-build                794.19       0
A-configure            16.06        0
A-preflight            0.45         0
B-P                    712.47       0
B-biased               341.41       0
B-build                681.80       0
B-configure            2.80         0
B-preflight            0.41         0
C-build                225.83       1
C-configure            2.64         0
R-P                    4.46         0
R-biased               26.50        1
R-build                752.44       0
R-configure            2.94         0
R-preflight            0.42         0
repo-checks            62.52        0

Required validation
-------------------
0005 applies alone and on 0001-0004: PASS, both git apply --check exit 0.
The alone check used a fresh local pin clone; the stack check
used a fresh local clone /tmp/p11-apply-PPx2eR/llama at the pin.
  git clone --no-hardlinks third_party/llama.cpp /tmp/p11-apply-alone-YYvlvy/llama
  git -C /tmp/p11-apply-alone-YYvlvy/llama apply --check "$PWD/third_party/patches/0005-tests-biased-V-FA-cases-with-FP32-CPU-reference.patch"
  git clone --no-hardlinks third_party/llama.cpp /tmp/p11-apply-PPx2eR/llama
  for patch in third_party/patches/000[1-4]-*.patch; do
    git -C /tmp/p11-apply-PPx2eR/llama apply "$PWD/$patch"
  done
  git -C /tmp/p11-apply-PPx2eR/llama apply --check "$PWD/third_party/patches/0005-tests-biased-V-FA-cases-with-FP32-CPU-reference.patch"
Patch body byte-identical to P100 0003: PASS; all clone diffs pass diff --check.
Full pin, diff stats, and check commands/results: patch-validation.log.gz.
Configuration/runtime flags and identical TILE/VEC source hashes:
configuration-audit.log.gz.
A exact p6a P: 85/85 tests passed; existing cases/names/tolerances retained.
  bin/check-forbidden.sh
    ok, exit 0.
  shellcheck -x bin/*.sh
    exit 0, no diagnostics.
  cmake -S . -B /tmp/p11-repo-build -G Ninja
  cmake --build /tmp/p11-repo-build -j 8
  ctest --test-dir /tmp/p11-repo-build --output-on-failure
    100% tests passed out of 28
    19 passed, 9 skipped; exit 0.
Skipped: xe_report_dpas_simd8, xe_report_large_grf,
xe_report_dpas_simd16_silent, xe_report_spill, xe_report_fp64,
xe_report_block2d, xe_report_slm, xe_report_sycl_joint_matrix,
xe_report_compile_commands. All output: repo-checks.log.gz.

Interpretation and limits
-------------------------
All biased-V cases pass against the FP32 CPU reference in A and B.
Both VEC and TILE satisfy the unchanged tolerance with this CPU launch
geometry. No failing SYCL route is isolated here. Because passing cases
print no ERR, these runs do not quantify a VEC improvement from F16 OFF.
R fails all 16 against the original FP16 reference despite A passing
all 16 with the same SYCL kernel configuration. This reproduces the
CPU-reference artifact; R failures are not SYCL accuracy measurements
against FP32. R and A inputs are separately randomized, so their ERRs
are not a paired numerical difference.
The prescribed TILE FP32 diagnostic does not build. Its numerical
accuracy is unmeasured; no code was changed to make it build.
Not verified: A770 numerics or runtime behavior; real-model V statistics;
any kernel fix; GPU/kernel performance or timing; full upstream suite;
oneDNN; the nine skipped static tests; driver-reported A770 maximum
work-group size. No fix is proposed or selected by this ticket.

Artifacts
---------
README.txt: commands, per-clone biased tables, P summaries, source formulas,
conditional A770/CPU split tables, wall times, validation, and limits.
<variant>-configure/build/preflight/biased/P.log.gz: unabridged output for
performed phases (C has only configure/build when it fails).
<variant>-source-diff.log.gz: complete scratch diagnostic source differences.
device-info.log.gz, cache-seed.log.gz, patch-validation.log.gz,
configuration-audit.log.gz,
repo-checks.log.gz, commands.log.gz, wall-times.log.gz: supporting evidence.

Manager verification (Claude Opus 5.5, 2026-10-11)
--------------------------------------------------
Rechecked: printed summaries and [FA-DISP] routes in R/A/B biased and P
logs; C's first error; 0005's diff body against P100 0003 (identical apart
from the trailing git signature); max_wg_per_cu = max_work_group_size /
max_compute_units at ggml-sycl.cpp:248-249 feeding launch_fattn at
fattn-common.hpp:1061-1104.
Limit that decides the interpretation: on this CPU device max_wg_per_cu is
8192/8 = 1024, so launch_fattn splits every biased-V TILE case into 64-key
partial sums. A's 16/16 PASS therefore shows FP16 accumulation is harmless
for 64-key sums, not for the 1K-16K-key sums the A770 geometry implies.
The A/B/C question stays open until the run is repeated with A770 launch
geometry (nsm=32, max_wg_per_cu=2): follow-up ticket p11b.
