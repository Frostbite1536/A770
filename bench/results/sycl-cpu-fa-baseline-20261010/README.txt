p6a-sycl-cpu-harness: CPU SYCL correctness baseline and negative control complete
Ticket date: 2026-10-10. These are CPU results, with optimization enabled.
No A770 execution or GPU performance/behavior is asserted. No tolerances or
existing tests were changed. The temporary TILE kernel edit was reverted.

Device, pin, and environment
----------------------------
A770 base: 095ddfc; worktree branch: p6a-sycl-cpu-harness.
llama.cpp: 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6 (bench/LLAMA_COMMIT).
Intel(R) OpenCL CPU: Intel(R) Xeon(R) CPU E3-1505M v6 @ 3.00GHz,
8 compute units; OpenCL 3.0; driver 2026.21.7.0.24_160000; fp16;
subgroup sizes 4 8 16 32 64; x86_64; max workgroup 8192.
Every shell command starts with:
  export PATH=/home/p51/Github/A770/.venv/bin:$PATH;
  O=/home/p51/Github/A770/tools/oneapi-2026.1
  export OCL_ICD_VENDORS=$O/etc/OpenCL/vendors LD_LIBRARY_PATH=$O/lib
  export ONEAPI_DEVICE_SELECTOR=opencl:cpu GGML_SYCL_TEST_CPU=1
The script sets these variables itself. Both CPU-only selection and the helper's
validated Intel/OpenCL/CPU marker must be present for the script preflight to
succeed; upstream can otherwise report success when all backends were skipped.
No installs, network, commits, pushes, credentials, or model weights were used.

Build and script
----------------
  GGML_SYCL_TEST_CPU=1 cmake -S third_party/llama.cpp \
    -B third_party/llama.cpp/build-sycl-cpu -G Ninja \
    -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=OFF \
    -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=OFF -DGGML_SYCL_DEVICE_ARCH= \
    -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=$O/bin/icx -DCMAKE_CXX_COMPILER=$O/bin/icpx \
    -DCMAKE_PREFIX_PATH=$O
  GGML_SYCL_TEST_CPU=1 cmake --build third_party/llama.cpp/build-sycl-cpu \
    --target test-backend-ops -j 8
Configure: Build files have been written to .../build-sycl-cpu; TBB found.
Approved ALiBi-loop rebuild: [112/112] Linking CXX executable bin/test-backend-ops;
wall_seconds=631.61 exit=0. The earlier first build and dependency stops are archived
in the prior-round log/readme section of fa-test.txt.gz.
Target test-backend-ops only (including its library dependencies).
JIT spir64, no device arch; Release -O3 -DNDEBUG. oneDNN and Level Zero OFF.
The script incrementally configures/builds, applies the single patch if needed,
checks reverse applicability, validates SYCL0, defaults -b SYCL0, and propagates
build/test errors. LLAMA overrides the scratch clone path.
No -cl-opt-disable is retained or used in any round-3 numerical run.

The single test-only patch is not for upstream:
- Runtime gate GGML_SYCL_TEST_CPU=1 accepts only the default Intel OpenCL CPU
  as SYCL0 and emits the validated device marker.
- Configure-time gate omits the GPU-only >4GB option rejected by the CPU JIT;
  uses -fsycl-device-code-split=per_kernel to limit CPU-JIT program size.
- Runtime gate uses the CPU compute-unit count for nsm; the upstream Xe-core
  division produced zero on this CPU and SIGFPE in host TILE scheduling.
- Manager-approved ungated exception: scalar ALiBi slope multiplication avoids
  the missing 32-wide pown builtin. P-6 does not pad under ALiBi; the stated
  rounding difference is within test tolerance. Two selected ALiBi cases pass,
  including GQA 12 (non-power-of-two head count).
Dispatch routing, kernels otherwise, and tests/tolerances are upstream.

Final documented numerical subset
---------------------------------
The full suite was not completed. Before per-kernel image splitting, just the
optimized two-case gate took 3015.89 s; a support scan has 5332 cases including
large contexts/query counts. A subset was selected under the ticket's ~3-hour
limit and supplemented for all requested coverage and non-power-of-two GQA.
85 cases (84 unique case strings; upstream repeats the selected GQA=20 case).
Required dimensions 64/128/256, GQA 1/2/4/8 and suite non-powers 6/12/20,
F16/q8_0/q4_0 K/V, query counts 1/2/3/4/16/32 within the requested 1..32 range.
Extra dimensions 576/512 and GQA 16 exercise a supported MLA view and q8 TILE.
F16/q8_0/q4_0 are each tested at all three required head dimensions; the
prefill supplement uses MKL. Quantized TILE is also exercised (D64 and q8 D256).
All selected cases have masking enabled. This is representative coverage,
not an exhaustive Cartesian product or a GPU benchmark.

Set P to the following EXACT one-line -p filter:
P='((^hsk=(64|128|256),hsv=(64|128|256),nh=[124],nr23=\[(1|2|4|6|8|12),1\],kv=(113|128|256|512|1024|4096),nb=(1|2|3|4|16|32),mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=(f16|q8_0|q4_0),type_V=(f16|q8_0|q4_0),permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$)|(^hsk=(64|128|256),hsv=(64|128|256),nh=(1|8),nr23=\[8,1\],kv=4096,nb=(1|4),mask=1,sinks=0,max_bias=(0|8)\.000000,logit_softcap=0\.000000,prec=f32,type_K=(f16|q8_0),type_V=(f16|q8_0),permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=(0|512)$))|(^hsk=64,hsv=64,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=64,hsv=64,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=64,hsv=64,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q4_0,type_V=q4_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=128,hsv=128,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=128,hsv=128,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=128,hsv=128,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q4_0,type_V=q4_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=256,hsv=256,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=256,hsv=256,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=256,hsv=256,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q4_0,type_V=q4_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=128,hsv=128,nh=4,nr23=\[12,1\],kv=512,nb=3,mask=1,sinks=0,max_bias=8\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=576,hsv=512,nh=1,nr23=\[20,1\],kv=512,nb=1,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=1,n_kv_max=0$|^hsk=256,hsv=256,nh=2,nr23=\[16,1\],kv=113,nb=1,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$)'

Common numerical command (all runs explicitly clear prior no-opt diagnostics):
  /usr/bin/time -f 'wall_seconds=%e exit=%x' -o <phase>.time \
    env -u SYCL_PROGRAM_COMPILE_OPTIONS -u SYCL_PROGRAM_BUILD_OPTIONS \
    GGML_SYCL_MKL_FA_DEBUG=1 SYCL_CACHE_PERSISTENT=1 \
    SYCL_CACHE_DIR=/tmp/p6a-sycl-cpu-cache \
    bin/sycl-cpu-tbo.sh -o FLASH_ATTN_EXT -p "$P" > <phase>.log 2>&1
Persistent caching changes JIT reuse, not math or optimization. Times below
include configure, incremental build/relink, empty preflight and tests.
The CPU backend is the comparison reference; its separate backend entry is
Skipped by -b SYCL0, not an extra successful backend execution. Counts exclude
preflight's 0/0 tests. [FA-DISP] lines are upstream dispatch diagnostics.

Final recovery command above, after reverting the negative control:
  85 passed / 0 failed / 0 unsupported
  85/85 tests passed
  2/2 backends passed
  wall_seconds=291.85 exit=0
  Routes: {'MKL': 10, 'TILE': 62, 'VEC': 13}; oneDNN 0 (compiled out).
Failing/unsupported groups in this subset: none.
Coverage counts (each category independently sums to 85):
  (K head dim, V head dim): ('64', '64')=42, ('128', '128')=22, ('256', '256')=19, ('576', '512')=2
  GQA ratios: 4=24, 1=47, 12=4, 20=2, 16=1, 8=5, 6=1, 2=1
  query tokens: 32=31, 1=27, 3=22, 2=2, 4=2, 16=1
  (K type, V type): ('f16', 'f16')=51, ('q8_0', 'q8_0')=17, ('q4_0', 'q4_0')=15, ('q8_0', 'q4_0')=1, ('q4_0', 'f16')=1

Negative control and exact recovery
----------------------------------
Only temporary kernel change, made with our own patch in the scratch clone:
  ggml/src/ggml-sycl/fattn-tile.hpp: tmp_f[i1] *= scale;
  changed to:                        tmp_f[i1] *= 1.0f;
Rebuild via the script (target test-backend-ops, -j 8); command as above with:
P_NEG='^hsk=128,hsv=128,nh=4,nr23=\[4,1\],kv=512,nb=3,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$'
  bin/sycl-cpu-tbo.sh -o FLASH_ATTN_EXT -p "$P_NEG"
Dispatch: [FA-DISP] #1 TILE D=128 n_kv=512 delta=0 V_ne1=512
Exact failing case is P_NEG without regex escapes/anchors.
  [FLASH_ATTN_EXT] ERR = 0.963351393 > 0.000500000
  0/1 tests passed; 0 passed / 1 failed / 0 unsupported
  1/2 backends passed
  Command exited with non-zero status 1
wall_seconds=341.74 exit=1
Negative patch was reversed exactly, followed by:
  git -C third_party/llama.cpp diff --stat
 ggml/src/ggml-sycl/CMakeLists.txt  |  8 +++++++-
 ggml/src/ggml-sycl/common.hpp      |  6 +++++-
 ggml/src/ggml-sycl/dpct/helper.hpp | 14 ++++++++++++++
 ggml/src/ggml-sycl/ggml-sycl.cpp   |  5 +++++
 4 files changed, 31 insertions(+), 2 deletions(-)

  git -C third_party/llama.cpp apply --check --reverse <saved test-only patch>
    exit 0.
Saved patch body compared byte-for-byte to the entire scratch-clone git diff:
identical. No fattn-tile.hpp diff remains. Rebuilt and reran the complete
85-case subset above: the control case and all other cases returned to OK.

Reported unsupported groups (support scan, not correctness)
-----------------------------------------------------------
  /usr/bin/time -f 'wall_seconds=%e exit=%x' -o <support>.time \
    env OCL_ICD_VENDORS="$O/etc/OpenCL/vendors" LD_LIBRARY_PATH="$O/lib" \
    ONEAPI_DEVICE_SELECTOR=opencl:cpu GGML_SYCL_TEST_CPU=1 \
    third_party/llama.cpp/build-sycl-cpu/bin/test-backend-ops \
    support -o FLASH_ATTN_EXT -b SYCL0
  5332 cases: 4038 SUPPORTED / 1294 NOT SUPPORTED; 2/2 backends passed.
  wall_seconds=0.53 exit=0
Every exact unsupported case is in the full log. Shapes can be supported
through MKL even when TILE's shape switch rejects them. Alternative independent
groupings of the same 1294 NOT SUPPORTED cases follow:
  (K head dim, V head dim):
    ('128', '128'): 2
    ('128', '64'): 4
    ('128', '96'): 13
    ('192', '128'): 178
    ('192', '192'): 176
    ('320', '256'): 257
    ('576', '512'): 209
    ('64', '128'): 2
    ('64', '64'): 224
    ('64', '72'): 1
    ('65', '67'): 3
    ('72', '64'): 1
    ('72', '72'): 112
    ('96', '64'): 112
  (K type, V type):
    ('bf16', 'bf16'): 1
    ('f16', 'f16'): 937
    ('f16', 'q4_0'): 1
    ('f32', 'f32'): 1
    ('f32', 'q8_0'): 1
    ('iq4_nl', 'iq4_nl'): 337
    ('q1_0', 'f16'): 1
    ('q1_0', 'q1_0'): 1
    ('q1_0', 'q4_0'): 1
    ('q2_0', 'f16'): 1
    ('q2_0', 'q2_0'): 1
    ('q2_0', 'q4_0'): 1
    ('q4_0', 'q1_0'): 1
    ('q4_0', 'q2_0'): 1
    ('q4_0', 'q4_0'): 1
    ('q4_1', 'q4_1'): 1
    ('q5_0', 'q5_0'): 1
    ('q5_1', 'q5_1'): 1
    ('q8_0', 'f16'): 1
    ('q8_0', 'f32'): 1
    ('q8_0', 'q8_0'): 2

Other numerical runs and JIT investigation (round 3)
----------------------------------------------------
Each used the common numerical command and filter specified below. The first
optimized gate predates persistent caching. Raw outputs and times are archived.
- Two-case VEC+TILE gate, auto images: 2 passed / 0 failed / 0 unsupported;
  2/2 tests passed; wall_seconds=3015.89 exit=0; VEC 1, TILE 1.
  P_GATE='^hsk=128,hsv=128,nh=4,(nr23=\[1,1\],kv=512,nb=1|nr23=\[4,1\],kv=512,nb=3),mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$'
- Original core subset, auto images: 1 passed / 0 numerical failed / 0 unsupported
  emitted, then a CPU compiler crash on q8 TILE, SIGSEGV (actual exit 139),
  313.66 s. GNU time's exit=0 for a signalled command is not a successful exit.
  Exact interrupted case:
  hsk=64,hsv=64,nh=4,nr23=[1,1],kv=113,nb=1,mask=1,sinks=0,max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=[0,1,2,3],kv_view=1,v_is_view_of_k=0,n_kv_max=0
  P_CORE='^hsk=(64|128|256),hsv=(64|128|256),nh=[124],nr23=\[(1|2|4|6|8|12),1\],kv=(113|128|256|512|1024|4096),nb=(1|2|3|4|16|32),mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=(f16|q8_0|q4_0),type_V=(f16|q8_0|q4_0),permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$'
- Revised 52-case auto-image filter (kv=113 excluded): 1 passed / 0 numerical
  failed / 0 unsupported emitted; our process was interrupted during q8 VEC
  CPU JIT (exit 130); wall time was not recorded. No suite summary.
  Filter is P_CORE with kv=(128|256|512|1024|4096).
- Env-gated per-kernel images added; optimized gate rerun: 2 passed / 0 failed /
  0 unsupported, 2/2 tests passed; wall_seconds=272.17 exit=0; VEC 1, TILE 1.
- Main 72-case baseline: 72 passed / 0 failed / 0 unsupported;
  72/72 tests passed; wall_seconds=1690.85 exit=0; TILE 58, VEC 13, MKL 1.
  P_72='(^hsk=(64|128|256),hsv=(64|128|256),nh=[124],nr23=\[(1|2|4|6|8|12),1\],kv=(113|128|256|512|1024|4096),nb=(1|2|3|4|16|32),mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=(f16|q8_0|q4_0),type_V=(f16|q8_0|q4_0),permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$)|(^hsk=(64|128|256),hsv=(64|128|256),nh=(1|8),nr23=\[8,1\],kv=4096,nb=(1|4),mask=1,sinks=0,max_bias=(0|8)\.000000,logit_softcap=0\.000000,prec=f32,type_K=(f16|q8_0),type_V=(f16|q8_0),permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=(0|512)$)'
- Supplement: 13 passed / 0 failed / 0 unsupported; 13/13 tests passed;
  wall_seconds=422.69 exit=0; TILE 4, MKL 9.
  P_EXTRA='^hsk=64,hsv=64,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=64,hsv=64,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=64,hsv=64,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q4_0,type_V=q4_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=128,hsv=128,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=128,hsv=128,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=128,hsv=128,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q4_0,type_V=q4_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=256,hsv=256,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=256,hsv=256,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=256,hsv=256,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q4_0,type_V=q4_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=128,hsv=128,nh=4,nr23=\[12,1\],kv=512,nb=3,mask=1,sinks=0,max_bias=8\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=576,hsv=512,nh=1,nr23=\[20,1\],kv=512,nb=1,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=1,n_kv_max=0$|^hsk=256,hsv=256,nh=2,nr23=\[16,1\],kv=113,nb=1,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$'
The original crashing q8 TILE case is OK in the final per-kernel runs.
The archived prior-round README/logs record the initial GPU-only option,
nsm=0 and pown stops and the failed -cl-opt-disable diagnostics. Per manager
round 3, no further no-opt workarounds were pursued. Historical failures are
not omitted or counted as passes in the final optimized baseline.

Required project checks and unset-gate behavior
-----------------------------------------------
  shellcheck -x bin/*.sh
    exit 0, no diagnostics.
  bin/check-forbidden.sh
    ok, exit 0.
  cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build
    exit 0; 100% tests passed out of 25
    17 passed, 8 skipped; Total Test time (real) = 63.86 sec
    combined configure/build/ctest: wall_seconds=65.60 exit=0
    Skipped: xe_report_dpas_simd8, xe_report_large_grf,
    xe_report_dpas_simd16_silent, xe_report_spill, xe_report_fp64,
    xe_report_block2d, xe_report_slm, xe_report_sycl_joint_matrix.
  git diff --check; git -C third_party/llama.cpp diff --check
    exit 0 (both).
  git -C third_party/llama.cpp apply --check --reverse <saved test-only patch>
    exit 0; saved patch exactly equals scratch diff, as described above.
Final validation output is included in fa-test.txt.gz.

Unset-gate verification: diff inspection shows the entire original helper body
retained after the new guarded early-return branch. With the variable absent
(or other than 1), device acceptance follows that original GPU-only path,
original nsm assignment remains, and CMake retains the original GPU link option
and upstream image splitting. The only approved compile-time exception is the
scalar ALiBi slope loop, documented in the source and patch header. Prior round
also observed upstream's CPU-selected unset-gate exception, archived in the log.
No GPU test with the variable unset was run.

Not verified
------------
Numerical correctness of the entire 5332-case suite or all supported cases;
all query counts between 1 and 32 or all coverage combinations; oneDNN;
the eight skipped xe-report tests; GPU behavior with the gate unset;
A770 execution, GPU performance, XMX, memory behavior or timing.
CPU wall times include CPU JIT/build overhead and are not GPU benchmark numbers.
No shipping decision is made here.

Artifacts
---------
README.txt: final commands, exact filters, counts, times, coverage and limitations.
fa-test.txt.gz: unabridged labeled stdout/stderr and timing files, including
previous-round logs/readme, round-3 build, gates, interrupted runs, support scans,
negative control, recovery, required checks, saved filters and final patch/stat.
