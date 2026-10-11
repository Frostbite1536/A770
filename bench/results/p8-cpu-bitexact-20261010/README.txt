P8 SYCL K/V conversion: CPU device, mechanism check
Ticket date: 2026-10-10. Worktree: p8-sycl-kv-conversion.
No A770 execution, performance, or end-to-end decode is asserted.

Device and reproducible commands
-------------------------------
A770 base: ba3613c. Compiler: icpx 2026.1.1 (2026.1.1.20260724).
All shell commands start with:
  export PATH=/home/p51/Github/A770/.venv/bin:$PATH;
  O=/home/p51/Github/A770/tools/oneapi-2026.1
  export OCL_ICD_VENDORS=$O/etc/OpenCL/vendors LD_LIBRARY_PATH=$O/lib
  export ONEAPI_DEVICE_SELECTOR=opencl:cpu GGML_SYCL_TEST_CPU=1
  export SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=/tmp/p8-sycl-kv-conversion-sycl-cache
  export GGML_SYCL_MKL_FA_DEBUG=1 A770_TOOLS=/home/p51/Github/A770/tools
SYCL_PROGRAM_COMPILE_OPTIONS and SYCL_PROGRAM_BUILD_OPTIONS were unset.
Toolchain is read-only in the main checkout. No installs or network.
llama.cpp is 79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6.
Baseline scratch source contains only the p6a test-only CPU harness patch.

Microbenchmark (F16 enabled by default):
  $O/bin/icpx -O2 -std=c++17 -fsycl -fsycl-device-code-split=per_kernel \
    bench/micro/dequant_nc_sycl.cpp -o /tmp/p8-sycl-kv-conversion/dequant_nc_sycl
  /tmp/p8-sycl-kv-conversion/dequant_nc_sycl --reps 3
  /tmp/p8-sycl-kv-conversion/dequant_nc_sycl --edges
F16 disabled: same build with -DP8_FLOAT_ARITH, then same run arguments.
All reported times below are CPU device, mechanism check; no GPU inference.
Timings exclude warm-up and JIT; medians of three event durations. CPU timings
were collected during other compilation and are not performance conclusions.
Static reports (no GPU execution):
  python bin/xe-report.py bench/micro/dequant_nc_sycl.cpp --wide
  python bin/xe-report.py bench/micro/dequant_nc_sycl.cpp -D P8_FLOAT_ARITH --wide
Both use A770_TOOLS=/home/p51/Github/A770/tools (acm-g10).
In these standalone kernel names, ggml_type 0=q4_0 and 1=q8_0;
variant 0=upstream, 1=rows/group, 2=vector8; UnaryNC has variants 0/1.
GRF=128 is IGC's GRF mode, not a claim of 128 live registers.

Source verification:
  q4_0, q8_0, dequantize_block_nc, convert_unary_nc: byte-identical upstream bodies.
  vector8: byte-identical benchmark and prepared convert.cpp bodies.
Integration preflight: both prepared geometry and vector convert.cpp versions
compiled with llama.cpp's actual Ninja compile command into /tmp objects,
exit 0 each. Only upstream deprecation warnings; baseline objects unchanged.

Repository checks:
  bin/check-forbidden.sh: ok, exit 0
  shellcheck -x bin/*.sh: exit 0
  cmake -S . -B /tmp/p8-sycl-kv-conversion/repo-build -G Ninja
  cmake --build /tmp/p8-sycl-kv-conversion/repo-build
  A770_TOOLS=/home/p51/Github/A770/tools ctest \
    --test-dir /tmp/p8-sycl-kv-conversion/repo-build --output-on-failure
  100% tests passed out of 28
  Total Test time (real) = 198.73 sec
No xe-report test skips.

Final microbenchmark (F16 arithmetic enabled; --reps 3)
Device: Intel(R) Xeon(R) CPU E3-1505M v6 @ 3.00GHz; CPU device, mechanism check; reps=3; arithmetic=GGML_SYCL_F16=ON
type dst D n_kv n_head_kv | upstream_us rows_us vec8_us D2D_us (median) | rows vec8 D2D bit-exact
q8_0 f16 64 4096 4 | 40306.795 8911.898 1750.435 269.081 | PASS PASS PASS
q4_0 f16 64 4096 4 | 22063.477 11726.716 1802.547 429.233 | PASS PASS PASS
q8_0 f32 64 4096 4 | 19630.948 5695.408 1914.212 936.449 | PASS PASS PASS
q4_0 f32 64 4096 4 | 31492.678 8540.766 1894.126 937.468 | PASS PASS PASS
q8_0 f16 64 4096 8 | 35895.901 27545.263 8943.456 947.103 | PASS PASS PASS
q4_0 f16 64 4096 8 | 41323.784 16547.507 5055.194 1047.080 | PASS PASS PASS
q8_0 f32 64 4096 8 | 38821.379 12996.067 5159.698 1927.089 | PASS PASS PASS
q4_0 f32 64 4096 8 | 46353.702 9706.527 5846.270 1793.048 | PASS PASS PASS
q8_0 f16 64 16384 4 | 56669.721 26885.356 22006.452 1990.942 | PASS PASS PASS
q4_0 f16 64 16384 4 | 50280.678 30886.577 18896.810 9871.222 | PASS PASS PASS
q8_0 f32 64 16384 4 | 48883.311 15358.859 12162.240 5097.373 | PASS PASS PASS
q4_0 f32 64 16384 4 | 54202.814 22582.486 2377.910 2592.410 | PASS PASS PASS
q8_0 f16 64 16384 8 | 78961.490 35996.567 27360.023 9194.334 | PASS PASS PASS
q4_0 f16 64 16384 8 | 76619.072 28433.315 26376.885 9092.539 | PASS PASS PASS
q8_0 f32 64 16384 8 | 97848.200 34068.701 22672.768 13874.964 | PASS PASS PASS
q4_0 f32 64 16384 8 | 63200.174 38114.147 24836.801 21181.280 | PASS PASS PASS
q8_0 f16 128 4096 4 | 23900.353 10190.303 14293.715 916.250 | PASS PASS PASS
q4_0 f16 128 4096 4 | 29189.212 19900.367 16985.894 943.679 | PASS PASS PASS
q8_0 f32 128 4096 4 | 23718.076 12040.708 10081.807 5853.290 | PASS PASS PASS
q4_0 f32 128 4096 4 | 36003.442 9809.466 1733.861 1718.456 | PASS PASS PASS
q8_0 f16 128 4096 8 | 40793.390 20604.444 12871.121 4343.708 | PASS PASS PASS
q4_0 f16 128 4096 8 | 44834.711 22993.453 18575.835 7833.010 | PASS PASS PASS
q8_0 f32 128 4096 8 | 34202.198 15996.300 11902.370 7047.982 | PASS PASS PASS
q4_0 f32 128 4096 8 | 35879.699 22809.345 9893.051 6334.983 | PASS PASS PASS
q8_0 f16 128 16384 4 | 51877.350 20034.608 12689.521 5964.828 | PASS PASS PASS
q4_0 f16 128 16384 4 | 60588.242 25801.729 20216.853 7357.194 | PASS PASS PASS
q8_0 f32 128 16384 4 | 50168.731 43969.692 20062.386 22940.247 | PASS PASS PASS
q4_0 f32 128 16384 4 | 53995.191 22172.485 27652.334 16431.625 | PASS PASS PASS
q8_0 f16 128 16384 8 | 95493.572 46804.263 30876.452 12091.938 | PASS PASS PASS
q4_0 f16 128 16384 8 | 103890.924 44006.967 30038.734 12890.052 | PASS PASS PASS
q8_0 f32 128 16384 8 | 101390.809 55039.285 39628.458 28906.310 | PASS PASS PASS
q4_0 f32 128 16384 8 | 71687.186 45336.289 24001.780 25063.292 | PASS PASS PASS
q8_0 f16 256 4096 4 | 32098.334 17995.351 16638.450 2488.363 | PASS PASS PASS
q4_0 f16 256 4096 4 | 27891.238 23895.534 7893.911 2540.325 | PASS PASS PASS
q8_0 f32 256 4096 4 | 75609.999 31885.023 23840.893 7907.286 | PASS PASS PASS
q4_0 f32 256 4096 4 | 30229.906 16016.085 12984.549 9105.634 | PASS PASS PASS
q8_0 f16 256 4096 8 | 39857.215 39520.031 24321.726 17741.356 | PASS PASS PASS
q4_0 f16 256 4096 8 | 39891.155 29321.612 20442.586 7383.224 | PASS PASS PASS
q8_0 f32 256 4096 8 | 46917.608 26893.485 17992.303 15641.078 | PASS PASS PASS
q4_0 f32 256 4096 8 | 38891.462 33895.945 24769.330 18994.673 | PASS PASS PASS
q8_0 f16 256 16384 4 | 70807.240 46428.870 38903.265 17998.725 | PASS PASS PASS
q4_0 f16 256 16384 4 | 61498.028 42976.331 26915.459 20630.953 | PASS PASS PASS
q8_0 f32 256 16384 4 | 59149.230 52367.825 32955.599 37941.423 | PASS PASS PASS
q4_0 f32 256 16384 4 | 53466.415 41888.756 30877.691 21502.620 | PASS PASS PASS
q8_0 f16 256 16384 8 | 108104.997 88159.463 48302.585 24271.484 | PASS PASS PASS
q4_0 f16 256 16384 8 | 82143.730 79085.071 40673.512 27559.215 | PASS PASS PASS
q8_0 f32 256 16384 8 | 109598.219 74270.000 54046.357 34886.049 | PASS PASS PASS
q4_0 f32 256 16384 8 | 93000.157 71881.034 42998.419 36311.548 | PASS PASS PASS
48/48 type/destination/shape combinations bit-exact; 144/144 comparisons PASS

Final microbenchmark (F16 arithmetic disabled; --reps 3)
Device: Intel(R) Xeon(R) CPU E3-1505M v6 @ 3.00GHz; CPU device, mechanism check; reps=3; arithmetic=GGML_SYCL_F16=OFF
type dst D n_kv n_head_kv | upstream_us rows_us vec8_us D2D_us (median) | rows vec8 D2D bit-exact
q8_0 f16 64 4096 4 | 22993.962 9931.144 4179.222 476.418 | PASS PASS PASS
q4_0 f16 64 4096 4 | 23164.916 2245.575 1759.440 457.357 | PASS PASS PASS
q8_0 f32 64 4096 4 | 23891.070 2912.990 1663.838 871.593 | PASS PASS PASS
q4_0 f32 64 4096 4 | 17647.088 5942.422 3640.226 1146.926 | PASS PASS PASS
q8_0 f16 64 4096 8 | 36169.171 16030.946 3479.811 895.076 | PASS PASS PASS
q4_0 f16 64 4096 8 | 30065.813 4408.804 2553.810 902.653 | PASS PASS PASS
q8_0 f32 64 4096 8 | 30870.529 10990.872 10047.257 8029.566 | PASS PASS PASS
q4_0 f32 64 4096 8 | 30067.269 19876.635 8447.113 7749.440 | PASS PASS PASS
q8_0 f16 64 16384 4 | 47999.274 24051.487 17687.455 1904.928 | PASS PASS PASS
q4_0 f16 64 16384 4 | 47288.657 24644.113 25939.421 1906.641 | PASS PASS PASS
q8_0 f32 64 16384 4 | 40749.637 23357.311 14570.168 5436.948 | PASS PASS PASS
q4_0 f32 64 16384 4 | 47893.632 16809.134 11201.127 7151.290 | PASS PASS PASS
q8_0 f16 64 16384 8 | 70918.836 37636.546 27875.247 12739.382 | PASS PASS PASS
q4_0 f16 64 16384 8 | 74615.228 29005.027 20911.303 13030.904 | PASS PASS PASS
q8_0 f32 64 16384 8 | 82522.058 29997.872 31976.155 16241.150 | PASS PASS PASS
q4_0 f32 64 16384 8 | 61739.641 26659.424 10526.981 15993.142 | PASS PASS PASS
q8_0 f16 128 4096 4 | 20351.860 20974.777 11762.357 1088.916 | PASS PASS PASS
q4_0 f16 128 4096 4 | 20961.530 9762.760 4525.159 911.889 | PASS PASS PASS
q8_0 f32 128 4096 4 | 18198.482 11020.728 7368.882 1733.163 | PASS PASS PASS
q4_0 f32 128 4096 4 | 19490.491 11578.430 12050.204 1847.517 | PASS PASS PASS
q8_0 f16 128 4096 8 | 37130.721 23904.208 17187.281 2850.591 | PASS PASS PASS
q4_0 f16 128 4096 8 | 48594.618 12808.688 8555.428 2273.469 | PASS PASS PASS
q8_0 f32 128 4096 8 | 29904.290 30006.502 17978.802 21526.953 | PASS PASS PASS
q4_0 f32 128 4096 8 | 35934.717 29001.851 11131.970 15944.938 | PASS PASS PASS
q8_0 f16 128 16384 4 | 48005.355 34906.350 27998.077 15076.346 | PASS PASS PASS
q4_0 f16 128 16384 4 | 54015.088 23659.021 23983.653 7124.263 | PASS PASS PASS
q8_0 f32 128 16384 4 | 55254.613 26003.546 20119.938 18438.637 | PASS PASS PASS
q4_0 f32 128 16384 4 | 48906.114 28081.378 23025.462 16814.023 | PASS PASS PASS
q8_0 f16 128 16384 8 | 91999.317 50147.566 34863.860 12823.749 | PASS PASS PASS
q4_0 f16 128 16384 8 | 77780.696 36990.615 25988.577 18783.433 | PASS PASS PASS
q8_0 f32 128 16384 8 | 91618.919 52052.100 37001.750 29900.471 | PASS PASS PASS
q4_0 f32 128 16384 8 | 62994.978 37229.507 32073.544 35691.796 | PASS PASS PASS
q8_0 f16 256 4096 4 | 21611.273 16512.213 7567.825 2075.129 | PASS PASS PASS
q4_0 f16 256 4096 4 | 15346.324 19225.979 15128.445 3936.734 | PASS PASS PASS
q8_0 f32 256 4096 4 | 24693.727 23684.688 15837.284 14479.614 | PASS PASS PASS
q4_0 f32 256 4096 4 | 32070.170 15890.037 12228.236 11870.315 | PASS PASS PASS
q8_0 f16 256 4096 8 | 39995.944 33297.834 25140.116 12030.823 | PASS PASS PASS
q4_0 f16 256 4096 8 | 35959.771 27281.782 11453.451 10636.906 | PASS PASS PASS
q8_0 f32 256 4096 8 | 32082.596 27085.817 12109.914 12935.947 | PASS PASS PASS
q4_0 f32 256 4096 8 | 36595.273 29885.126 34900.107 19886.650 | PASS PASS PASS
q8_0 f16 256 16384 4 | 65894.340 40902.264 36995.358 11464.108 | PASS PASS PASS
q4_0 f16 256 16384 4 | 46006.672 47212.670 35569.604 16016.481 | PASS PASS PASS
q8_0 f32 256 16384 4 | 56936.626 51175.900 35711.163 29255.127 | PASS PASS PASS
q4_0 f32 256 16384 4 | 61187.571 38570.347 40475.015 28330.754 | PASS PASS PASS
q8_0 f16 256 16384 8 | 93904.409 67852.599 42063.052 30865.836 | PASS PASS PASS
q4_0 f16 256 16384 8 | 109207.516 72946.704 45063.582 23002.469 | PASS PASS PASS
q8_0 f32 256 16384 8 | 104875.249 68692.895 39496.371 50823.036 | PASS PASS PASS
q4_0 f32 256 16384 8 | 93981.502 55882.933 43383.516 34341.334 | PASS PASS PASS
48/48 type/destination/shape combinations bit-exact; 144/144 comparisons PASS

Edges (F16 arithmetic enabled)
Device: Intel(R) Xeon(R) CPU E3-1505M v6 @ 3.00GHz; CPU device, mechanism check; reps=1; arithmetic=GGML_SYCL_F16=ON
type dst D n_kv n_head_kv | upstream_us rows_us vec8_us D2D_us (median) | rows vec8 D2D bit-exact
q8_0 f16 32 19 3 | 95.225 24.590 18.884 65.371 | PASS PASS PASS
q4_0 f16 32 19 3 | 96.762 21.194 6.695 56.510 | PASS PASS PASS
q8_0 f32 32 19 3 | 89.713 31.852 21.879 58.432 | PASS PASS PASS
q4_0 f32 32 19 3 | 92.618 21.519 23.130 44.947 | PASS PASS PASS
q8_0 f16 96 19 3 | 86.712 39.168 21.308 41.174 | PASS PASS PASS
q4_0 f16 96 19 3 | 76.270 33.042 40.608 60.061 | PASS PASS PASS
q8_0 f32 96 19 3 | 91.138 39.978 11.932 66.355 | PASS PASS PASS
q4_0 f32 96 19 3 | 77.612 22.352 11.439 49.848 | PASS PASS PASS
q8_0 f16 224 19 3 | 121.194 104.764 24.728 45.037 | PASS PASS PASS
q4_0 f16 224 19 3 | 78.306 60.309 23.444 57.996 | PASS PASS PASS
q8_0 f32 224 19 3 | 98.431 65.283 22.874 85.598 | PASS PASS PASS
q4_0 f32 224 19 3 | 87.306 59.719 24.779 93.240 | PASS PASS PASS
q8_0 f16 512 19 3 | 149.441 118.084 65.536 62.534 | PASS PASS PASS
q4_0 f16 512 19 3 | 142.135 103.774 71.023 79.242 | PASS PASS PASS
q8_0 f32 512 19 3 | 140.470 122.131 61.937 77.182 | PASS PASS PASS
q4_0 f32 512 19 3 | 114.505 100.965 79.343 74.714 | PASS PASS PASS
q8_0 f16 768 19 3 | 218.974 119.542 94.656 104.385 | PASS PASS PASS
q4_0 f16 768 19 3 | 201.942 204.292 98.457 72.502 | PASS PASS PASS
q8_0 f32 768 19 3 | 247.445 220.925 78.845 135.668 | PASS PASS PASS
q4_0 f32 768 19 3 | 185.292 205.040 75.662 96.607 | PASS PASS PASS
unary f32->f16 D=1 rows=19 heads=3 batches=2 padded-strides: PASS
unary f32->f16 D=17 rows=19 heads=3 batches=2 padded-strides: PASS
unary f32->f16 D=67 rows=19 heads=3 batches=2 padded-strides: PASS
unary f32->f16 D=96 rows=19 heads=3 batches=2 padded-strides: PASS
unary f32->f16 D=255 rows=19 heads=3 batches=2 padded-strides: PASS
unary f32->f16 D=256 rows=19 heads=3 batches=2 padded-strides: PASS
unary f32->f16 D=513 rows=19 heads=3 batches=2 padded-strides: PASS
20/20 quantized edge combinations; 7/7 unary edge shapes bit-exact

Static acm-g10 report (F16 arithmetic enabled)
device acm-g10  kernel                                                   simd  grf thr  spill    priv    slm dpas fp64
               UnaryNC<0>                                                 16  128   8      0       0      0    0    0 
               UnaryNC<1>                                                 16  128   8      0       0      0    0    0 
               DequantNC<(ggml_type)1, sycl::_V1::detail::half_impl::half, 0>   16  128   8      0       0      0    0    0 
               DequantNC<(ggml_type)1, sycl::_V1::detail::half_impl::half, 1>   16  128   8      0       0      0    0    0 
               DequantNC<(ggml_type)1, sycl::_V1::detail::half_impl::half, 2>   16  128   8      0       0      0    0    0 
               DequantNC<(ggml_type)0, sycl::_V1::detail::half_impl::half, 0>   16  128   8      0       0      0    0    0 
               DequantNC<(ggml_type)0, sycl::_V1::detail::half_impl::half, 1>   16  128   8      0       0      0    0    0 
               DequantNC<(ggml_type)0, sycl::_V1::detail::half_impl::half, 2>   16  128   8      0       0      0    0    0 
               DequantNC<(ggml_type)1, float, 0>                          16  128   8      0       0      0    0    0 
               DequantNC<(ggml_type)1, float, 1>                          16  128   8      0       0      0    0    0 
               DequantNC<(ggml_type)1, float, 2>                          16  128   8      0       0      0    0    0 
               DequantNC<(ggml_type)0, float, 0>                          16  128   8      0       0      0    0    0 
               DequantNC<(ggml_type)0, float, 1>                          16  128   8      0       0      0    0    0 
               DequantNC<(ggml_type)0, float, 2>                          16  128   8      0       0      0    0    0 
note: spill/private/slm in bytes as reported by IGC; slm is static only, dynamic SLM is budgeted in the design doc.

Static acm-g10 report (F16 arithmetic disabled)
device acm-g10  kernel                                                   simd  grf thr  spill    priv    slm dpas fp64
               UnaryNC<0>                                                 16  128   8      0       0      0    0    0 
               UnaryNC<1>                                                 16  128   8      0       0      0    0    0 
               DequantNC<(ggml_type)1, sycl::_V1::detail::half_impl::half, 0>   16  128   8      0       0      0    0    0 
               DequantNC<(ggml_type)1, sycl::_V1::detail::half_impl::half, 1>   16  128   8      0       0      0    0    0 
               DequantNC<(ggml_type)1, sycl::_V1::detail::half_impl::half, 2>   16  128   8      0       0      0    0    0 
               DequantNC<(ggml_type)0, sycl::_V1::detail::half_impl::half, 0>   16  128   8      0       0      0    0    0 
               DequantNC<(ggml_type)0, sycl::_V1::detail::half_impl::half, 1>   16  128   8      0       0      0    0    0 
               DequantNC<(ggml_type)0, sycl::_V1::detail::half_impl::half, 2>   16  128   8      0       0      0    0    0 
               DequantNC<(ggml_type)1, float, 0>                          16  128   8      0       0      0    0    0 
               DequantNC<(ggml_type)1, float, 1>                          16  128   8      0       0      0    0    0 
               DequantNC<(ggml_type)1, float, 2>                          16  128   8      0       0      0    0    0 
               DequantNC<(ggml_type)0, float, 0>                          16  128   8      0       0      0    0    0 
               DequantNC<(ggml_type)0, float, 1>                          16  128   8      0       0      0    0    0 
               DequantNC<(ggml_type)0, float, 2>                          16  128   8      0       0      0    0    0 
note: spill/private/slm in bytes as reported by IGC; slm is static only, dynamic SLM is budgeted in the design doc.

Edges (F16 arithmetic disabled)
Device: Intel(R) Xeon(R) CPU E3-1505M v6 @ 3.00GHz; CPU device, mechanism check; reps=1; arithmetic=GGML_SYCL_F16=OFF
type dst D n_kv n_head_kv | upstream_us rows_us vec8_us D2D_us (median) | rows vec8 D2D bit-exact
q8_0 f16 32 19 3 | 107.225 25.060 34.044 40.861 | PASS PASS PASS
q4_0 f16 32 19 3 | 68.585 39.669 26.340 34.925 | PASS PASS PASS
q8_0 f32 32 19 3 | 73.296 24.094 6.397 56.645 | PASS PASS PASS
q4_0 f32 32 19 3 | 115.157 9.405 5.953 51.507 | PASS PASS PASS
q8_0 f16 96 19 3 | 95.576 43.175 10.820 31.847 | PASS PASS PASS
q4_0 f16 96 19 3 | 82.182 40.209 21.364 41.001 | PASS PASS PASS
q8_0 f32 96 19 3 | 111.486 41.674 30.284 48.509 | PASS PASS PASS
q4_0 f32 96 19 3 | 102.442 33.988 29.136 69.982 | PASS PASS PASS
q8_0 f16 224 19 3 | 118.790 51.819 44.449 45.416 | PASS PASS PASS
q4_0 f16 224 19 3 | 97.392 57.769 23.900 49.845 | PASS PASS PASS
q8_0 f32 224 19 3 | 106.178 61.995 42.903 85.604 | PASS PASS PASS
q4_0 f32 224 19 3 | 105.009 59.044 33.269 72.523 | PASS PASS PASS
q8_0 f16 512 19 3 | 133.983 114.924 78.371 75.752 | PASS PASS PASS
q4_0 f16 512 19 3 | 124.753 108.956 73.090 103.515 | PASS PASS PASS
q8_0 f32 512 19 3 | 129.616 127.985 43.456 76.150 | PASS PASS PASS
q4_0 f32 512 19 3 | 126.150 130.158 65.840 76.389 | PASS PASS PASS
q8_0 f16 768 19 3 | 204.359 219.460 76.388 84.915 | PASS PASS PASS
q4_0 f16 768 19 3 | 198.458 230.567 74.950 79.864 | PASS PASS PASS
q8_0 f32 768 19 3 | 187.647 263.424 82.870 77.370 | PASS PASS PASS
q4_0 f32 768 19 3 | 205.369 165.813 87.976 91.530 | PASS PASS PASS
unary f32->f16 D=1 rows=19 heads=3 batches=2 padded-strides: PASS
unary f32->f16 D=17 rows=19 heads=3 batches=2 padded-strides: PASS
unary f32->f16 D=67 rows=19 heads=3 batches=2 padded-strides: PASS
unary f32->f16 D=96 rows=19 heads=3 batches=2 padded-strides: PASS
unary f32->f16 D=255 rows=19 heads=3 batches=2 padded-strides: PASS
unary f32->f16 D=256 rows=19 heads=3 batches=2 padded-strides: PASS
unary f32->f16 D=513 rows=19 heads=3 batches=2 padded-strides: PASS
20/20 quantized edge combinations; 7/7 unary edge shapes bit-exact

AOT store widths checked in IGC assembly (F16 arithmetic enabled):
q8_0 -> f16: one store.ugm.d32x4 (16 bytes/item).
q4_0 -> f16: two store.ugm.d32x2 (8 bytes/store/item).
q8_0 -> f32: two store.ugm.d32x4 (16 bytes/store/item).
q4_0 -> f32: two store.ugm.d32x4 (16 bytes/store/item).
The source promises only 2-byte load alignment. IGC emitted d64 (q8_0)
and d32 (q4_0) gathers and d16u32 scale gathers, with no block/2D messages.

Backend commands and final before/after evidence
-----------------------------------------------
Baseline = pinned source plus the p6a CPU-only harness patch, no conversion edits.
After = the same harness plus conversion patches 0003 and 0004.
Every run uses SYCL0 with the script's validated Intel OpenCL CPU marker.
GIT_OPTIONAL_LOCKS=0 was added for subsequent Git/build commands.
One supplemental harness attempt was interrupted by us before numerical tests
(exit 130), then rerun. No test result from that attempt is counted.
Ninja -n -d explain identified deleted /tmp/icpx-* objects as SYCL link
prerequisites, causing repeated incremental links. No build-source/script fix.
No tests or tolerances were changed. Existing backend tolerances apply to FA;
the standalone conversion comparisons require equality of every output byte.

Use the common environment above and these commands for before and after:
  bin/sycl-cpu-tbo.sh -o FLASH_ATTN_EXT -p "$P"
  bin/sycl-cpu-tbo.sh -o FLASH_ATTN_EXT \
    --test-file /tmp/p8-sycl-kv-conversion/fa-cases.txt -p '^name=p8_'
  bin/sycl-cpu-tbo.sh -o CPY -p "$P_CPY"
  bin/sycl-cpu-tbo.sh -o GET_ROWS -p "$P_ROWS"
P_CPY='^type_src=(f32|f16|q8_0|q4_0),type_dst=(f32|f16|q8_0|q4_0),ne_src=\[(64|96|128|256),'
P_ROWS='^type=(f32|f16|q8_0|q4_0),n=256,m=(5|8),r=(4|2),be1=(1|7),be2=1,'
The exact p6a one-line P filter is reproduced below:
P='((^hsk=(64|128|256),hsv=(64|128|256),nh=[124],nr23=\[(1|2|4|6|8|12),1\],kv=(113|128|256|512|1024|4096),nb=(1|2|3|4|16|32),mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=(f16|q8_0|q4_0),type_V=(f16|q8_0|q4_0),permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$)|(^hsk=(64|128|256),hsv=(64|128|256),nh=(1|8),nr23=\[8,1\],kv=4096,nb=(1|4),mask=1,sinks=0,max_bias=(0|8)\.000000,logit_softcap=0\.000000,prec=f32,type_K=(f16|q8_0),type_V=(f16|q8_0),permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=(0|512)$))|(^hsk=64,hsv=64,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=64,hsv=64,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=64,hsv=64,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q4_0,type_V=q4_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=128,hsv=128,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=128,hsv=128,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=128,hsv=128,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q4_0,type_V=q4_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=256,hsv=256,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=256,hsv=256,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=256,hsv=256,nh=8,nr23=\[4,1\],kv=1024,nb=32,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q4_0,type_V=q4_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=128,hsv=128,nh=4,nr23=\[12,1\],kv=512,nb=3,mask=1,sinks=0,max_bias=8\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$|^hsk=576,hsv=512,nh=1,nr23=\[20,1\],kv=512,nb=1,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=f16,type_V=f16,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=1,n_kv_max=0$|^hsk=256,hsv=256,nh=2,nr23=\[16,1\],kv=113,nb=1,mask=1,sinks=0,max_bias=0\.000000,logit_softcap=0\.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$)'

phase | before summary | after summary | before statuses | after statuses
fa | 85/85 tests passed | 85/85 tests passed | {'OK': 85} | {'OK': 85}
fa-extra | 24/24 tests passed | 24/24 tests passed | {'OK': 24} | {'OK': 24}
cpy | 52/52 tests passed | 52/52 tests passed | {'OK': 52} | {'OK': 52}
get-rows | 33/33 tests passed | 33/33 tests passed | {'OK': 33} | {'OK': 33}
before-fa routes: {'MKL': 10, 'TILE': 62, 'VEC': 13}
after-fa routes: {'MKL': 10, 'TILE': 62, 'VEC': 13}
before-fa-extra routes: {'TILE': 24}
after-fa-extra routes: {'TILE': 24}

The 85-case corpus includes q8_0: 17 cases (nb=1:6, nb=3:4, nb=32:7)
and q4_0: 15 cases (nb=1:4, nb=3:4, nb=32:7). Padded cache-view cases
exercise nc conversion; the supplemental query-count matrix below exercises
the contiguous-allocation converter, not the nc converter.

Supplemental cases use the runner's existing --test-file format, with no
existing case edits: D=64/128/256, query tokens=3/4/8/16, q8_0/q4_0 K/V,
KV=512, four KV heads, GQA=4, F32 queries, F16 causal mask, F32 precision.
Cache view strides are token-major (s01=heads*D/32 and s02=D/32 blocks),
matching the standalone benchmark. These allocations are contiguous despite
permuted strides and use the contiguous conversion path. The existing
p6a cases include genuinely padded K/V views and exercise the nc converter.
Output shape=[D,16,query_tokens,1].
To recreate the exact file, write these 24 lines to the --test-file path:
74 0 64 16 3 1 4 1040187392 0 0 10 4 0 64 3 16 1 0 0 0 0 8 64 512 4 1 34 272 68 139264 8 64 512 4 1 34 272 68 139264 1 512 3 1 1 0 0 0 0 p8_D64_nb3_q8_0
74 0 64 16 3 1 4 1040187392 0 0 10 4 0 64 3 16 1 0 0 0 0 2 64 512 4 1 18 144 36 73728 2 64 512 4 1 18 144 36 73728 1 512 3 1 1 0 0 0 0 p8_D64_nb3_q4_0
74 0 64 16 4 1 4 1040187392 0 0 10 4 0 64 4 16 1 0 0 0 0 8 64 512 4 1 34 272 68 139264 8 64 512 4 1 34 272 68 139264 1 512 4 1 1 0 0 0 0 p8_D64_nb4_q8_0
74 0 64 16 4 1 4 1040187392 0 0 10 4 0 64 4 16 1 0 0 0 0 2 64 512 4 1 18 144 36 73728 2 64 512 4 1 18 144 36 73728 1 512 4 1 1 0 0 0 0 p8_D64_nb4_q4_0
74 0 64 16 8 1 4 1040187392 0 0 10 4 0 64 8 16 1 0 0 0 0 8 64 512 4 1 34 272 68 139264 8 64 512 4 1 34 272 68 139264 1 512 8 1 1 0 0 0 0 p8_D64_nb8_q8_0
74 0 64 16 8 1 4 1040187392 0 0 10 4 0 64 8 16 1 0 0 0 0 2 64 512 4 1 18 144 36 73728 2 64 512 4 1 18 144 36 73728 1 512 8 1 1 0 0 0 0 p8_D64_nb8_q4_0
74 0 64 16 16 1 4 1040187392 0 0 10 4 0 64 16 16 1 0 0 0 0 8 64 512 4 1 34 272 68 139264 8 64 512 4 1 34 272 68 139264 1 512 16 1 1 0 0 0 0 p8_D64_nb16_q8_0
74 0 64 16 16 1 4 1040187392 0 0 10 4 0 64 16 16 1 0 0 0 0 2 64 512 4 1 18 144 36 73728 2 64 512 4 1 18 144 36 73728 1 512 16 1 1 0 0 0 0 p8_D64_nb16_q4_0
74 0 128 16 3 1 4 1035273459 0 0 10 4 0 128 3 16 1 0 0 0 0 8 128 512 4 1 34 544 136 278528 8 128 512 4 1 34 544 136 278528 1 512 3 1 1 0 0 0 0 p8_D128_nb3_q8_0
74 0 128 16 3 1 4 1035273459 0 0 10 4 0 128 3 16 1 0 0 0 0 2 128 512 4 1 18 288 72 147456 2 128 512 4 1 18 288 72 147456 1 512 3 1 1 0 0 0 0 p8_D128_nb3_q4_0
74 0 128 16 4 1 4 1035273459 0 0 10 4 0 128 4 16 1 0 0 0 0 8 128 512 4 1 34 544 136 278528 8 128 512 4 1 34 544 136 278528 1 512 4 1 1 0 0 0 0 p8_D128_nb4_q8_0
74 0 128 16 4 1 4 1035273459 0 0 10 4 0 128 4 16 1 0 0 0 0 2 128 512 4 1 18 288 72 147456 2 128 512 4 1 18 288 72 147456 1 512 4 1 1 0 0 0 0 p8_D128_nb4_q4_0
74 0 128 16 8 1 4 1035273459 0 0 10 4 0 128 8 16 1 0 0 0 0 8 128 512 4 1 34 544 136 278528 8 128 512 4 1 34 544 136 278528 1 512 8 1 1 0 0 0 0 p8_D128_nb8_q8_0
74 0 128 16 8 1 4 1035273459 0 0 10 4 0 128 8 16 1 0 0 0 0 2 128 512 4 1 18 288 72 147456 2 128 512 4 1 18 288 72 147456 1 512 8 1 1 0 0 0 0 p8_D128_nb8_q4_0
74 0 128 16 16 1 4 1035273459 0 0 10 4 0 128 16 16 1 0 0 0 0 8 128 512 4 1 34 544 136 278528 8 128 512 4 1 34 544 136 278528 1 512 16 1 1 0 0 0 0 p8_D128_nb16_q8_0
74 0 128 16 16 1 4 1035273459 0 0 10 4 0 128 16 16 1 0 0 0 0 2 128 512 4 1 18 288 72 147456 2 128 512 4 1 18 288 72 147456 1 512 16 1 1 0 0 0 0 p8_D128_nb16_q4_0
74 0 256 16 3 1 4 1031798784 0 0 10 4 0 256 3 16 1 0 0 0 0 8 256 512 4 1 34 1088 272 557056 8 256 512 4 1 34 1088 272 557056 1 512 3 1 1 0 0 0 0 p8_D256_nb3_q8_0
74 0 256 16 3 1 4 1031798784 0 0 10 4 0 256 3 16 1 0 0 0 0 2 256 512 4 1 18 576 144 294912 2 256 512 4 1 18 576 144 294912 1 512 3 1 1 0 0 0 0 p8_D256_nb3_q4_0
74 0 256 16 4 1 4 1031798784 0 0 10 4 0 256 4 16 1 0 0 0 0 8 256 512 4 1 34 1088 272 557056 8 256 512 4 1 34 1088 272 557056 1 512 4 1 1 0 0 0 0 p8_D256_nb4_q8_0
74 0 256 16 4 1 4 1031798784 0 0 10 4 0 256 4 16 1 0 0 0 0 2 256 512 4 1 18 576 144 294912 2 256 512 4 1 18 576 144 294912 1 512 4 1 1 0 0 0 0 p8_D256_nb4_q4_0
74 0 256 16 8 1 4 1031798784 0 0 10 4 0 256 8 16 1 0 0 0 0 8 256 512 4 1 34 1088 272 557056 8 256 512 4 1 34 1088 272 557056 1 512 8 1 1 0 0 0 0 p8_D256_nb8_q8_0
74 0 256 16 8 1 4 1031798784 0 0 10 4 0 256 8 16 1 0 0 0 0 2 256 512 4 1 18 576 144 294912 2 256 512 4 1 18 576 144 294912 1 512 8 1 1 0 0 0 0 p8_D256_nb8_q4_0
74 0 256 16 16 1 4 1031798784 0 0 10 4 0 256 16 16 1 0 0 0 0 8 256 512 4 1 34 1088 272 557056 8 256 512 4 1 34 1088 272 557056 1 512 16 1 1 0 0 0 0 p8_D256_nb16_q8_0
74 0 256 16 16 1 4 1031798784 0 0 10 4 0 256 16 16 1 0 0 0 0 2 256 512 4 1 18 576 144 294912 2 256 512 4 1 18 576 144 294912 1 512 16 1 1 0 0 0 0 p8_D256_nb16_q4_0

Negative control and exact recovery
-----------------------------------
Only the vector kernel's source-head index was changed, in both the benchmark
and scratch convert.cpp. Destination indexing stayed unchanged.
Correct: i03*s03 + i02*s02 + int64_t(i01)*s01 + i00/32
Broken:  i03*s03 + ((i02 + 1) % ne02)*s02 + int64_t(i01)*s01 + i00/32
Rebuilt the benchmark with its normal build flags, then --reps 1.
Rebuilt llama.cpp with the normal harness and ran:
  bin/sycl-cpu-tbo.sh -o FLASH_ATTN_EXT -p "$P_NEG"
P_NEG='^hsk=64,hsv=64,nh=4,nr23=\[1,1\],kv=512,nb=3,mask=1,sinks=0,max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=\[0,1,2,3\],kv_view=1,v_is_view_of_k=0,n_kv_max=0$'
An initial generic token-major FA control passed (1/1, exit 0): its
allocation is contiguous, so it does not enter the changed nc kernel.
The padded built-in kv_view=1 case above is the effective FA control.
The 0/0 lines below are harness preflights, not numerical case summaries.
negative-micro: exit 1
MISMATCH type=q8_0 dst=f16 D=64 kv=4096 nh=4 variant=2 element=0
negative-fa: exit 1
  0/0 tests passed
[FA-DISP] #1 TILE D=64 n_kv=512 delta=0 V_ne1=512
[FLASH_ATTN_EXT] ERR = 2.058709064 > 0.000500000   FLASH_ATTN_EXT(hsk=64,hsv=64,nh=4,nr23=[1,1],kv=512,nb=3,mask=1,sinks=0,max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=q8_0,type_V=q8_0,permute=[0,1,2,3],kv_view=1,v_is_view_of_k=0,n_kv_max=0): FAIL
  0/1 tests passed
  Backend SYCL0: FAIL
FAIL
The two edited files were restored byte for byte from their positive snapshots.
The normal benchmark was rebuilt, then the full --reps 3 matrix and --edges
were run again; both returned exit 0. The same FA case was rerun via the
harness and returned exit 0. Recovery result lines:
q4_0 f32 256 16384 8 | 32853.379 21921.287 11956.654 16233.031 | PASS PASS PASS
48/48 type/destination/shape combinations bit-exact; 144/144 comparisons PASS
unary f32->f16 D=513 rows=19 heads=3 batches=2 padded-strides: PASS
20/20 quantized edge combinations; 7/7 unary edge shapes bit-exact
  0/0 tests passed
[FA-DISP] #1 TILE D=64 n_kv=512 delta=0 V_ne1=512
  1/1 tests passed

Final isolation and limits
--------------------------
Patch series forward/reverse applicability checked on the scratch clone;
applying then reversing restored convert.cpp byte for byte. Final source
has both patches, no negative-control edit. Non-conversion git diff equals
the initial four-file p6a-only diff byte for byte. convert.hpp is unchanged.
No commit, push, installation, network, model weights, or other-worker
process termination. Required scan/lint and repo ctest are recorded above.
Not verified: A770 execution/correctness/timing, end-to-end decode, exhaustive
backend coverage, or runtime INT_MAX fallback. No ship decision is made.

Manager verification (Claude Opus 5.5, 2026-10-10)
--------------------------------------------------
Reran on the after build: microbenchmark F16 and F32 arithmetic, --reps 1 and
--edges, all exit 0, 48/48 combinations bit-exact in each mode; p6a FA subset
85/85; GET_ROWS 33/33; CPY 51/52.
The CPY failure is pre-existing and intermittent, not from 0003/0004:
  CPY(type_src=f16,type_dst=q4_0,ne_src=[256,4,4,4],permute_src=[0,0,0,0],...)
  ERR = 0.000005403 > 0.000003815 (f16 -> q4_0 quantisation; random inputs)
The same case run 6 times: patched build 6/6 OK; a build with upstream
convert.cpp (p6b scratch clone, convert.cpp unchanged) 4/6 OK, 2 FAIL
(ERR 5.3e-6, 7.9e-6). Quantisation goes through cpy.cpp, not the dequant
paths changed here. Flag for the A770 run: expect this case to flake.
Independent negative control on the microbenchmark: q8_0 vec8 source offset
x->qs + iqs changed to x->qs + (iqs ^ 8) -> "MISMATCH type=q8_0 dst=f16 D=64
... variant=2", exit 1. (An index change applied to both load and store, i00
^ 8, is a permutation of work-items and correctly still passes.)
