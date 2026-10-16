foreach(name test_fp16 test_reference test_tiled_model error_budget)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE a770_ref)
endforeach()
add_test(NAME fp16 COMMAND test_fp16)
foreach(case tiny uniform double gqa causal masked softcap large validation)
  add_test(NAME reference_${case} COMMAND test_reference ${case})
endforeach()
foreach(case matrix masked rounding validation)
  add_test(NAME tiled_${case} COMMAND test_tiled_model ${case})
endforeach()
add_test(NAME error_budget COMMAND error_budget)

# bench/analyze.py must parse real test-backend-ops SQL output and llama-bench JSONL (fixtures carried over from
# volta-fa: the formats are backend-independent).
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(NAME mm_gbps
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/mm-gbps/test_mm_gbps.py)
  add_test(NAME step_breakdown
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/step-breakdown/test_step_breakdown.py)
  # Hand checks: Q/K/V/S/P = 8192+16384+16384+8192+4096 = 53248 B.
  add_test(NAME plan_tiles_smoke
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/bin/plan-tiles.py
            --d 128 --br 32 --bc 64 --sg-size 8 --sgs-per-wg 4 --grf 128)
  set_tests_properties(plan_tiles_smoke PROPERTIES PASS_REGULAR_EXPRESSION "= 53248 B")
  # G=7: exact divisor ncols2=1 -> 7 passes; padded 8, 1*8<=32 -> 1 pass.
  add_test(NAME decode_gqa_model_smoke
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/bin/decode-gqa-model.py
            --d 128 --kv-heads 4 --gqa 7 --depth 32768 --tokens 1)
  set_tests_properties(decode_gqa_model_smoke PROPERTIES
    PASS_REGULAR_EXPRESSION "passes upstream = 7; passes padded = 1")
  add_test(NAME analyze_tbo
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/bench/analyze.py tbo
            ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tbo-perf-cpu.sql ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tbo-perf-cpu.sql)
  set_tests_properties(analyze_tbo PROPERTIES PASS_REGULAR_EXPRESSION "type_V=q8_0 \\| [0-9.]+ ms \\| [0-9.]+ ms \\| 1\\.00×")
  add_test(NAME analyze_lb_rounds
    COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/bench/analyze.py lb
            ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/lb-r1.jsonl,${CMAKE_CURRENT_SOURCE_DIR}/fixtures/lb-r2.jsonl
            ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/lb-r3.jsonl --labels a,b)
  set_tests_properties(analyze_lb_rounds PROPERTIES PASS_REGULAR_EXPRESSION "105\\.00 \\(100\\.00–110\\.00, n=2\\) \\| 130\\.00 ± 1\\.00 \\| \\+23\\.8%")
endif()

# Bench-only range cases intentionally fail on q8_1 paths. This ctest checks that
# the CPU-device run completes all 96 cases; numerical outcomes remain in its log.
# It does not assert that those known numerical failures have been fixed.
set(sycl_cpu_runtime /home/p51/Github/A770/tools/oneapi-2026.1)
if(EXISTS "${sycl_cpu_runtime}/etc/OpenCL/vendors/intel-ocl-cpu.icd" AND
   EXISTS "${sycl_cpu_runtime}/lib/intel-ocl-cpu/libintelocl.so" AND
   EXISTS "${sycl_cpu_runtime}/bin/icpx")
  file(CONFIGURE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/sycl-cpu-range.cmake CONTENT [=[
set(llama "@PROJECT_BINARY_DIR@/llama")
set(log "@PROJECT_BINARY_DIR@/p14-range/ctest-range-sycl.txt")
file(MAKE_DIRECTORY "@PROJECT_BINARY_DIR@/p14-range")
if(NOT EXISTS "${llama}/.git")
  # Local clone only: never fetch, install, or write into the main checkout.
  set(source "@PROJECT_SOURCE_DIR@/third_party/llama.cpp")
  if(NOT EXISTS "${source}/.git")
    set(source "/home/p51/Github/A770/third_party/llama.cpp")
  endif()
  if(NOT EXISTS "${source}/.git")
    message("SKIP: local pinned llama.cpp clone unavailable for SYCL CPU range cases")
    return()
  endif()
  execute_process(COMMAND git clone --no-hardlinks "${source}" "${llama}" RESULT_VARIABLE rc)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "Local llama.cpp clone failed: ${rc}")
  endif()
endif()
execute_process(
  COMMAND "@CMAKE_COMMAND@" -E env
    --unset=GGML_SYCL_ENABLE_OPT --unset=GGML_SYCL_XMX_GATHER_TYPES --unset=GGML_SYCL_XMX_GATHER_SHAPES
    "LLAMA=${llama}" SYCL_CPU_BENCH_PATCHES=1
    SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=/tmp/p14-sycl-cache
    bash "@PROJECT_SOURCE_DIR@/bin/sycl-cpu-tbo.sh" test -o MUL_MAT -b SYCL0 -p src1_scale
  RESULT_VARIABLE rc OUTPUT_FILE "${log}" ERROR_FILE "${log}" TIMEOUT 7200)
file(READ "${log}" output)
if(NOT "${rc}" MATCHES "^[01]$" OR
   NOT output MATCHES "GGML_SYCL_TEST_CPU: SYCL0 is Intel OpenCL CPU:")
  message(FATAL_ERROR "SYCL CPU range harness failed (${rc}); see ${log}")
endif()
string(ASCII 27 esc)
string(REGEX REPLACE "${esc}[[][0-9;]*m" "" output "${output}")
string(REGEX MATCHALL "MUL_MAT[(][^\n]*src1_scale=[^\n]*[)]: [^\n]*(OK|FAIL)" cases "${output}")
list(LENGTH cases count)
string(REGEX MATCH "([0-9]+)/96 tests passed" summary "${output}")
set(passed "${CMAKE_MATCH_1}")
string(REGEX MATCHALL "MUL_MAT[(][^\n]*src1_scale=[^\n]*[)]: [^\n]*OK" ok_cases "${output}")
list(LENGTH ok_cases ok_count)
if(NOT count EQUAL 96 OR NOT summary)
  message(FATAL_ERROR "Incomplete SYCL CPU range corpus (${count}/96 cases); see ${log}")
endif()
if(NOT ok_count EQUAL passed OR
   (rc EQUAL 0 AND NOT passed EQUAL 96) OR (rc EQUAL 1 AND passed EQUAL 96))
  message(FATAL_ERROR "SYCL CPU range exit status disagrees with summary; see ${log}")
endif()
message("SYCL on the Intel OpenCL CPU device: ${summary}; rc=${rc}; ${log}")
message("Bench-only corpus execution checked; these fail on q8_1 paths by design.")
]=] @ONLY)
  add_test(NAME sycl_cpu_mul_mat_range
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/sycl-cpu-range.cmake)
  # Clones and builds SYCL llama.cpp on first run (slow): `ctest -LE sycl_cpu` skips it.
  set_tests_properties(sycl_cpu_mul_mat_range PROPERTIES TIMEOUT 7300 RUN_SERIAL TRUE
    SKIP_REGULAR_EXPRESSION "SKIP:" LABELS sycl_cpu)
else()
  message(STATUS "SYCL CPU range cases skipped: oneAPI CPU runtime/compiler not installed")
  add_test(NAME sycl_cpu_mul_mat_range COMMAND ${CMAKE_COMMAND} -E echo
    "SKIP: oneAPI CPU runtime/compiler not installed for SYCL CPU range cases")
  set_tests_properties(sycl_cpu_mul_mat_range PROPERTIES SKIP_REGULAR_EXPRESSION "SKIP:")
endif()

# Phase 0 has no dry-run mode; check shell syntax without requiring an A770.
add_test(NAME phase0_syntax COMMAND bash -n ${PROJECT_SOURCE_DIR}/bench/phase0.sh)

# bin/xe-report.py must keep catching the Xe-HPG traps it exists for. Needs ocloc (bin/setup-dev.sh);
# skipped (code 77) without it.
if(Python3_Interpreter_FOUND)
  foreach(case dpas_simd8 large_grf dpas_simd16_silent spill fp64 block2d slm sycl_joint_matrix compile_commands)
    add_test(NAME xe_report_${case}
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_xe_report.py ${case})
    set_tests_properties(xe_report_${case} PROPERTIES SKIP_RETURN_CODE 77)
  endforeach()
endif()
