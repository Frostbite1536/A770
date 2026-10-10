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

# bin/xe-report.py must keep catching the Xe-HPG traps it exists for. Needs ocloc (bin/setup-dev.sh);
# skipped (code 77) without it.
if(Python3_Interpreter_FOUND)
  foreach(case dpas_simd8 large_grf dpas_simd16_silent spill fp64 block2d slm sycl_joint_matrix compile_commands)
    add_test(NAME xe_report_${case}
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_xe_report.py ${case})
    set_tests_properties(xe_report_${case} PROPERTIES SKIP_RETURN_CODE 77)
  endforeach()
endif()
