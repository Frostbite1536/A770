cmake_minimum_required(VERSION 3.24)
project(a770 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# SYCL is optional: with the default host compiler (dev laptop, CI) only the CPU reference and tests are
# built. Kernels still get static Xe-HPG reports via bin/xe-report.py (ocloc, no GPU needed).
# With icpx (oneAPI 2026.1, INV-ARCH-003) kernels are compiled ahead of time for the A770 only.
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 2026.1 OR CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 2026.2)
    message(FATAL_ERROR "A770 pins oneAPI DPC++ 2026.1.x (INV-ARCH-003), got ${CMAKE_CXX_COMPILER_VERSION}")
  endif()
  set(A770_DEVICE "acm-g10" CACHE STRING "AOT device; acm-g10 is the A770/A750 die (INV-ARCH-001)")
  if(NOT A770_DEVICE STREQUAL "acm-g10")
    message(FATAL_ERROR "A770 targets acm-g10 only (INV-ARCH-001), got ${A770_DEVICE}")
  endif()
  # Applied per kernel target with a770_sycl_target() as kernels land; host-only targets stay plain C++.
  function(a770_sycl_target tgt)
    target_compile_options(${tgt} PRIVATE -fsycl -fsycl-targets=spir64_gen)
    target_link_options(${tgt} PRIVATE -fsycl -fsycl-targets=spir64_gen
                        "SHELL:-Xsycl-target-backend \"-device ${A770_DEVICE}\"")
  endfunction()
  set(A770_HAVE_SYCL ON)
else()
  message(STATUS "A770: no icpx, building CPU reference and tests only")
  set(A770_HAVE_SYCL OFF)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  # icpx defaults to -fp-model=fast (finite-math-only, reassociation): the tests check -inf LSE and
  # NaN-free output (INV-NUM-002), so the whole project builds precise. A kernel that wants fast math
  # opts in per target, with the parity tests as the judge.
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fp-model=precise>)
endif()

enable_testing()

# Kernel targets (src/attention) will be added under if(A770_HAVE_SYCL) as they land.

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/reference/CMakeLists.txt)
  add_subdirectory(src/reference)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
  add_subdirectory(tests)
endif()
