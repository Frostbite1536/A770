add_library(a770_ref STATIC attention_reference.cpp attention_tiled_model.cpp)
target_include_directories(a770_ref PUBLIC ${PROJECT_SOURCE_DIR}/include)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Keep the CPU error sweep practical even in the default (untyped) build.
  # No fast-math: the explicit FP32 rounding points are part of the model.
  target_compile_options(a770_ref PRIVATE -O2 -ffp-contract=off)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  # Precise comes from the top level; precise still allows FMA contraction, which changes the model's rounding.
  target_compile_options(a770_ref PRIVATE -O2 -ffp-contract=off -Wno-overriding-option)
endif()
