// The Xe2/Xe-HPC form (sub-group 16, short8 A). On acm-g10 this compiles with no error and no DPAS
// instruction; the output registers are never written. xe-report must flag it (NO-DPAS).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void dpas_simd16(__global const short8* a, __global const int8* b, __global float8* c) {
    const int i = get_global_id(0);
    c[i] = intel_sub_group_f16_f16_matrix_mad_k16(a[i], b[i], (float8)(0.0f));
}
