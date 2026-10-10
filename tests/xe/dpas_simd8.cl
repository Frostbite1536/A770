// Xe-HPG XMX: 8x8x16 f16 DPAS at sub-group 8. A is int8 (two halves per work-item per row).
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void dpas_simd8(__global const int8* a, __global const int8* b, __global float8* c) {
    const int i = get_global_id(0);
    c[i] = intel_sub_group_f16_f16_matrix_mad_k16(a[i], b[i], (float8)(0.0f));
}
