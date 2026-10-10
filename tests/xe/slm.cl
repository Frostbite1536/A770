// SLM_FLOATS * 4 bytes of shared local memory per work-group.
#ifndef SLM_FLOATS
#define SLM_FLOATS 16384
#endif
__kernel void slm(__global const float* x, __global float* y) {
    __local float t[SLM_FLOATS];
    const int l = get_local_id(0);
    for (int i = l; i < SLM_FLOATS; i += get_local_size(0)) t[i] = x[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    y[get_global_id(0)] = t[(l * 7) % SLM_FLOATS];
}
