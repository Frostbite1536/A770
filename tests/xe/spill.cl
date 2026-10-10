// 48 live float16 accumulators per work-item at SIMD16 cannot fit 128 GRFs: IGC spills.
#define N 48
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void spill(__global const float16* x, __global float16* y, int n) {
    const int g = get_global_id(0);
    float16 acc[N];
    #pragma unroll
    for (int i = 0; i < N; ++i) acc[i] = x[g + i];
    for (int k = 0; k < n; ++k) {
        #pragma unroll
        for (int i = 0; i < N; ++i) acc[i] = mad(acc[i], acc[(i + 1) % N], x[k]);
    }
    float16 s = 0;
    #pragma unroll
    for (int i = 0; i < N; ++i) s += acc[i];
    y[g] = s;
}
