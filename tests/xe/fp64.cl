// The A770 has no FP64 hardware; IGC refuses cl_khr_fp64 for acm-g10.
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void fp64(__global const double* x, __global double* y) { const int i = get_global_id(0); y[i] = x[i] * x[i] + 1.0; }
