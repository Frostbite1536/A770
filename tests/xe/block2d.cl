// 2D block loads exist on Xe-HPC/Xe2, not on acm-g10: the builtin is undeclared there.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void block2d(__global ushort* x, __global ushort* y) {
    ushort8 v;
    intel_sub_group_2d_block_read_16b_8r16x1c(x, 64, 8, 64, (int2)(0, 0), (private ushort*)&v);
    y[get_global_id(0)] = v.s0;
}
