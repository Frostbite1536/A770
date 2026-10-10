// SYCL joint_matrix at the Xe-HPG XMX shape (8x8x16 f16, sub-group 8). The Xe2 shape (8x16x16 at
// sub-group 16) crashes IGC for acm-g10 (ICE, 2026-10-10), so only the legal shape is instantiated.
#include <sycl/sycl.hpp>
namespace jm = sycl::ext::oneapi::experimental::matrix;
template <int SG, int M, int N, int K>
void mm(sycl::queue& q, const sycl::half* a, const sycl::half* b, float* c) {
    q.parallel_for(sycl::nd_range<1>{SG, SG}, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, M, K, jm::layout::row_major> ma;
        jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, K, N, jm::layout::row_major> mb;
        jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator, M, N> mc;
        jm::joint_matrix_fill(sg, mc, 0.0f);
        auto ga = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(a);
        auto gb = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(b);
        auto gc = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(c);
        jm::joint_matrix_load(sg, ma, ga, K);
        jm::joint_matrix_load(sg, mb, gb, N);
        jm::joint_matrix_mad(sg, mc, ma, mb, mc);
        jm::joint_matrix_store(sg, mc, gc, N, jm::layout::row_major);
    });
}
template void mm<8, 8, 8, 16>(sycl::queue&, const sycl::half*, const sycl::half*, float*);
