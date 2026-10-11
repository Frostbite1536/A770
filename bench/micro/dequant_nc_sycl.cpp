// Non-contiguous K/V conversion at llama.cpp 79e2e74. CPU runs are mechanism checks.
// Build (JIT spir64, one TU; no llama.cpp headers or libraries):
// /home/p51/Github/A770/tools/oneapi-2026.1/bin/icpx -O2 -std=c++17 -fsycl \
//   -fsycl-device-code-split=per_kernel bench/micro/dequant_nc_sycl.cpp -o /tmp/dequant_nc_sycl
// ONEAPI_DEVICE_SELECTOR=opencl:cpu /tmp/dequant_nc_sycl --reps 5
// --edges checks ragged work-groups, long rows, and unary padded 4-D views.
// Add -DP8_FLOAT_ARITH to test GGML_SYCL_F16=OFF. Both half and float outputs are checked.
// FA cache view: ne=[D,n_kv,n_head_kv], s01=n_head_kv*D/32, s02=D/32 (blocks).
// Upstream helpers and generic body below are copied byte for byte; the rows variant
// changes only row indexing and its bounds check. Vec8 matches the patched body.
#include <sycl/sycl.hpp>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>
#ifndef P8_FLOAT_ARITH
#define GGML_SYCL_F16
#endif
#define __dpct_inline__ inline
constexpr int WARP_SIZE = 16, SYCL_DEQUANTIZE_BLOCK_SIZE = 256;
constexpr int QK4_0 = 32, QR4_0 = 2, QK8_0 = 32, QR8_0 = 1;
enum ggml_type : int { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0 };
struct block_q4_0 { sycl::half d; uint8_t qs[16]; };
struct block_q8_0 { sycl::half d; int8_t qs[32]; };
static_assert(sizeof(block_q4_0) == 18 && sizeof(block_q8_0) == 34);
static_assert(alignof(block_q4_0) == 2 && alignof(block_q8_0) == 2);
#ifdef GGML_SYCL_F16
using dfloat = sycl::half;
using dfloat2 = sycl::half2;
#else
using dfloat = float;
using dfloat2 = sycl::float2;
#endif
using dequantize_kernel_t = void (*)(const void *, int64_t, int, dfloat2 &);
template<typename dst_t, typename src_t>
inline dst_t ggml_sycl_cast(src_t x) {
    if constexpr (std::is_same_v<dst_t, src_t>) { return x; }
    else { return float(x); }
}

static __dpct_inline__ void dequantize_q4_0(const void *vx, const int64_t ib,
                                            const int iqs, dfloat2 &v) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const dfloat d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

#ifdef GGML_SYCL_F16
    // v = v - {8.0f, 8.0f};
    // v = v * {d, d};
    v.s0() = (v.s0() - 8.0f) * d;
    v.s1() = (v.s1() - 8.0f) * d;

#else
    v.x() = (v.x() - 8.0f) * d;
    v.y() = (v.y() - 8.0f) * d;
#endif // GGML_SYCL_F16
}

static __dpct_inline__ void dequantize_q8_0(const void *vx, const int64_t ib,
                                            const int iqs, dfloat2 &v) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const dfloat d = x[ib].d;

    v.x() = x[ib].qs[iqs + 0];
    v.y() = x[ib].qs[iqs + 1];

#ifdef GGML_SYCL_F16
    // v = v * {d, d};
    v.s0() *= d;
    v.s1() *= d;
#else
    v.x() *= d;
    v.y() *= d;
#endif // GGML_SYCL_F16
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_nc(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    auto          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i00 = 2 * (int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2));

    if (i00 >= ne00) {
        return;
    }

    const int64_t i01 = item_ct1.get_group(1);
    const int64_t i02 = item_ct1.get_group(0) % ne02;
    const int64_t i03 = item_ct1.get_group(0) / ne02;

    const int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

    const int64_t ib = ibx0 + i00/qk; // block index
    const int64_t iqs = (i00%qk)/qr; // quant index
    const int64_t iybs = i00 - i00%qk; // y block start index
    const int64_t y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    #ifdef GGML_SYCL_F16
        sycl::half2 v;
    #else
        sycl::float2 v;
    #endif

    dequantize_kernel(vx, ib, iqs, v);

    const int64_t iy0 = ((i03*ne02 + i02)*ne01 + i01)*ne00 + iybs + iqs;
    y[iy0 + 0]        = ggml_sycl_cast<dst_t>(v.x());
    y[iy0 + y_offset] = ggml_sycl_cast<dst_t>(v.y());
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_nc_rows(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    auto          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i00 = 2 * (int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2));

    if (i00 >= ne00) {
        return;
    }

    const int64_t i01 = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    if (i01 >= ne01) {
        return;
    }
    const int64_t i02 = item_ct1.get_group(0) % ne02;
    const int64_t i03 = item_ct1.get_group(0) / ne02;

    const int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

    const int64_t ib = ibx0 + i00/qk; // block index
    const int64_t iqs = (i00%qk)/qr; // quant index
    const int64_t iybs = i00 - i00%qk; // y block start index
    const int64_t y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    #ifdef GGML_SYCL_F16
        sycl::half2 v;
    #else
        sycl::float2 v;
    #endif

    dequantize_kernel(vx, ib, iqs, v);

    const int64_t iy0 = ((i03*ne02 + i02)*ne01 + i01)*ne00 + iybs + iqs;
    y[iy0 + 0]        = ggml_sycl_cast<dst_t>(v.x());
    y[iy0 + y_offset] = ggml_sycl_cast<dst_t>(v.y());
}

template <ggml_type type, typename dst_t>
static void dequantize_block_q_0_nc(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const sycl::nd_item<3> & item_ct1) {
    static_assert(type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q8_0, "bad type");
    static_assert(QK4_0 == 32 && QK8_0 == 32, "bad block size");
    const int nth_row = ne00/8;
    const int64_t i = item_ct1.get_global_id(2);
    if (i >= ne01*nth_row) {
        return;
    }
    const int i01 = i / nth_row;
    const int64_t i00 = 8*(i - int64_t(i01)*nth_row);
    const int iqs = i00 % 32;
    const int64_t i02 = item_ct1.get_group(0) % ne02;
    const int64_t i03 = item_ct1.get_group(0) / ne02;
    const int64_t ib = i03*s03 + i02*s02 + int64_t(i01)*s01 + i00/32;
    dst_t * yb = y + ((i03*ne02 + i02)*ne01 + i01)*ne00 + (i00 - iqs);

    if constexpr (type == GGML_TYPE_Q8_0) {
        const block_q8_0 * x = (const block_q8_0 *) vx + ib;
        const dfloat d = x->d;
        int8_t q[8];
        __builtin_memcpy(q, __builtin_assume_aligned(x->qs + iqs, 2), 8);
        alignas(16) dst_t values[8];
#pragma unroll
        for (int k = 0; k < 8; k += 2) {
            dfloat2 v;
            v.x() = q[k + 0];
            v.y() = q[k + 1];
#ifdef GGML_SYCL_F16
            v.s0() *= d;
            v.s1() *= d;
#else
            v.x() *= d;
            v.y() *= d;
#endif
            values[k + 0] = ggml_sycl_cast<dst_t>(v.x());
            values[k + 1] = ggml_sycl_cast<dst_t>(v.y());
        }
        constexpr int ne_cpy = 16/sizeof(dst_t) < 8 ? 16/sizeof(dst_t) : 8;
#pragma unroll
        for (int k = 0; k < 8; k += ne_cpy) {
            __builtin_memcpy(__builtin_assume_aligned(yb + iqs + k, 16), values + k, ne_cpy*sizeof(dst_t));
        }
    } else {
        const block_q4_0 * x = (const block_q4_0 *) vx + ib;
        const dfloat d = x->d;
        uint8_t q[4];
        __builtin_memcpy(q, __builtin_assume_aligned(x->qs + iqs/2, 2), 4);
        alignas(16) dst_t lo[4];
        alignas(16) dst_t hi[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            dfloat2 v;
            v.x() = q[k] & 0xF;
            v.y() = q[k] >> 4;
#ifdef GGML_SYCL_F16
            v.s0() = (v.s0() - 8.0f) * d;
            v.s1() = (v.s1() - 8.0f) * d;
#else
            v.x() = (v.x() - 8.0f) * d;
            v.y() = (v.y() - 8.0f) * d;
#endif
            lo[k] = ggml_sycl_cast<dst_t>(v.x());
            hi[k] = ggml_sycl_cast<dst_t>(v.y());
        }
        __builtin_memcpy(__builtin_assume_aligned(yb +      iqs/2, 4*sizeof(dst_t)), lo, 4*sizeof(dst_t));
        __builtin_memcpy(__builtin_assume_aligned(yb + 16 + iqs/2, 4*sizeof(dst_t)), hi, 4*sizeof(dst_t));
    }
}

template <ggml_type type, typename dst_t, int variant> class DequantNC;
template <ggml_type type, typename dst_t, int variant>
static sycl::event launch(sycl::queue & queue, const void * x, dst_t * y,
        int64_t D, int64_t kv, int64_t nh) {
    constexpr int qr = type == GGML_TYPE_Q8_0 ? QR8_0 : QR4_0;
    constexpr dequantize_kernel_t dk = type == GGML_TYPE_Q8_0 ? dequantize_q8_0 : dequantize_q4_0;
    const int64_t s01 = nh*D/32, s02 = D/32, s03 = kv*s01;
    int64_t items = 256, rows = 1;
    if constexpr (variant == 1) {
        if (D < 512) {
            items = ((D + 2*WARP_SIZE - 1)/(2*WARP_SIZE))*WARP_SIZE;
            rows = 256/items;
        }
    }
    const sycl::range<3> local(1, rows, items);
    const sycl::range<3> groups = variant == 2 ? sycl::range<3>(nh, 1, (kv*(D/8)+255)/256) :
        sycl::range<3>(nh, (kv+rows-1)/rows, (D+2*items-1)/(2*items));
    return queue.parallel_for<DequantNC<type, dst_t, variant>>(sycl::nd_range<3>(groups*local, local),
        [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            if constexpr (variant == 0) dequantize_block_nc<32, qr, dk>(x, y, D, kv, nh, s01, s02, s03);
            if constexpr (variant == 1) dequantize_block_nc_rows<32, qr, dk>(x, y, D, kv, nh, s01, s02, s03);
            if constexpr (variant == 2) dequantize_block_q_0_nc<type>(x, y, D, kv, nh, s01, s02, s03, item);
        });
}

static double median_us(sycl::event event) {
    event.wait_and_throw();
    return (event.get_profiling_info<sycl::info::event_profiling::command_end>() -
            event.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-3;
}
template <ggml_type type, typename dst_t>
static void run(sycl::queue & queue, int D, int kv, int nh, int reps) {
    using block_t = std::conditional_t<type == GGML_TYPE_Q8_0, block_q8_0, block_q4_0>;
    const size_t ny = size_t(D)*kv*nh, nblk = ny/32, bytes = ny*sizeof(dst_t);
    std::vector<block_t> input(nblk);
    uint32_t r = 12345;
    for (auto & b : input) {
        r = r*1664525u + 1013904223u;
        // Include signed zero, subnormal, and positive/negative finite scales.
        const uint16_t scale = (r % 23 == 0) ? uint16_t((r & 0x8000) | (r & 0x3ff)) :
            uint16_t((r & 0x8000) | (0x1400 + ((r >> 8) % 0x4800)));
        std::memcpy(&b.d, &scale, 2);
        for (auto & q : b.qs) { r = r*1664525u + 1013904223u; q = uint8_t(r >> 24); }
    }
    auto * x = sycl::malloc_device<block_t>(nblk, queue);
    auto * y0 = sycl::malloc_device<dst_t>(ny, queue);
    auto * y1 = sycl::malloc_device<dst_t>(ny, queue);
    auto * y2 = sycl::malloc_device<dst_t>(ny, queue);
    if (!x || !y0 || !y1 || !y2) throw std::runtime_error("device allocation failed");
    queue.memcpy(x, input.data(), nblk*sizeof(block_t)).wait_and_throw();
    auto go = [&](int variant) {
        if (variant == 0) return launch<type, dst_t, 0>(queue, x, y0, D, kv, nh);
        if (variant == 1) return launch<type, dst_t, 1>(queue, x, y1, D, kv, nh);
        if (variant == 2) return launch<type, dst_t, 2>(queue, x, y2, D, kv, nh);
        return queue.memcpy(y2, y1, bytes);
    };
    double times[4];
    std::vector<dst_t> reference(ny), result(ny);
    for (int variant = 0; variant < 4; ++variant) {
        if (variant != 0) queue.memset(variant == 1 ? y1 : y2, 0xa5, bytes).wait_and_throw();
        go(variant).wait_and_throw(); // warm-up, exclude JIT
        std::vector<double> samples;
        for (int rep = 0; rep < reps; ++rep) samples.push_back(median_us(go(variant)));
        std::sort(samples.begin(), samples.end());
        times[variant] = samples.size()%2 ? samples[samples.size()/2] :
            (samples[samples.size()/2 - 1]+samples[samples.size()/2])/2;
        queue.memcpy(variant == 0 ? reference.data() : result.data(),
                     variant == 0 ? y0 : variant == 1 ? y1 : y2, bytes).wait_and_throw();
        if (variant && std::memcmp(reference.data(), result.data(), bytes)) {
            size_t first = 0;
            while (!std::memcmp(&reference[first], &result[first], sizeof(dst_t))) ++first;
            std::fprintf(stderr, "MISMATCH type=%s dst=%s D=%d kv=%d nh=%d variant=%d element=%zu\n",
                type == GGML_TYPE_Q8_0 ? "q8_0" : "q4_0", sizeof(dst_t) == 2 ? "f16" : "f32",
                D, kv, nh, variant, first);
            throw std::runtime_error("bit-exact comparison failed");
        }
    }
    std::printf("%s %s %d %d %d | %.3f %.3f %.3f %.3f | PASS PASS PASS\n",
        type == GGML_TYPE_Q8_0 ? "q8_0" : "q4_0", sizeof(dst_t) == 2 ? "f16" : "f32",
        D, kv, nh, times[0], times[1], times[2], times[3]);
    sycl::free(x, queue); sycl::free(y0, queue); sycl::free(y1, queue); sycl::free(y2, queue);
}
template <typename src_t, typename dst_t>
static void convert_unary_nc(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t ne00, const int64_t ne01,
                          const int64_t ne02, const int64_t s01, const int64_t s02, const int64_t s03,
                          const sycl::nd_item<3> & item_ct1) {

    const int64_t work_group_size = item_ct1.get_local_range(2);
    const int64_t global_id       = item_ct1.get_local_id(2) + work_group_size * item_ct1.get_group(2);

    const int64_t i01 = item_ct1.get_group(1);
    const int64_t i02 = item_ct1.get_group(0) % ne02;
    const int64_t i03 = item_ct1.get_group(0) / ne02;

    // make each work-item deal with more elements since sycl global range can not exceed max int
    const src_t * x = static_cast<const src_t *>(vx);
    const int64_t ix = i03 * s03 + i02 * s02 + i01 * s01;
    const int64_t iy = ((i03 * ne02 + i02) * ne01 + i01) * ne00;

#pragma unroll
    for (int64_t i00 = global_id; i00 < ne00; i00 += work_group_size * item_ct1.get_group_range(2)) {
        y[iy + i00] = static_cast<dst_t>(x[ix + i00]);
    }
}

template <typename src_t, typename dst_t>
static void convert_unary_nc_rows(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t ne00, const int64_t ne01,
                          const int64_t ne02, const int64_t s01, const int64_t s02, const int64_t s03,
                          const sycl::nd_item<3> & item_ct1) {

    const int64_t work_group_size = item_ct1.get_local_range(2);
    const int64_t global_id       = item_ct1.get_local_id(2) + work_group_size * item_ct1.get_group(2);

    const int64_t i01 = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    if (i01 >= ne01) {
        return;
    }
    const int64_t i02 = item_ct1.get_group(0) % ne02;
    const int64_t i03 = item_ct1.get_group(0) / ne02;

    // make each work-item deal with more elements since sycl global range can not exceed max int
    const src_t * x = static_cast<const src_t *>(vx);
    const int64_t ix = i03 * s03 + i02 * s02 + i01 * s01;
    const int64_t iy = ((i03 * ne02 + i02) * ne01 + i01) * ne00;

#pragma unroll
    for (int64_t i00 = global_id; i00 < ne00; i00 += work_group_size * item_ct1.get_group_range(2)) {
        y[iy + i00] = static_cast<dst_t>(x[ix + i00]);
    }
}

template <int variant> class UnaryNC;
template <int variant>
static sycl::event launch_unary(sycl::queue & queue, const float * x, sycl::half * y,
        int64_t D, int64_t rows, int64_t heads, int64_t batches,
        int64_t s01, int64_t s02, int64_t s03) {
    const int64_t items = variant == 1 && D < 256 ? ((D+15)/16)*16 : 256;
    const int64_t rows_per_wg = 256/items;
    const sycl::range<3> local(1, rows_per_wg, items);
    const sycl::range<3> groups(heads*batches, (rows+rows_per_wg-1)/rows_per_wg, (D+items-1)/items);
    return queue.parallel_for<UnaryNC<variant>>(sycl::nd_range<3>(groups*local, local),
        [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            if constexpr (variant == 0) convert_unary_nc<float>(x, y, D, rows, heads, s01, s02, s03, item);
            else convert_unary_nc_rows<float>(x, y, D, rows, heads, s01, s02, s03, item);
        });
}
static void unary_edges(sycl::queue & queue) {
    for (int D : {1, 17, 67, 96, 255, 256, 513}) {
        const int rows = 19, heads = 3, batches = 2;
        const int64_t s01 = heads*(D+7), s02 = D+7, s03 = rows*s01+13;
        const size_t nx = batches*s03, ny = size_t(D)*rows*heads*batches;
        std::vector<float> input(nx);
        for (size_t i = 0; i < nx; ++i) input[i] = (int(i%4093)-2046)*0.001f;
        auto * x = sycl::malloc_device<float>(nx, queue);
        auto * y0 = sycl::malloc_device<sycl::half>(ny, queue);
        auto * y1 = sycl::malloc_device<sycl::half>(ny, queue);
        if (!x || !y0 || !y1) throw std::runtime_error("device allocation failed");
        queue.memcpy(x, input.data(), nx*sizeof(float)).wait_and_throw();
        queue.memset(y1, 0xa5, ny*2).wait_and_throw();
        launch_unary<0>(queue, x, y0, D, rows, heads, batches, s01, s02, s03).wait_and_throw();
        launch_unary<1>(queue, x, y1, D, rows, heads, batches, s01, s02, s03).wait_and_throw();
        std::vector<sycl::half> a(ny), b(ny);
        queue.memcpy(a.data(), y0, ny*2).wait_and_throw();
        queue.memcpy(b.data(), y1, ny*2).wait_and_throw();
        if (std::memcmp(a.data(), b.data(), ny*2)) throw std::runtime_error("unary geometry MISMATCH");
        std::printf("unary f32->f16 D=%d rows=19 heads=3 batches=2 padded-strides: PASS\n", D);
        sycl::free(x, queue); sycl::free(y0, queue); sycl::free(y1, queue);
    }
}

int main(int argc, char ** argv) {
    try {
        int reps = 5;
        const bool edges = argc == 2 && !std::strcmp(argv[1], "--edges");
        if (edges) reps = 1;
        else if (argc != 1) {
            if (argc != 3 || std::strcmp(argv[1], "--reps")) throw std::runtime_error("usage: dequant_nc_sycl [--reps N | --edges]");
            char * end = nullptr;
            long parsed = std::strtol(argv[2], &end, 10);
            if (*end || parsed < 1 || parsed > 10000) throw std::runtime_error("--reps must be in [1,10000]");
            reps = int(parsed);
        }
        sycl::queue queue(sycl::default_selector_v,
            {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
        std::printf("Device: %s; %s; reps=%d; arithmetic=%s\n",
            queue.get_device().get_info<sycl::info::device::name>().c_str(),
            queue.get_device().is_cpu() ? "CPU device, mechanism check" : "GPU device",
            reps,
#ifdef GGML_SYCL_F16
            "GGML_SYCL_F16=ON"
#else
            "GGML_SYCL_F16=OFF"
#endif
        );
        std::puts("type dst D n_kv n_head_kv | upstream_us rows_us vec8_us D2D_us (median) | rows vec8 D2D bit-exact");
        if (edges) {
            for (int D : {32, 96, 224, 512, 768}) {
                run<GGML_TYPE_Q8_0, sycl::half>(queue, D, 19, 3, reps);
                run<GGML_TYPE_Q4_0, sycl::half>(queue, D, 19, 3, reps);
                run<GGML_TYPE_Q8_0, float>(queue, D, 19, 3, reps);
                run<GGML_TYPE_Q4_0, float>(queue, D, 19, 3, reps);
            }
            unary_edges(queue);
            std::puts("20/20 quantized edge combinations; 7/7 unary edge shapes bit-exact");
        } else {
            for (int D : {64, 128, 256}) for (int kv : {4096, 16384}) for (int nh : {4, 8}) {
                run<GGML_TYPE_Q8_0, sycl::half>(queue, D, kv, nh, reps);
                run<GGML_TYPE_Q4_0, sycl::half>(queue, D, kv, nh, reps);
                run<GGML_TYPE_Q8_0, float>(queue, D, kv, nh, reps);
                run<GGML_TYPE_Q4_0, float>(queue, D, kv, nh, reps);
            }
            std::puts("48/48 type/destination/shape combinations bit-exact; 144/144 comparisons PASS");
        }
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
