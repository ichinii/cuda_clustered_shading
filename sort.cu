#pragma once

#include "common.cu"

// static constexpr const size_t elements = (1<<20);
#define radix_threads 32 // step sync
#define radix_stepsync (radix_threads==32)
#if radix_stepsync
    #define __radix_syncwarp __syncwarp()
    #define __radix_syncthreads
#else
    #define __radix_syncwarp __syncthreads()
    #define __radix_syncthreads __syncthreads()
#endif
#define radix_blocks (elements/radix_threads)
#define merge_serial_elements 2
#define merge_desired_threads 128
#define merge_threads (std::min(elements/merge_serial_elements, merge_desired_threads))
#define merge_blocks ((elements/merge_serial_elements-1)/merge_threads+1)

template <typename R, typename T>
using SortPredicate = R(*)(T);

__device__ void scan(int *a) {
    int tid = threadIdx.x;
    for (int i = 1; i < radix_threads; i*=2) {
        int ai = i <= tid ? a[tid - i] : 0;
        __radix_syncthreads;
        a[tid] = a[tid] + ai;
        __radix_syncwarp;
    }
}

template <typename R, typename T>
__global__ void radix_sort(T *c, SortPredicate<R, T> p) {
    __shared__ int b[radix_threads];
    __shared__ T a[radix_threads];
    int tid = threadIdx.x;
    int ts = blockIdx.x * blockDim.x;
    a[tid] = c[tid + ts];
    __radix_syncthreads;
    for (int i = 0; i < 32; ++i) {
        int x = (1<<i) & p(a[tid]);
        b[tid] = x == 0 ? 1 : 0;
        __radix_syncwarp;
        scan(b);
        int d = x == 0 ? b[tid]-1 : b[radix_threads-1] - b[tid] + tid;
        T t = a[tid];
        __radix_syncthreads;
        a[d] = t;
        __radix_syncwarp;
    }
    c[tid + ts] = a[tid];
}

template <typename R, typename T>
__device__ int merge_path_partition(T *a, T *b, int d, int stride, SortPredicate<R, T> p) {
    int begin = max(0, d-stride);
    int end = min(d, stride);
    while (begin < end) {
        int mid = (begin + end) / 2;
        if (p(a[mid]) < p(b[d - mid - 1]))
            begin = mid + 1;
        else
            end = mid;
    }
    return begin;
}

template <typename R, typename T>
__device__ void serial_merge(T *a, int an, T *b, int bn, T *c, SortPredicate<R, T> p) {
    int x = 0;
    int y = 0;
    for (int i = 0; i < an+bn; ++i) {
        if (bn == y || x < an && p(a[x]) < p(b[y])) {
            c[i] = a[x++];
        } else {
            c[i] = b[y++];
        }
    }
}

template <typename R, typename T>
__device__ void parallel_merge(T *a, T *b, int stride, int serial, int tid, SortPredicate<R, T> p) {
    int d0 = serial * tid;
    int d1 = serial * (tid+1);
    int x0 = merge_path_partition(a, a+stride, d0, stride, p);
    int x1 = merge_path_partition(a, a+stride, d1, stride, p);
    int y0 = d0 - x0;
    int y1 = d1 - x1;
    // printf("%d (%d %d) (%d %d) (%d %d)\n", tid, d0, d1, x0, x1, y0, y1);
    serial_merge(a + x0, x1 - x0, a + y0 + stride, y1 - y0, b + d0, p);
}

template <typename R, typename T>
__global__ void merge_sort(T *a, T *b, int stride, int serial, SortPredicate<R, T> p) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int o = tid*serial/(stride*2)*stride*2;
    parallel_merge(a+o, b+o, stride, serial, tid%(stride/serial*2), p);
}

template <typename R, typename T>
__device__ R sort_predicate_generic(T a) {
    return R(a);
}

template <typename R, typename T>
void sort(T **a, int elements, SortPredicate<R, T> p = sort_predicate_generic<R, T>) {
    T *b;
    cudaMallocManaged(&b, elements * sizeof(T));

    radix_sort<<<radix_blocks, radix_threads>>>(*a, p);
    cudaDeviceSynchronize();

    for (int i = radix_threads; i < elements; i*=2) {
        merge_sort<<<merge_blocks, merge_threads>>>(*a, b, i, merge_serial_elements, p);
        cudaDeviceSynchronize();
        std::swap(*a, b);
    }

    cudaFree(b);
}

