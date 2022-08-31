
#pragma once

#include "common.cu"
#include "sort.cu"

//TODO: return int is wrong. remove predicate thing
__device__ unsigned int sort_predicate_keyvalue(KeyValue a) {
    return a.k;
}
using SortByKey = SortPredicate<unsigned int, KeyValue>;
__device__ SortByKey d_sortp = sort_predicate_keyvalue;

__global__ void init_lights(Light *l, int n) {
    int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gtid < n) {
        const int g = 8;
        vec3 p = vec3(gtid % g, (gtid/g) % g, -gtid / (g*g));
        p = p / vec3(g-1);
        l[gtid] = Light {
            .p = (n33(p) * 2.0f - vec3(1, 1, 0)) * vec3(20, 20, -14),
            .r = pow(n31(p), 4.0f) * 2.5f + 0.2f,
        };
    }
};

// TODO: cull on the morton code
__global__ void get_mortons(Light *l, KeyValue *m, int n, Perspective proj) {
    int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gtid < n) {
        vec3 p0 = transform_ndc(l[gtid].p, proj) / 2.0f + 0.5f;
        uvec3 p = clamp(uvec3(p0 * 256.0f), 0u, 255u);
        unsigned int k = 0;
        for (int i = 0; i < 8; ++i) {
            k += ((p.x<<(i*4-i)) & (1<<(i*4)));
            k += ((p.y<<(i*4-i+1)) & (1<<(i*4+1)));
            k += ((p.z<<(i*4-i+2)) & (1<<(i*4+2)));
        }
        m[gtid].k = k;
        m[gtid].v = gtid;
    }
}

__global__ void get_aabbs(KeyValue *m, Light *l, Aabb *a, int n, Perspective p) {
    int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gtid < n) {
        int i = m[gtid].v;
        // TODO: maybe it can be useful not to take the min and max
        vec3 frontLeftBot = transform_ndc(l[i].p + vec3(-l[i].r, -l[i].r, l[i].r), p);
        vec3 front_right_top = transform_ndc(l[i].p + vec3(l[i].r), p);
        vec3 backCenter = transform_ndc(l[i].p + vec3(0, 0, -l[i].r), p);
        a[gtid] = Aabb {
            .back_left_bot = vec3(vec2(frontLeftBot), backCenter.z),
            .front_right_top = front_right_top,
        };
    }
}

__global__ void reduce_aabbs(Aabb *in, Aabb *out) {
    __shared__ Aabb a[256];
    int tid = threadIdx.x;
    int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    a[tid] = in[gtid];
    __syncthreads();
    for (int i = 1; i < 256; i*=2) {
        if (i <= tid) {
            a[tid].back_left_bot = min(a[tid].back_left_bot, a[tid - i].back_left_bot);
            a[tid].front_right_top = max(a[tid].front_right_top, a[tid - i].front_right_top);
        }
        __syncthreads();
    }
    if (tid == 255)
        out[blockIdx.x] = a[255];
}

__device__ bool intersect_aabb(Aabb a, Aabb b) {
    return a.back_left_bot.x < b.front_right_top.x
        && a.back_left_bot.y < b.front_right_top.y
        && a.back_left_bot.z < b.front_right_top.z
        && b.back_left_bot.x < a.front_right_top.x
        && b.back_left_bot.y < a.front_right_top.y
        && b.back_left_bot.z < a.front_right_top.z;
}

__global__ void assign_lights(KeyValue *m, Aabb *a256, Aabb *a, int n, Span *spans, int *outIndices, int *size, int capacity) {
    const int indices_capacity = 256;
#if opt
    const int groups_capacity = indices_capacity;
    __shared__ int groups[groups_capacity];
    __shared__ int groups_count;
#endif
    __shared__ int indices[indices_capacity];
    __shared__ int begin;
    __shared__ int count;

    int tid = threadIdx.x;
    int tile_index = blockIdx.x;

    if (tid == 0) {
#if opt
        groups_count = 0;
#endif
        count = 0;
    }

    __syncthreads();

    vec3 coord = vec3(tile_index_to_coord(tile_index));
    Aabb self = {
        .back_left_bot = coord / vec3(grid_size) * 2.0f - 1.0f,
        .front_right_top = (coord + 1.0f) / vec3(grid_size) * 2.0f - 1.0f,
    };

#if opt
    for (int i = tid; i < (n-1)/256+1; i+=256) {
        bool b = intersect_aabb(self, a256[i]);
        if (intersect_aabb(self, a256[i])) {
            int index = atomicAdd(&groups_count, 1);
            if (index < groups_capacity) {
                groups[index] = i;
            }
        }
    }

    __syncthreads();

    for (int j = 0; j < min(groups_count, groups_capacity); ++j) {
        int i = groups[j]*256 + tid;
#else
    for (int i = tid; i < n; i+=256) {
#endif
        if (intersect_aabb(self, a[i])) {
            int index = atomicAdd(&count, 1);
            if (index < indices_capacity) {
                indices[index] = m[i].v;
            }
        }
    }

    __syncthreads();

    if (tid == 0) {
        count = min(count, indices_capacity);
        begin = atomicAdd(size, count);
        count = min(begin + count, capacity) - begin;
        spans[tile_index].begin = begin;
        spans[tile_index].count = count;
    }

    __syncthreads();

    for (int i = tid; i < count; i+=256) {
        outIndices[begin + i] = indices[i];
    }
}
