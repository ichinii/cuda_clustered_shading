#pragma once

#include "common.cu"

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
        vec3 p = vec3(gtid % g, (gtid/g) % g, gtid / (g*g));
        p = p / vec3(g) * -2.0f + 1.0f;
        l[gtid] = Light {
            .p = p * 20.0f,
            .r = n31(p) + 0.5f,
        };
    }
};

__device__ vec3 transform_ndc(vec3 v, Perspective p) {
    auto s = -1.0f / (tan(p.fov/2.0f * pi<float>()/180.0f));
    auto x = v.x / v.z * s;
    auto y = v.y / v.z * s;
    auto z = (v.z+p.near)/(p.far-p.near)*2+1;
    return vec3(x, y, z);
}

// TODO: cull on the morton code
__global__ void get_mortons(Light *l, KeyValue *m, int n, Perspective proj) {
    int gtid = threadIdx.x + blockDim.x * blockIdx.x;
    if (gtid < n) {
        vec3 p0 = transform_ndc(l[gtid].p, proj) / 2.0f + 0.5f;
        uvec3 p = uvec3(p0 * 255.555f);
        // uvec3 p = uvec3(l[gtid].p);
        unsigned int r = min(int(l[gtid].r), 255);
        unsigned int k = 0;
        for (int i = 0; i < 8; ++i) {
            k += ((p.x<<(i*4-i)) & (1<<(i*4)));
            k += ((p.y<<(i*4-i+1)) & (1<<(i*4+1)));
            k += ((p.z<<(i*4-i+2)) & (1<<(i*4+2)));
            k += ((r<<(i*4-i+3)) & (1<<(i*4+3)));
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
        vec3 frontRightTop = transform_ndc(l[i].p + vec3(l[i].r), p);
        vec3 backCenter = transform_ndc(l[i].p + vec3(0, 0, -l[i].r), p);
        a[gtid] = Aabb {
            .backLeftBot = vec3(vec2(frontLeftBot), backCenter.z),
            .frontRightTop = frontRightTop,
        };
    }
}

__global__ void reduce_aabbs(Aabb *in, Aabb *out) {
    __shared__ Aabb a[32];
    int tid = threadIdx.x;
    int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    a[tid] = in[gtid];
    __syncwarp();
    for (int i = 1; i < 32; i*=2) {
        if (i <= tid) {
            a[tid].backLeftBot = min(a[tid].backLeftBot, a[tid - i].backLeftBot);
            a[tid].frontRightTop = max(a[tid].frontRightTop, a[tid - i].frontRightTop);
        }
        __syncwarp();
    }
    if (tid == 31)
        out[blockIdx.x] = a[31];
}

__device__ bool intersect_aabb(Aabb a, Aabb b) {
    return a.backLeftBot.x < b.frontRightTop.x
        && a.backLeftBot.y < b.frontRightTop.y
        && a.backLeftBot.z < b.frontRightTop.z
        && b.backLeftBot.x < a.frontRightTop.x
        && b.backLeftBot.y < a.frontRightTop.y
        && b.backLeftBot.z < a.frontRightTop.z;
}

__global__ void assign_lights(KeyValue *m, Aabb *a32, Aabb *a, int n, Span *spans, int *outIndices, int *size, int capacity) {
    const int indices_capacity = 256;
#ifdef OPT_BVH
    const int groups_capacity = indices_capacity/4;
    __shared__ int groups[groups_capacity];
    __shared__ int groups_count;
#endif
    __shared__ int indices[indices_capacity];
    __shared__ int begin;
    __shared__ int count;

    int tid = threadIdx.x;
    int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    int tile_index = blockIdx.x;

    if (tid == 0) {
#ifdef OPT_BVH
        groups_count = 0;
#endif
        count = 0;
    }

    __syncwarp();

    vec3 coord = vec3(tileIndexToCoord(tile_index));
    Aabb self = {
        .backLeftBot = coord / vec3(grid_size) * 2.0f - 1.0f,
        .frontRightTop = (coord + 1.0f) / vec3(grid_size) * 2.0f - 1.0f,
    };

#ifdef OPT_BVH
    for (int i = tid; i < (n-1)/32+1; i+=32) {
        bool b = intersect_aabb(self, a32[i]);
        if (intersect_aabb(self, a32[i])) {
            int index = atomicAdd(&groups_count, 1);
            if (index < groups_capacity) {
                groups[index] = i;
            }
        }
    }

    __syncwarp();

    for (int j = 0; j < min(groups_count, groups_capacity); ++j) {
        int i = groups[j]*32 + tid;
#else
    for (int i = tid; i < n; i+=32) {
#endif
        if (intersect_aabb(self, a[i])) {
            int index = atomicAdd(&count, 1);
            if (index < indices_capacity) {
                indices[index] = m[i].v;
            }
        }
    }

    __syncwarp();

    if (tid == 0) {
        count = min(count, indices_capacity);
        begin = atomicAdd(size, count);
        // if (begin != 0 || count != 0)
        //     printf("_ %d %d\n", begin, count);
        count = min(begin + count, capacity) - begin;
        spans[tile_index].begin = begin;
        spans[tile_index].count = count;
    }

    __syncwarp();

    for (int i = tid; i < count; i+=32) {
        outIndices[begin + i] = indices[i];
    }
}
