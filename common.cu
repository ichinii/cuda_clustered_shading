#pragma once

#define OPT_BVH

// #define USE_THRUST_SORT
// #ifdef USE_THRUST_SORT
// #include <thrust/sort.h>
// #endif

#include <bits/stdc++.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define grid_size 8u
#define tiles_count (grid_size * grid_size * grid_size)

using namespace glm;

__device__ struct Perspective {
    float fov, near, far;
};

__global__ struct Camera {
    vec3 eye;
    vec3 dir;
    uvec2 res;
    Perspective proj;
};

__global__ struct Light {
    vec3 p;
    float r;
};

__global__ struct KeyValue {
    int k;
    int v;
};

__global__ struct Aabb {
    vec3 back_left_bot;
    vec3 front_right_top;
};

__global__ struct Plane {
    vec3 n; // normal
    float o; // offset in direction of normal
};

__global__ struct Span {
    int begin;
    int count;
};

__global__ struct View {
    vec2 origin;
    vec2 look_at;
    float zoom;
    int lights_offset;
};

__device__ uvec3 tileIndexToCoord(int i) {
    return uvec3(i % grid_size, (i / grid_size) % grid_size, i / (grid_size * grid_size));
}

__host__ unsigned int tileCoordToIndex(uvec3 v) {
    return v.x + v.y * grid_size + v.z * grid_size * grid_size;
}

__device__ float n31(vec3 s) {
    return fract(9457.824 * sin(dot(vec3(385.291, 458.958, 941.950), s)));
}

__device__ float n21(vec2 s) {
    return fract(12095.283 * sin(dot(vec2(585.905, 821.895), s)));
}

__device__ vec3 transform_ndc(vec3 v, Perspective p) {
    auto s = -1.0f / (tan(p.fov/2.0f * pi<float>()/180.0f));
    auto x = v.x / v.z * s;
    auto y = v.y / v.z * s;
    auto z = (v.z+p.near)/(p.far-p.near)*2+1;
    return vec3(x, y, z);
}

__device__ vec3 transform_ndc_invert(vec3 v, Perspective p) {
    auto s = -1.0f / (tan(p.fov/2.0f * pi<float>()/180.0f));
    auto z = (v.z-1)/2*(p.far-p.near)-p.near;
    auto x = v.x * z / s;
    auto y = v.y * z / s;
    return vec3(x, y, z);
}
