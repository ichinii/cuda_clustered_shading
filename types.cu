#pragma once

#include "common.cu"

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
    vec3 backLeftBot;
    vec3 frontRightTop;
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
    float distance;
};
