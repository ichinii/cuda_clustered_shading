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



using namespace glm;



#define grid_size 32u
#define tiles_count (grid_size * grid_size * grid_size)

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
