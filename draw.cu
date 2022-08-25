#pragma once

#include "common.cu"

#define max_dist 200.0f
#define min_dist 0.01f
#define max_it 1000
#define planes_size (grid_size + 1u)
#define planes_count (planes_size + planes_size + planes_size)

__device__ struct Scene {
    Span *spans;
    int *indices;
    KeyValue *mortons;
    Light *l;
    Plane *f;
    int lights_count;
    unsigned int tile_index;
    View v;
    Camera cam;
};

__device__ float sdmin(float a, float b) {
    return abs(a) < abs(b) ? a : b;
}

__device__ float sdf_plane(vec3 p, vec3 n) {
    return dot(p, n);
}

__device__ float sdf_sphere(vec3 p) {
    return length(p);
}

// thanks Inigo Quilez (https://iquilezles.org/articles/distfunctions/)
__device__ float sdf_capsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a;
    vec3 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0f, 1.0f);
    return length(pa - ba * h) - r;
}

__device__ float sdf_tile_frustum(vec3 p, Plane *f, uvec3 coord) {
    float d = -max_dist;
    for (int i = 0; i < 3; ++i) {
        int j = coord[i] + planes_size * i;
        d = max(d, sdf_plane(p + f[j].n * f[j].o, -f[j].n));
        d = max(d, sdf_plane(p + f[j+1].n * f[j+1].o, f[j+1].n));
    }
    return d;
}

__device__ float sdf_tile_planes(vec3 p, Plane *f, uvec3 coord) {
    float d = max_dist;
    for (int i = 0; i < 3; ++i) {
        int j = coord[i] + planes_size * i;
        d = min(d, abs(sdf_plane(p + f[j].n * f[j].o, -f[j].n)));
        d = min(d, abs(sdf_plane(p + f[j+1].n * f[j+1].o, f[j+1].n)));
    }
    return d;
}

__device__ float sdf_frustum(vec3 p, Plane *f) {
    float d = -max_dist;
    for (int i = 0; i < 3; ++i) {
        int b = i * planes_size;
        int e = (i+1) * planes_size-1;
        d = max(d, sdf_plane(p + f[b].n * f[b].o, -f[b].n));
        d = max(d, sdf_plane(p + f[e].n * f[e].o, f[e].n));
    }
    return d;
}

__device__ float sdf_lights(vec3 p, KeyValue *mortons, Light *l, int b, int e) {
    float d = max_dist;
    for (int i = b; i < e; ++i) {
        d = sdmin(d, sdf_sphere(l[mortons[i].v].p - p) - l[mortons[i].v].r);
    }
    return d;
}

__device__ float sdf_tile_lights(vec3 p, int* indices, int n, Light *lights) {
    float d = max_dist;
    for (int i = 0; i < n; ++i) {
        Light l = lights[indices[i]];
        d = sdmin(d, sdf_sphere(l.p - p) - l.r);
    }
    return d;
}

__device__ float sdf_scene_frustum(Scene s, vec3 p) {
    return sdf_frustum(p, s.f);
}

__device__ float sdf_scene_tile_lights(Scene s, vec3 p) {
    return sdf_tile_lights(p, s.indices + s.spans[s.tile_index].begin, s.spans[s.tile_index].count, s.l);
}

__device__ float sdf_scene_tile_frustum(Scene s, vec3 p) {
    return sdf_tile_frustum(p, s.f, tileIndexToCoord(s.tile_index));
}

__device__ float sdf_scene_lights(Scene s, vec3 p) {
    const int stride = 32;
    int i = abs(s.v.lights_offset) % ((s.lights_count-1)/stride+1);
    return sdf_lights(p, s.mortons, s.l, i * stride, min((i+1) * stride, s.lights_count));
}

__device__ struct Ray {
    bool hit;
    float l;
    float sgn;
};

using SdfScene = float(*)(Scene, vec3);

__device__ Ray march(Scene s, SdfScene sdf, vec3 ro, vec3 rd) {
    auto lo = 0.0f;
    for (int i = 0; i < max_it && lo < max_dist; ++i) {
        float sl = sdf(s, ro);
        float l = abs(sl);
        ro += l * rd;
        lo += l;

        if (l < min_dist)
            return Ray {true, lo, sign(sl)};
    }
    return Ray {false};
}

__device__ vec3 normal(Scene s, SdfScene sdf, vec3 p) {
    float l = sdf(s, p);
    float o = min_dist * 0.5f;
    return normalize(
        l - vec3(
            sdf(s, p - vec3(o, 0, 0)),
            sdf(s, p - vec3(0, o, 0)),
            sdf(s, p - vec3(0, 0, o))
        )
    );
}

__device__ mat3 look_at(vec3 d) {
    vec3 r = normalize(cross(d, vec3(0, 1, 0)));
    vec3 u = normalize(cross(r, d));
    return mat3(r, u, d);
}

__device__ float trace(Scene s, SdfScene sdf, vec3 ro, vec3 rd) {
    float c = 0.0f;
    for (float i = 0.0f; i < 8.0f; ++i) {

        Ray r = march(s, sdf, ro, rd);
        if (!r.hit)
            break;

        vec3 p = ro + rd * r.l;
        vec3 n = normal(s, sdf, p);
        ro = p + rd * min_dist * 10.0f;
        float front = max(0.0f, dot(rd, -n));
        if (r.l < min_dist * 1.0f) {
            i -= 0.5f;
        } else if (0.0f < front) {
            vec3 directional_light = normalize(vec3(1, 3, 2));
            float a = dot(n, directional_light);
            c += pow(1.0f-front, 5.0f)
                + 0.8f * max(0.0f, a)
                + 0.8f * pow(max(0.0f, -a), 5.0f)
                + 0.2f;
        }
    }
    return c;
}

__global__ void get_image(vec4 *c, Scene s) {
    int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    vec2 uv = vec2(gtid % s.cam.res.x, gtid / s.cam.res.x) / vec2(s.cam.res) * 2.0f - 1.0f;

    // vec3 ro = cam.eye;
    // vec3 rd = look_at(normalize(cam.dir)) * normalize(vec3(uv, 1.0));
    vec3 ro = s.v.zoom * vec3(
        sin(s.v.origin.x) * cos(s.v.origin.y),
        sin(s.v.origin.y),
        cos(s.v.origin.x) * cos(s.v.origin.y)
    );
    vec3 rd = look_at(normalize(-ro)) * normalize(vec3(uv, 1.0));
    vec3 center = vec3(s.v.look_at.x, 0, s.v.look_at.y);
    ro += center;

    c[gtid] = vec4((1.0f/255.0f) * vec3(n21(uv)), 1);
    if (s.v.visible_flags & View::VisibleFlag::frustum)
        c[gtid].b += 0.8f * trace(s, &sdf_scene_frustum, ro, rd);
    if (s.v.visible_flags & View::VisibleFlag::tile_frustum)
        c[gtid].g += 0.8f * trace(s, &sdf_scene_tile_frustum, ro, rd);
    if (s.v.visible_flags & View::VisibleFlag::tile_lights)
        c[gtid].r += 0.3f * trace(s, &sdf_scene_tile_lights, ro, rd);
    if (s.v.visible_flags & View::VisibleFlag::lights)
        c[gtid].b += 0.3f * trace(s, &sdf_scene_lights, ro, rd);

    // draw look_at position
    c[gtid].g += trace(s, [] (Scene s, vec3 p) -> float {
        return length(vec3(s.v.look_at.x, 0, s.v.look_at.y) - p) - 0.1f;
    }, ro, rd);

    // gamma correction
    c[gtid] = vec4(pow(vec3(c[gtid]), vec3(1.0 / 2.2)), c[gtid].a);
}

void frustumPlanes(Plane *planes, Camera cam)
{
    const auto up = vec3(0, 1, 0);
    const auto invProj = inverse(perspective(radians(cam.proj.fov), 1.0f, cam.proj.near, cam.proj.far));
    auto verticalPlanes = planes;
    auto horizontalPlanes = verticalPlanes + planes_size;
    auto parallelPlanes = horizontalPlanes + planes_size;

    for (auto i = 0ul; i < planes_size; ++i) {
        auto x = -1.0f + 2.0f * i / grid_size;
        auto forward = normalize(vec3(invProj * vec4(x, 0, 0, 1)));
        auto normal = normalize(cross(forward, up));
        verticalPlanes[i].n = normal;
        verticalPlanes[i].o = 0.0f;
    }

    for (auto i = 0ul; i < planes_size; ++i) {
        auto y = -1.0f + 2.0f * i / grid_size;
        auto forward = normalize(vec3(invProj * vec4(0, y, 0, 1)));
        auto right = normalize(cross(forward, up));
        auto normal = normalize(cross(right, forward));
        horizontalPlanes[i].n = normal;
        horizontalPlanes[i].o = 0.0f;
    }

    for (auto i = 0ul; i < planes_size; ++i) {
        parallelPlanes[i].n = vec3(0, 0, 1);
        parallelPlanes[i].o = cam.proj.near + (cam.proj.far-cam.proj.near) * float(grid_size-i) / grid_size;
    }
}

void draw(vec4 *image, Camera cam, Span *spans, int* indices, KeyValue* mortons, Light *lights, int n, uvec3 tile_coord, View view) {
    Plane *frustum;
    cudaMallocManaged(&frustum, planes_count * sizeof(Plane));
    cudaDeviceSynchronize();
    frustumPlanes(frustum, cam);
    cudaDeviceSynchronize();

    int w = 256;
    int b = (cam.res.x*cam.res.y-1)/w+1;
    unsigned int tile_index = tileCoordToIndex(tile_coord);
    get_image<<<b, w>>>(image, Scene {spans, indices, mortons, lights, frustum, n, tile_index, view, cam});
}
