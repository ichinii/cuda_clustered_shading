#include "sort.cu"
#include "draw.cu"
#include "assign_lights.cu"
#include "display.cu"

template <typename T>
void dump(T* a, int n, const char* label) {
    std::cout << "\t" << label << std::endl;
    for (int i = 0; i < n; ++i)
        std::cout << a[i] << ", " << std::endl;
}

std::ostream& operator<< (std::ostream& os, KeyValue a) {
    return os << "(k: " << a.k << ", v: " <<  a.v << ")";
}

std::ostream& operator<< (std::ostream& os, Light a) {
    return os << "(p: [" << a.p.x << ", " << a.p.y << ", " << a.p.z << "], r: " << a.r << ")";
}

std::ostream& operator<< (std::ostream& os, Span a) {
    return os << "(begin: " << a.begin << ", " << ", count: " << a.count << ")";
}

std::ostream& operator<< (std::ostream& os, Aabb a) {
    return os << "(back_left_bot: [" << a.back_left_bot.x << ", " << a.back_left_bot.y << ", " << a.back_left_bot.z << "], front_right_top: [" << a.front_right_top.x << ", " << a.front_right_top.y << ", " << a.front_right_top.z << "])";
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    int elements = 8*8*32;

    if (argc == 2) {
        elements = 1<<std::stoi(argv[1]);
        elements = max(32, elements);
    }
    std::cout << "elements: " << elements << std::endl;

    int w = 256;
    int b = (elements-1)/w+1;
    int b32 = (elements-1)/32+1;

    Camera cam {
        .eye = {0, 0, 5},
        .dir = {0, 0, -1},
        .res = {960, 960},
        .proj = Perspective {90.0f, 1.0f, 32.0f},
    };

    Light *lights;
    KeyValue *mortons;
    Aabb *aabbs;
    Aabb *aabbs32;
    glm::vec4 *image;
    int *indices;
    const int indices_capacity = 1024*1024;
    int *indices_size;
    Span *spans;
    cudaMallocManaged(&lights, elements * sizeof(Light));
    cudaMallocManaged(&mortons, elements * sizeof(KeyValue));
    cudaMallocManaged(&aabbs32, ((elements-1)/w+1) * sizeof(Aabb));
    cudaMallocManaged(&aabbs, elements * sizeof(Aabb));
    SortByKey h_sortp;
    cudaMemcpyFromSymbol(&h_sortp, d_sortp, sizeof(SortByKey));
    cudaMallocManaged(&image, (cam.res.x * cam.res.y) * sizeof(glm::vec4));
    cudaMallocManaged(&indices, indices_capacity * sizeof(int));
    cudaMallocManaged(&indices_size, sizeof(int));
    cudaMallocManaged(&spans, tiles_count * sizeof(Span));

    cudaDeviceSynchronize();
    init_lights<<<b, w>>>(lights, elements);

    cudaDeviceSynchronize();
    get_mortons<<<b, w>>>(lights, mortons, elements, cam.proj);

#ifdef OPT_BVH
    cudaDeviceSynchronize();
    sort(&mortons, elements, h_sortp);
#endif

    cudaDeviceSynchronize();
    get_aabbs<<<b, w>>>(mortons, lights, aabbs, elements, cam.proj);

#ifdef OPT_BVH
    cudaDeviceSynchronize();
    reduce_aabbs<<<b, w>>>(aabbs, aabbs32);
#endif

    cudaDeviceSynchronize();
    assign_lights<<<tiles_count, w>>>(mortons, aabbs32, aabbs, elements, spans, indices, indices_size, indices_capacity);

    cudaDeviceSynchronize();
    // dump(mortons, elements, "mortons");
    // dump(aabbs32, elements/32, "aabss32");
    // dump(aabbs, elements, "aabss");
    // dump(spans, tiles_count, "spans");

    auto update = [&] (uvec3 tile_coord, View view) {
        cudaDeviceSynchronize();
        draw(image, cam, spans, indices, mortons, lights, elements, tile_coord, view);

        cudaDeviceSynchronize();
        return image;
    };

    display(cam.res, update);

    cudaFree(lights);
    cudaFree(mortons);
    cudaFree(aabbs);
    cudaFree(aabbs32);
    cudaFree(image);
    cudaFree(indices);
    cudaFree(indices_size);
    cudaFree(spans);

    return 0;
}
