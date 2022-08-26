#include "sort.cu"
#include "draw.cu"
#include "assign_lights.cu"
#include "display.cu"

using Args = std::map<std::string, float>;

Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        auto s = std::string(argv[i]);

        auto d = s.find("=");
        if (d == std::string::npos)
            continue;

        auto key = s.substr(0, d);
        auto value = std::stof(s.substr(d+1));
        args[key] = value;
    }
    return args;
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    auto args = parse_args(argc, argv);
    args.try_emplace("lights", 10);
    args.try_emplace("fov", 90);
    args.try_emplace("far", 32);

    // number of lights
    int elements = 1 << static_cast<int>(args["lights"]);
    std::cout << "elements: " << elements << std::endl;

    // number of executed threads per block
    int w = 256;
    // number of executed blocks of threads
    int b = (elements-1)/w+1;
    // int b32 = (elements-1)/32+1;

    Camera cam {
        .eye = {0, 0, 5},
        .dir = {0, 0, -1},
        .res = {960, 960},
        .proj = Perspective {args["fov"], 1.0f, args["far"]},
    };

    // create resources on the video card
    Light *lights;
    KeyValue *mortons;
    Aabb *aabbs;
    Aabb *aabbs256;
    glm::vec4 *image;
    int *indices;
    const int indices_capacity = 1024*1024;
    int *indices_size;
    Span *spans;
    SortByKey h_sortp;
    cudaMemcpyFromSymbol(&h_sortp, d_sortp, sizeof(SortByKey));
    cudaMallocManaged(&lights, elements * sizeof(Light));
    cudaMallocManaged(&mortons, elements * sizeof(KeyValue));
    cudaMallocManaged(&aabbs256, b * sizeof(Aabb));
    cudaMallocManaged(&aabbs, elements * sizeof(Aabb));
    cudaMallocManaged(&image, (cam.res.x * cam.res.y) * sizeof(glm::vec4));
    cudaMallocManaged(&indices, indices_capacity * sizeof(int));
    cudaMallocManaged(&indices_size, sizeof(int));
    cudaMallocManaged(&spans, tiles_count * sizeof(Span));

    // *** begin clustered shading algorithm ***

    // initialize a bunch of light sources
    cudaDeviceSynchronize();
    init_lights<<<b, w>>>(lights, elements);

    // transform light sources into normalized device coordinates space
    // generate morton code for transformed position
    cudaDeviceSynchronize();
    get_mortons<<<b, w>>>(lights, mortons, elements, cam.proj);

#if opt
    // sort all light sources after the generated morton code
    cudaDeviceSynchronize();
    sort(&mortons, elements, h_sortp);
#endif

    // calculate an axis aligned bounding box for each light and
    // transform it into ndc space aswell
    cudaDeviceSynchronize();
    get_aabbs<<<b, w>>>(mortons, lights, aabbs, elements, cam.proj);

#if opt
    // create groups of 256 elements and create one AABB for each group
    cudaDeviceSynchronize();
    reduce_aabbs<<<b, w>>>(aabbs, aabbs256);
#endif

    // intersect tiles and light sources
    // store light sources for each tile
    cudaDeviceSynchronize();
    assign_lights<<<tiles_count, w>>>(mortons, aabbs256, aabbs, elements, spans, indices, indices_size, indices_capacity);

    // *** end clustered shading algorithm ***

    cudaDeviceSynchronize();

    // render an image using ray marching
    auto update = [&] (uvec3 tile_coord, View view) {
        cudaDeviceSynchronize();
        draw(image, cam, spans, indices, mortons, lights, elements, tile_coord, view);

        cudaDeviceSynchronize();
        return image;
    };

    // main loop. render and display
    display(cam.res, update);

    // clean up resources
    cudaFree(lights);
    cudaFree(mortons);
    cudaFree(aabbs);
    cudaFree(aabbs256);
    cudaFree(image);
    cudaFree(indices);
    cudaFree(indices_size);
    cudaFree(spans);

    return 0;
}
