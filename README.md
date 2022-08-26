# clustered shading implemented in cuda
r&d project

### dependencies
- Video card by NVIDIA with compute capability >= ?
- cuda
- OpenGL 4.0
- GLEW
- glfw
- glm

### build (linux)
```
    # use cuda compiler, link libraries
nvcc -lGL -lGLU -lGLEW -lglfw main.cu
```
options:
```
    # cluster grid extent. default: 8
-Dgrid_size=8
    # enable bounding volume optimization. default: 1
-Dopt=1
```

### run
```
./a.out
```
arguments:
```
    # populates the scene with 2^ARG number of light sources.\
    # default: 10 => 2^11 = 2048 light sources
lights=10
    # camera field of view
fov=90
    # camera far plane
far=32
```

### controls
```
action              | input

camera position     | arrow keys
camera direction    | mouse
zoom                | mouse wheel
toggle hide objects | 1, 2, 3, 4
move tile xy        | w, a, s, d
move tile z         | q, e
iterate lights      | n, m
```
