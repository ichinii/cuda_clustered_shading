# clustered shading implemented in cuda
r&d project

### dependencies
Video card by NVidia with compute capability >= ?
cuda
OpenGL 4.0
GLEW
glfw
glm

### build (linux)
```
nvcc -lGL -lGLU -lGLEW -lglfw main.cu
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
