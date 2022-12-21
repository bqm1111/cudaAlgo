# GPU-path-tracing-tutorial-2
Basic CUDA path tracer with triangle mesh support (based on CUDA raytracer from http://cg.alexandra.dk/?p=278)
Sam Lapere, 2015

More details at https://raytracey.blogspot.com/2015/12/gpu-path-tracing-tutorial-2-interactive.html

In order to keep the code to a minimum, there are lots of hardcoded values at the moment. The comments should clarify most of what's happening but let me know if something isn't clear. 

The code probably contains some bugs as I haven't had much time to do many testing. It will probably be revised for the next tutorial.

The executable needs glew32.dll and glut32.dll to run and the triangle meshes (bunny and teapot) should be stored in folder named "data" that resides in the project folder.

Screenshot (path tracing a simple triangle mesh):

![Image description](https://github.com/straaljager/GPU-path-tracing-tutorial-2/blob/master/bunny_glass_corn.png)

For more screenshots produced with this code, see http://raytracey.blogspot.co.nz

Stanford Bunny mesh from https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj
Berkeley teapot mesh from http://inst.eecs.berkeley.edu/~cs184/sp09/assignments/teapot.obj

