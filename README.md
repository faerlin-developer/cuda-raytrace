<div align="center">
    <h3 align="center">Basic Ray Tracing in CUDA</h3>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#basic-ray-tracing">Basic Ray Tracing</a></li>
    <li><a href="#build-and-run">Build and Run</a></li>
  </ol>
</details>

## About The Project

<br/>
<div align="center">
    <img src="output.png" width="512">
    <br/>
    <figcaption>Figure 1. Output of program.</figcaption>
</div>
<br/>

We implement a basic ray tracer in C++ using CUDA. Ray tracing produces a 2D image of a scene consisting of 3D objects.
Our basic ray tracer processes a scene containing 3D spheres and generates a 2D image as viewed from a camera positioned
along the z-axis.

<!--

Our basic ray tracer
will only support scenes o spheres, and the camera is restricted to the zaxis,
acing the origin. Moreover, we will not support any lighting o the scene to avoid
the complications o secondary rays. Instead o computing lighting eects, we will
simply assign each sphere a color and then shade them with some precomputed
unction i they are visible.
So, what will the ray tracer do? It will re a ray rom each pixel and keep track o
which rays hit which spheres. It will also track the depth o each o these hits. In
the case where a ray passes through multiple spheres, only the sphere closest
to the camera can be seen. In essence, our “ray tracer” is not doing much more
than hiding suraces that cannot be seen by the camera.
We will model our spheres with a data structure that stores the sphere’s center
coordinate o (x, y, z), its radius, and its color o (r, b, g).

-->


