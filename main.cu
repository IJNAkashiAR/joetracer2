#include "Camera.h"
#include "Ray.h"
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <iostream>
#include <math.h>

// workaround issue between gcc >= 4.7 and cuda 5.5
#if (defined __GNUC__) && (__GNUC__ > 4 || __GNUC_MINOR__ >= 7)
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128
#endif

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#define HEIGHT 800
#define WIDTH 1000


__device__ void rayColor(const Ray3& r, Point3& returnedCol) {
  Vector3 unitDirection = r.direction().normalized();
  float t = 0.5*(unitDirection.y() + 1.0);
  returnedCol = ((1.0-t)*Point3(1.0, 1.0, 1.0) + t*Point3(0.5, 0.7, 1.0));
}

__global__ void createCamera(Camera** c) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    *c = new Camera();
}

__global__ void render(Camera** c, int size, float **pixels) {
  int indexX = threadIdx.x;
  int indexY = threadIdx.y;
  int strideX = blockDim.x;
  int strideY = blockDim.y;

  for (int i = (HEIGHT-strideY) + indexY - 1 ; i >= 0; i -= strideY) {
    for (int j = indexX; j < WIDTH; j += strideX) {
      float u = float(j) / (WIDTH - 1);
      float v = float(i) / (HEIGHT - 1);
      Point3 origin = (*c)->origin;
      Vector3 direction = (*c)->lowerLeftCorner + u*(*c)->horizontal + v*(*c)->vertical - (*c)->origin;
      Ray3 ray = Ray3(origin, direction);
      float b = 0.25;

      Point3 colour;
      rayColor(ray, colour);
      
      pixels[i][j * 3] = (255.999 * colour.x());
      pixels[i][j * 3 + 1] = (255.999 * colour.y());
      pixels[i][j * 3 + 2] = (255.999 * colour.z());
    }
  }
}

int main(void) {
  int imageSize = HEIGHT * WIDTH;
  float *raw;     // = new float[HEIGHT * WIDTH * 3];
  float **pixels; // = new int[HEIGHT * WIDTH * 3];

  cudaMallocManaged(&pixels, HEIGHT * sizeof(float *));
  // Allocate Unified Memory – accessible from CPU or GPU
  for (int i = 0; i < HEIGHT; i++)
    cudaMallocManaged(pixels+i, WIDTH * 3 * sizeof(float));

  Camera** c;
  cudaMallocManaged((void**) &c, sizeof(Camera));
  createCamera<<<1, 1>>>(c);


  
  // Run kernel on 1M elements on the GPU
  render<<<dim3(1, 1), dim3(16, 16)>>>(c, imageSize, pixels);

  // Wait for GPU to finish before accessing on hosty
  cudaDeviceSynchronize();

  std::cout << "P3\n" << WIDTH << ' ' << HEIGHT << "\n255\n";

  for (int i = HEIGHT - 1; i >= 0; --i) {
    for (int j = 0; j < WIDTH; ++j) {
      // std::cerr << i << " " << j << '\n';
      std::cout << (int)pixels[i][j * 3] << " " << (int)pixels[i][j * 3 + 1] << " "
                << (int)pixels[i][j * 3 + 2] << '\n';
    }
  }
    // Free memory
    cudaFree(pixels);
    return 0;
}