#include "Camera.cuh"
#include "Hittable.cuh"
#include "HittableList.cuh"
#include "Ray.cuh"
#include "Sphere.cuh"
#include <eigen3/Eigen/src/Core/util/Constants.h>
#include <iostream>
#include <math.h>
#include "Constants.cuh"
#include <curand.h>

// workaround issue between gcc >= 4.7 and cuda 5.5
#if (defined __GNUC__) && (__GNUC__ > 4 || __GNUC_MINOR__ >= 7)
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128
#endif

// Taken from https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#define HEIGHT 800
#define WIDTH 1000

__device__ void rayColor(const Ray3 &r, const Hittable** world, Point3 &returnedCol) {
  HitRecord record;
  if ((*world)->hit(r, 0, FLOATINF, record)) {
    returnedCol = 0.5 * (record.normalAtHit + Point3(1, 1, 1));
    return;
  }
  Vector3 unitDirection = r.direction().normalized();
  float t = 0.5 * (unitDirection.y() + 1.0);
  returnedCol = ((1.0 - t) * Point3(1.0, 1.0, 1.0) + t * Point3(0.5, 0.7, 1.0));
}

__global__ void generateWorld(Hittable** dWorld, Hittable** dList) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *(dList) = new Sphere(Point3(0, 0, -1), 0.5);
    *(dList+1) = new Sphere(Point3(0, -100.5, -1), 100);
    *dWorld = new HittableList(dList, 2, 2);
  }
  return;
}

__global__ void render(Camera **c, int size, float **pixels, Hittable** world) {
  int indexX = threadIdx.x + blockIdx.x * blockDim.x;
  int indexY = threadIdx.y + blockIdx.y * blockDim.y;

  if (indexX >= WIDTH || indexY >= HEIGHT)
    return;

  float u = float(indexX) / (WIDTH - 1);
  float v = float(indexY) / (HEIGHT - 1);
  Point3 origin = (*c)->origin;
  Vector3 direction = (*c)->lowerLeftCorner + u * (*c)->horizontal +
                      v * (*c)->vertical - (*c)->origin;
  Ray3 ray = Ray3(origin, direction);
  float b = 0.25;

  Point3 colour;
  rayColor(ray, world, colour);

  pixels[indexY][indexX * 3] = (255.999 * colour.x());
  pixels[indexY][indexX * 3 + 1] = (255.999 * colour.y());
  pixels[indexY][indexX * 3 + 2] = (255.999 * colour.z());
}

int main(int argc, char **argv) {

  int imageSize = HEIGHT * WIDTH;
  float **pixels;

  Hittable **dWorld;
  Hittable **dList;
  cudaMallocManaged(&dWorld, sizeof(Hittable *));
  cudaMallocManaged(&dList, 2 * sizeof(Hittable *));

  generateWorld<<<1, 1>>>(dWorld, dList);
  
  int threadX = 16;
  int threadY = 16;

  dim3 threads(threadX, threadY);
  dim3 blocks(WIDTH / threadX + 1, HEIGHT / threadY + 1);

  // Allocated unified memory for pixels
  cudaMallocManaged(&pixels, HEIGHT * sizeof(float *));
  for (int i = 0; i < HEIGHT; i++)
    cudaMallocManaged(pixels + i, WIDTH * 3 * sizeof(float));

  Camera **c;
  cudaMallocManaged(&c, sizeof(Camera));
  createCameraPointer<<<1, 1>>>(c, (float)WIDTH / HEIGHT, 1.0);
  
  // Run kernel on 1M elements on the GPU
  render<<<blocks, threads>>>(c, imageSize, pixels, dWorld);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  std::cout << "P3\n" << WIDTH << ' ' << HEIGHT << "\n255\n";

  for (int i = HEIGHT - 1; i >= 0; --i) {
    for (int j = 0; j < WIDTH; ++j) {
      std::cout << (int)pixels[i][j * 3] << " " << (int)pixels[i][j * 3 + 1] << " " << (int)pixels[i][j * 3 + 2] << '\n';
    }
  }
  // Free memory
  cudaFree(pixels);
  return 0;
}