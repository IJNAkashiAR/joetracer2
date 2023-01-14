#ifndef CAMERA_H
#define CAMERA_H

#include "Point.cuh"
#include "Vector.cuh"

class Camera {
public:
  __device__ Camera() {
    aspectRatio = 16.0 / 9.0;
    viewportHeight = 2.0;
    viewportWidth = aspectRatio * viewportHeight;
    focalLength = 1.0;

    origin = Point3(0, 0, 0);
    horizontal = Vector3(viewportWidth, 0, 0);
    vertical = Vector3(0, viewportHeight, 0);
    lowerLeftCorner =
        origin - horizontal / 2 - vertical / 2 - Vector3(0, 0, focalLength);
  };

  __device__ Camera(float ar, float fl) {
    aspectRatio = ar;
    viewportHeight = 2.0;
    viewportWidth = aspectRatio * viewportHeight;
    focalLength = fl;

    origin = Point3(0, 0, 0);
    horizontal = Vector3(viewportWidth, 0, 0);
    vertical = Vector3(0, viewportHeight, 0);
    lowerLeftCorner =
        origin - horizontal / 2 - vertical / 2 - Vector3(0, 0, focalLength);
  };

  float aspectRatio;
  float viewportHeight;
  float viewportWidth;
  float focalLength;

  Point3 origin;
  Vector3 horizontal;
  Vector3 vertical;

  Vector3 lowerLeftCorner;
};

__global__ void createCameraPointer(Camera **c, float aspectRatio, float focalLength) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    *c = new Camera(aspectRatio, focalLength);
}

#endif
