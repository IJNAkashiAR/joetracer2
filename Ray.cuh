#ifndef RAY_H
#define RAY_H

#include "Point.cuh"
#include "Vector.cuh"

class Ray3 {
 private:
  Point3 o;
  Vector3 d;  

  public:
  __device__ Ray3();

  __device__ Ray3(const Point3& origin, const Vector3& direction) : o(origin), d(direction) {}

  __device__ Point3 positionAtTime(float t) const {
    return o + t * d;
  }
  
  __device__ Point3 origin() const {
    return o;
  }

  __device__ Vector3 direction() const {
    return d;
  }


};

#endif
