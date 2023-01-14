#ifndef HITTABLE_H
#define HITTABLE_H

#include "Ray.cuh"
#include "Vector.cuh"

struct HitRecord {
  Point3 hitPoint;
  Vector3 normalAtHit;
  float timeOfHit;
  bool isFrontFacing;

  __device__ inline void setFaceNormal(const Ray3& r, const Vector3& outwardNormal) {
    isFrontFacing = r.direction().dot(outwardNormal) < 0;
    normalAtHit = isFrontFacing? outwardNormal : -outwardNormal;
  }
};

class Hittable {
 public:
  __device__ virtual bool hit(const Ray3& r, float minTime, float maxTime, HitRecord& hitRecord) const = 0;
};

#endif
