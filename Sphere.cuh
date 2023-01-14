#ifndef SPHERE_H
#define SPHERE_H

#include "Hittable.cuh"
#include <math.h>

class Sphere : public Hittable {  
 public:

  Point3 centre;
  float radius;
  
  __device__ Sphere() {};

  __device__ Sphere(Point3 c, float r) : centre(c), radius(r) {};

  __device__ virtual bool hit(const Ray3& r, float minTime, float maxTime, HitRecord& rec) const override;
};

__device__ bool Sphere::hit(const Ray3 &r, float minTime, float maxTime, HitRecord &rec) const {
  
  Vector3 differenceVector = r.origin() - centre;
  float a = r.direction().dot(r.direction());
  float halfB = differenceVector.dot(r.direction());
  float c = differenceVector.dot(differenceVector) - radius * radius;

  float discriminant = halfB * halfB - a * c;
  if (discriminant < 0) return false;
  float sqrtDiscriminant = sqrt(discriminant);

  // Check the nearest root that's still inside the time range
  float root = (-halfB - sqrtDiscriminant) / a;
  if (root < minTime || maxTime < root) {
    root = (-halfB + sqrtDiscriminant) / a;
    if (root < minTime || maxTime < root) {
      return false;
    }
  }

  rec.timeOfHit = root;
  rec.hitPoint = r.positionAtTime(rec.timeOfHit);
  
  // The normal always facing outwards. Doesn't change
  Vector3 outwardNormal = (rec.hitPoint - centre) / radius;
  rec.setFaceNormal(r, outwardNormal);
  return true;
}

#endif                   
