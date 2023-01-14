#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "Hittable.cuh"

class HittableList : public Hittable {
public:
  __device__ HittableList(){};

  __device__ HittableList(Hittable **list, int size, int maxS) {
    objects = list;
    listSize = size;
    maxSize = maxS;
  };

  __device__ virtual bool hit(const Ray3 &r, float minTime, float maxTime,
                              HitRecord &hitRecord) const override;

  __device__ void add(Hittable *object);

  Hittable **objects;
  int listSize;
  int maxSize;
};

__device__ void HittableList::add(Hittable *object) {
  if (maxSize <= listSize)
    return;
  else {
    objects[listSize] = object;
    listSize++;
  }
}

__device__ bool HittableList::hit(const Ray3 &r, float minTime, float maxTime, HitRecord &hitRecord) const {
  HitRecord tempHitRecord;
  bool hitSomething = false;
  float timeOfClosestHit = maxTime;
  for (int i = 0; i < listSize; i++) {
    if (objects[i]->hit(r, minTime, timeOfClosestHit, tempHitRecord)) {
      hitSomething = true;
      timeOfClosestHit = tempHitRecord.timeOfHit;
      hitRecord = tempHitRecord;
    }
  }
  return hitSomething;
}

#endif
