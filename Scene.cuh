#ifndef SCENE_H
#define SCENE_H

#include "Camera.cuh"
#include "Point.cuh"
#include "Spectrum.cuh"

class Scene {
  private:
  int height;
  int width;

  /* HittableList hittables; */

  /* HittableList *focusableList; */

  // A box that includes all items in the scene.
  /* BVHNode* box; */

  Camera camera;

  Spectrum background;
 public:

  Scene();

 Scene(int w, int h, Camera c, Spectrum bg) : width(w), height(h), camera(c), background(bg) {};
};

#endif
