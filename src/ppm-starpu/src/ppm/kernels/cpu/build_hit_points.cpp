#include "ppm/kernels/build_hit_points.h"

#include <stdio.h>

_extern_c_ void k_cpu_build_hit_points(void* buffers[], void* args_orig) {

  args_build_hit_points* args = (args_build_hit_points*) args_orig;

  const unsigned width  = args->width;
  const unsigned height = args->height;
  const unsigned spp    = args->spp;
  const unsigned hit_points = args->hit_points;

  printf("k_cpu_build_hit_points %d %d %d %d\n", width, height, spp, hit_points);

  EyePath* todo_eye_paths = new EyePath[hit_points];
}


#ifdef ASD
void Worker::BuildHitPoints(uint /*iteration*/) {

  // const unsigned int width = engine->width;
  // const unsigned int height = engine->height;
  // const unsigned int superSampling = engine->superSampling;

  // const unsigned int hitPointTotal = engine->hitPointTotal;

  //Seed* EyePathsSeeds = new Seed[hitPointTotal];

  EyePath* todoEyePaths = new EyePath[hitPointTotal];

  memset(hitPointsStaticInfo_iterationCopy, 0, sizeof(HitPointStaticInfo) * engine->hitPointTotal);

#ifndef USE_SPPM
  memset(hitPoints_iterationCopy, 0, sizeof(HitPoint) * engine->hitPointTotal);
#endif

  unsigned int hitPointsIndex = 0;

  // Generate eye rays
  //std::cerr << "Building eye paths rays with " << superSampling << "x"
  //    << superSampling << " super-sampling:" << std::endl;
  //std::cerr << "  0/" << height << std::endl;

//  double lastPrintTime = WallClockTime();
  const float invSuperSampling = 1.f / superSampling;

  for (unsigned int y = 0; y < height; ++y) {

//    if (WallClockTime() - lastPrintTime > 2.0) {
//      std::cerr << "  " << y << "/" << height << std::endl;
//      lastPrintTime = WallClockTime();
//    }

    //for all hitpoints
    for (unsigned int x = 0; x < width; ++x) {
      for (unsigned int sy = 0; sy < superSampling; ++sy) {
        for (unsigned int sx = 0; sx < superSampling; ++sx) {

          EyePath *eyePath = &todoEyePaths[hitPointsIndex];

          //EyePathsSeeds[hitPointsIndex] = mwc(hitPointsIndex + deviceID);


          eyePath->scrX = x + (sx + getFloatRNG(seedBuffer[hitPointsIndex]))
              * invSuperSampling - 0.5f;

          eyePath->scrY = y + (sy + getFloatRNG(seedBuffer[hitPointsIndex]))
              * invSuperSampling - 0.5f;

          float u0 = getFloatRNG(seedBuffer[hitPointsIndex]);
          float u1 = getFloatRNG(seedBuffer[hitPointsIndex]);
          float u2 = getFloatRNG(seedBuffer[hitPointsIndex]);

          //            scene->camera->GenerateRay(eyePath->scrX,
          //                eyePath->scrY, width, height, &eyePath->ray,
          //                u0, u1, u2);


          ss->GenerateRay(eyePath->scrX, eyePath->scrY, width, height, &eyePath->ray, u0,
              u1, u2, &ss->camera);

          eyePath->depth = 0;
          eyePath->throughput = Spectrum(1.f, 1.f, 1.f);

          eyePath->done = false;
          eyePath->splat = false;
          eyePath->sampleIndex = hitPointsIndex;

          hitPointsIndex++;

        }
      }
    }
  }

  // Iterate through all eye paths
  //std::cerr << "Building eye paths hit points: " << std::endl;
//  lastPrintTime = WallClockTime();
  // Note: (todoEyePaths.size() > 0) is extremly slow to execute


  uint todoEyePathCount = hitPointTotal;
  uint chunk_counter = 0;

  //std::cerr << "  " << todoEyePathCount / 1000 << "k eye paths left"
  //    << std::endl;

  uint* eyePathIndexes = new uint[getRaybufferSize()];

  while (todoEyePathCount > 0) {

//    if (WallClockTime() - lastPrintTime > 2.0) {
//      std::cerr << "  " << todoEyePathCount / 1000 << "k eye paths left" << std::endl;
//      lastPrintTime = WallClockTime();
//    }

    //std::vector<EyePath *>::iterator todoEyePathsIterator =
    //    todoEyePaths.begin() + roundPointer;

    //transversing in chunks
    uint start = (chunk_counter / getRaybufferSize()) * getRaybufferSize();

    uint end;
    if (hitPointTotal - start < getRaybufferSize())
      end = hitPointTotal;
    else
      end = start + getRaybufferSize();

    for (uint i = start; i < end; i++) {

      EyePath *eyePath = &todoEyePaths[i];

      // Check if we reached the max path depth
      if (!eyePath->done && eyePath->depth > MAX_EYE_PATH_DEPTH) {

        // Add an hit point
        //HitPointInfo &hp = *(engine->GetHitPointInfo(
        //    eyePath->pixelIndex));
        HitPointStaticInfo &hp = hitPointsStaticInfo_iterationCopy[eyePath->sampleIndex];

        hp.type = CONSTANT_COLOR;
        hp.scrX = eyePath->scrX;
        hp.scrY = eyePath->scrY;
        hp.throughput = Spectrum();

        //ihp.accumPhotonCount = 0;
        //ihp.accumReflectedFlux = Spectrum();

        //ihp.photonCount = 0;
        //hp.reflectedFlux = Spectrum();
        eyePath->done = true;

      } else if (!eyePath->done) {
        eyePath->depth++;

        uint p = RaybufferAddRay(eyePath->ray);

        eyePathIndexes[p] = i;
      }

      if (eyePath->done && !eyePath->splat) {
        --todoEyePathCount;
        chunk_counter++;
        eyePath->splat = true;
      }

      //if (rayBuffer->IsFull())
      //  break;

    }

    if (getRayBufferRayCount() > 0) {

      IntersectRayBuffer();

      //printf("%d\n",rayBuffer->GetRayCount());
      AdvanceEyePaths(&todoEyePaths[0], eyePathIndexes);

      resetRayBuffer();
    }
  }

  delete[] todoEyePaths;
  delete[] eyePathIndexes;


}
#endif
