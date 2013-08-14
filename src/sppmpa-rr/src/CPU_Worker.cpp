/*
 * CUDA_Worker.cpp
 *
 *  Created on: Oct 31, 2012
 *      Author: rr
 */

#include "CPU_Worker.h"
#include "omp.h"

extern Config* config;

CPU_Worker::~CPU_Worker() {

}

void CPU_Worker::AdvanceEyePaths(RayBuffer *rayBuffer, EyePath* todoEyePaths, uint* eyePathIndexes) {

#ifndef __DEBUG
  omp_set_num_threads(config->max_threads);
#pragma omp parallel for schedule(guided)
#endif
  for (uint i = 0; i < rayBuffer->GetRayCount(); i++) {

    EyePath *eyePath = &todoEyePaths[eyePathIndexes[i]];

    const RayHit *rayHit = &rayBuffer->GetHitBuffer()[i];

    if (rayHit->Miss()) {
      // Add an hit point

      HitPointPositionInfo* hp = GetHitPointInfo(eyePath->sampleIndex);

      hp->type = CONSTANT_COLOR;
      hp->scrX = eyePath->scrX;
      hp->scrY = eyePath->scrY;

      if (ss->infiniteLight || ss->sunLight || ss->skyLight) {

        if (ss->infiniteLight)
          ss->InfiniteLight_Le(&hp->throughput, &eyePath->ray.d, ss->infiniteLight,
              ss->infiniteLightMap);
        if (ss->sunLight)
          ss->SunLight_Le(&hp->throughput, &eyePath->ray.d, ss->sunLight);
        if (ss->skyLight)
          ss->SkyLight_Le(&hp->throughput, &eyePath->ray.d, ss->skyLight);

        hp->throughput *= eyePath->throughput;
      } else
        hp->throughput = Spectrum();

      eyePath->done = true;

    } else {

      // Something was hit
      Point hitPoint;
      Spectrum surfaceColor;
      Normal N, shadeN;

      if (engine->GetHitPointInformation(ss, &eyePath->ray, rayHit, hitPoint, surfaceColor,
          N, shadeN))
        continue;

      // Get the material
      const unsigned int currentTriangleIndex = rayHit->index;
      const unsigned int currentMeshIndex = ss->meshIDs[currentTriangleIndex];

      const uint materialIndex = ss->meshMats[currentMeshIndex];

      POINTERFREESCENE::Material *hitPointMat = &ss->materials[materialIndex];

      uint matType = hitPointMat->type;

      if (matType == MAT_AREALIGHT) {

        // Add an hit point
        HitPointPositionInfo* hp = GetHitPointInfo(eyePath->sampleIndex);

        hp->type = CONSTANT_COLOR;
        hp->scrX = eyePath->scrX;
        hp->scrY = eyePath->scrY;

        Vector md = -eyePath->ray.d;
        ss->AreaLight_Le(&hitPointMat->param.areaLight, &md, &N, &hp->throughput);
        hp->throughput *= eyePath->throughput;

        // Free the eye path
        eyePath->done = true;

      } else {

        Vector wo = -eyePath->ray.d;
        float materialPdf;

        Vector wi;
        bool specularMaterial = true;
        float u0 = getFloatRNG(seedBuffer[eyePath->sampleIndex]);
        float u1 = getFloatRNG(seedBuffer[eyePath->sampleIndex]);
        float u2 = getFloatRNG(seedBuffer[eyePath->sampleIndex]);
        Spectrum f;

        switch (matType) {

        case MAT_MATTE:
          ss->Matte_Sample_f(&hitPointMat->param.matte, &wo, &wi, &materialPdf, &f,
              &shadeN, u0, u1, &specularMaterial);
          f *= surfaceColor;
          break;

        case MAT_MIRROR:
          ss->Mirror_Sample_f(&hitPointMat->param.mirror, &wo, &wi, &materialPdf, &f,
              &shadeN, &specularMaterial);
          f *= surfaceColor;
          break;

        case MAT_GLASS:
          ss->Glass_Sample_f(&hitPointMat->param.glass, &wo, &wi, &materialPdf, &f, &N,
              &shadeN, u0, &specularMaterial);
          f *= surfaceColor;

          break;

        case MAT_MATTEMIRROR:
          ss->MatteMirror_Sample_f(&hitPointMat->param.matteMirror, &wo, &wi,
              &materialPdf, &f, &shadeN, u0, u1, u2, &specularMaterial);
          f *= surfaceColor;

          break;

        case MAT_METAL:
          ss->Metal_Sample_f(&hitPointMat->param.metal, &wo, &wi, &materialPdf, &f,
              &shadeN, u0, u1, &specularMaterial);
          f *= surfaceColor;

          break;

        case MAT_MATTEMETAL:
          ss->MatteMetal_Sample_f(&hitPointMat->param.matteMetal, &wo, &wi, &materialPdf,
              &f, &shadeN, u0, u1, u2, &specularMaterial);
          f *= surfaceColor;

          break;

        case MAT_ALLOY:
          ss->Alloy_Sample_f(&hitPointMat->param.alloy, &wo, &wi, &materialPdf, &f,
              &shadeN, u0, u1, u2, &specularMaterial);
          f *= surfaceColor;

          break;

        case MAT_ARCHGLASS:
          ss->ArchGlass_Sample_f(&hitPointMat->param.archGlass, &wo, &wi, &materialPdf,
              &f, &N, &shadeN, u0, &specularMaterial);
          f *= surfaceColor;

          break;

        case MAT_NULL:
          wi = eyePath->ray.d;
          specularMaterial = 1;
          materialPdf = 1.f;

          // I have also to restore the original throughput
          //throughput = prevThroughput;
          break;

        default:
          // Huston, we have a problem...
          specularMaterial = 1;
          materialPdf = 0.f;
          break;

        }
        if ((materialPdf <= 0.f) || f.Black()) {

          // Add an hit point
          HitPointPositionInfo* hp = GetHitPointInfo(eyePath->sampleIndex);
          hp->type = CONSTANT_COLOR;
          hp->scrX = eyePath->scrX;
          hp->scrY = eyePath->scrY;
          hp->throughput = Spectrum();

          eyePath->done = true;
        } else if (specularMaterial || (!hitPointMat->difuse)) {

          eyePath->throughput *= f / materialPdf;
          eyePath->ray = Ray(hitPoint, wi);
        } else {
          // Add an hit point
          HitPointPositionInfo* hp = GetHitPointInfo(eyePath->sampleIndex);
          hp->type = SURFACE;
          hp->scrX = eyePath->scrX;
          hp->scrY = eyePath->scrY;

          hp->materialSS = materialIndex;

          hp->throughput = eyePath->throughput * surfaceColor;
          hp->position = hitPoint;
          hp->wo = -eyePath->ray.d;
          hp->normal = shadeN;

          // Free the eye path
          eyePath->done = true;

        }

      }

    }
  }

}

void CPU_Worker::IntersectRay(const Ray *ray, RayHit *rayHit) {

  ss->dataSet->Intersect(ray, rayHit);

}

void CPU_Worker::Intersect(RayBuffer *rayBuffer) {

  // Trace rays
  const Ray *rb = rayBuffer->GetRayBuffer();
  RayHit *hb = rayBuffer->GetHitBuffer();

  double start = WallClockTime();

#ifndef __DEBUG
  omp_set_num_threads(config->max_threads);
#pragma omp parallel for schedule(guided)
#endif
  for (unsigned int i = 0; i < rayBuffer->GetRayCount(); ++i) {
    hb[i].SetMiss();
    IntersectRay(&rb[i], &hb[i]);
  }

  profiler->addRayTracingTime(WallClockTime() - start);
  profiler->addRaysTraced(rayBuffer->GetSize());

}

u_int64_t CPU_Worker::AdvancePhotonPath(u_int64_t photonTarget) {

  uint todoPhotonCount = 0;

  PhotonPath* livePhotonPaths = new PhotonPath[rayBuffer->GetSize()];

  rayBuffer->Reset();

  size_t initc = rayBuffer->GetSize();//min(rayBuffer->GetSize(), photonTarget);

  double start = WallClockTime();

  for (size_t i = 0; i < initc; ++i) {

    int p = rayBuffer->ReserveRay();

    Ray * b = &(rayBuffer->GetRayBuffer())[p];

    engine->InitPhotonPath(engine->ss, &livePhotonPaths[i], b, seedBuffer[i]);

  }

  while (todoPhotonCount < initc) {

    //Intersect(rayBuffer);

#ifndef __DEBUG
    omp_set_num_threads(config->max_threads);
#pragma omp parallel for schedule(guided)
#endif
    for (unsigned int i = 0; i < rayBuffer->GetRayCount(); ++i) {

      PhotonPath *photonPath = &livePhotonPaths[i];
      Ray *ray = &rayBuffer->GetRayBuffer()[i];
      RayHit *rayHit = &rayBuffer->GetHitBuffer()[i];

      while(!photonPath->done) {
        rayHit->SetMiss();
        ss->dataSet->Intersect(ray, rayHit);
        if (rayHit->Miss()) {
          photonPath->done = true;
        } else { // Something was hit

          Point hitPoint;
          Spectrum surfaceColor;
          Normal N, shadeN;

          if (engine->GetHitPointInformation(engine->ss, ray, rayHit, hitPoint, surfaceColor,
              N, shadeN))
            continue;

          const unsigned int currentTriangleIndex = rayHit->index;
          const unsigned int currentMeshIndex = engine->ss->meshIDs[currentTriangleIndex];

          POINTERFREESCENE::Material *hitPointMat =
              &engine->ss->materials[engine->ss->meshMats[currentMeshIndex]];

          uint matType = hitPointMat->type;

          if (matType == MAT_AREALIGHT) {
            photonPath->done = true;
          } else {

            float fPdf;
            Vector wi;
            Vector wo = -ray->d;
            bool specularBounce = true;

            float u0 = getFloatRNG(seedBuffer[i]);
            float u1 = getFloatRNG(seedBuffer[i]);
            float u2 = getFloatRNG(seedBuffer[i]);

            Spectrum f;

            switch (matType) {

            case MAT_MATTE:
              engine->ss->Matte_Sample_f(&hitPointMat->param.matte, &wo, &wi, &fPdf, &f,
                  &shadeN, u0, u1, &specularBounce);

              f *= surfaceColor;
              break;

            case MAT_MIRROR:
              engine->ss->Mirror_Sample_f(&hitPointMat->param.mirror, &wo, &wi, &fPdf,
                  &f, &shadeN, &specularBounce);
              f *= surfaceColor;
              break;

            case MAT_GLASS:
              engine->ss->Glass_Sample_f(&hitPointMat->param.glass, &wo, &wi, &fPdf, &f,
                  &N, &shadeN, u0, &specularBounce);
              f *= surfaceColor;

              break;

            case MAT_MATTEMIRROR:
              engine->ss->MatteMirror_Sample_f(&hitPointMat->param.matteMirror, &wo, &wi,
                  &fPdf, &f, &shadeN, u0, u1, u2, &specularBounce);
              f *= surfaceColor;

              break;

            case MAT_METAL:
              engine->ss->Metal_Sample_f(&hitPointMat->param.metal, &wo, &wi, &fPdf, &f,
                  &shadeN, u0, u1, &specularBounce);
              f *= surfaceColor;

              break;

            case MAT_MATTEMETAL:
              engine->ss->MatteMetal_Sample_f(&hitPointMat->param.matteMetal, &wo, &wi,
                  &fPdf, &f, &shadeN, u0, u1, u2, &specularBounce);
              f *= surfaceColor;

              break;

            case MAT_ALLOY:
              engine->ss->Alloy_Sample_f(&hitPointMat->param.alloy, &wo, &wi, &fPdf, &f,
                  &shadeN, u0, u1, u2, &specularBounce);
              f *= surfaceColor;

              break;

            case MAT_ARCHGLASS:
              engine->ss->ArchGlass_Sample_f(&hitPointMat->param.archGlass, &wo, &wi,
                  &fPdf, &f, &N, &shadeN, u0, &specularBounce);
              f *= surfaceColor;

              break;

            case MAT_NULL:
              wi = ray->d;
              specularBounce = 1;
              fPdf = 1.f;
              break;

            default:
              // Huston, we have a problem...
              specularBounce = 1;
              fPdf = 0.f;
              break;
            }

            if (!specularBounce) // if difuse
              lookupA->AddFlux(HPsIterationRadianceFlux, engine->ss, engine->alpha,
                  hitPoint, shadeN, -ray->d, photonPath->flux, currentPhotonRadius2);

            if (photonPath->depth < MAX_PHOTON_PATH_DEPTH) {
              // Build the next vertex path ray
              if ((fPdf <= 0.f) || f.Black()) {
                photonPath->done = true;
              } else {
                photonPath->depth++;
                photonPath->flux *= f / fPdf;

                // Russian Roulette
                const float p = 0.75f;
                if (photonPath->depth < 3) {
                  *ray = Ray(hitPoint, wi);
                } else if (getFloatRNG(seedBuffer[i]) < p) {
                  photonPath->flux /= p;
                  *ray = Ray(hitPoint, wi);
                } else {
                  photonPath->done = true;
                }
              }
            } else {
              photonPath->done = true;
            }
          }
        }
      }
      todoPhotonCount++;
    }

    //uint oldc = rayBuffer->GetRayCount();

    rayBuffer->Reset();

    /*for (unsigned int i = 0; i < oldc; ++i) {

      PhotonPath *photonPath = &livePhotonPaths[i];
      Ray *ray = &rayBuffer->GetRayBuffer()[i];

      if (photonPath->done && todoPhotonCount < photonTarget) {
        todoPhotonCount++;

        Ray n;
        engine->InitPhotonPath(engine->ss, photonPath, &n, seedBuffer[i]);

        livePhotonPaths[i].done = false;

        size_t p = rayBuffer->AddRay(n);
        livePhotonPaths[p] = *photonPath;

      } else if (!photonPath->done) {
        rayBuffer->AddRay(*ray);
      }
    }*/
  }


  rayBuffer->Reset();

  delete[] livePhotonPaths;
  /*profiler->addPhotonTracingTime(WallClockTime() - start);
  profiler->addPhotonsTraced(todoPhotonCount);

  float MPhotonsSec = todoPhotonCount / ((WallClockTime()-start) * 1000000.f);

    //printf("\nCPU Rate: %.3f MPhotons/sec\n",MPhotonsSec);*/

  return initc;
}

void CPU_Worker::updateDeviceHitPoints() {

}


void CPU_Worker::AccumulateFluxPPMPA(uint iteration, u_int64_t photonTraced) {

#ifndef __DEBUG
  omp_set_num_threads(config->max_threads);
#pragma omp parallel for schedule(guided)
#endif
  for (uint i = 0; i < engine->hitPointTotal; i++) {

    HitPointPositionInfo *ehp = GetHitPointInfo(i);
    HitPointRadianceFlux *ihp = GetHitPoint(i);

    ihp->radiance = Spectrum();

    switch (ehp->type) {
    case CONSTANT_COLOR:
      ihp->radiance = ehp->throughput;

      break;
    case SURFACE:

      if ((ihp->accumPhotonCount > 0)) {

        ihp->reflectedFlux = ihp->accumReflectedFlux;

        //out of the loop
        const double k = 1.0 / (M_PI * currentPhotonRadius2 * photonTraced);

        ihp->radiance = ihp->reflectedFlux * k;

      }
      break;
    default:
      assert (false);
    }

    ihp->accumPhotonCount = 0;
    ihp->accumReflectedFlux = Spectrum();


  }
}
