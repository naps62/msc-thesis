/*
 * Worker.cpp
 *
 *  Created on: Nov 10, 2012
 *      Author: rr
 */

#include "Worker.h"
#include "omp.h"

 extern Config* config;

Worker::~Worker() {
  // TODO Auto-generated destructor stub
}

void Worker::BuildHitPoints(uint iteration) {

  const unsigned int width = engine->width;
  const unsigned int height = engine->height;
  const unsigned int superSampling = engine->superSampling;

  const unsigned int hitPointTotal = engine->hitPointTotal;

  EyePath* todoEyePaths = new EyePath[hitPointTotal];

#ifndef USE_SPPM
  memset(HPsIterationRadianceFlux, 0, sizeof(HitPointRadianceFlux) * engine->hitPointTotal);
#endif

  unsigned int hitPointsIndex = 0;
  const float invSuperSampling = 1.f / superSampling;

  for (unsigned int y = 0; y < height; ++y) {

    //for all hitpoints
    for (unsigned int x = 0; x < width; ++x) {
      for (unsigned int sy = 0; sy < superSampling; ++sy) {
        for (unsigned int sx = 0; sx < superSampling; ++sx) {

          EyePath *eyePath = &todoEyePaths[hitPointsIndex];

          eyePath->scrX = x + (sx + getFloatRNG(seedBuffer[hitPointsIndex]))
              * invSuperSampling - 0.5f;

          eyePath->scrY = y + (sy + getFloatRNG(seedBuffer[hitPointsIndex]))
              * invSuperSampling - 0.5f;

          float u0 = getFloatRNG(seedBuffer[hitPointsIndex]);
          float u1 = getFloatRNG(seedBuffer[hitPointsIndex]);
          float u2 = getFloatRNG(seedBuffer[hitPointsIndex]);

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

  uint todoEyePathCount = hitPointTotal;
  uint chunk_counter = 0;

  uint* eyePathIndexes = new uint[getRaybufferSize()];

  while (todoEyePathCount > 0) {

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
        HitPointPositionInfo* hp = GetHitPointInfo(eyePath->sampleIndex);

        hp->type = CONSTANT_COLOR;
        hp->scrX = eyePath->scrX;
        hp->scrY = eyePath->scrY;
        hp->throughput = Spectrum();

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
    }

    if (getRayBufferRayCount() > 0) {

      IntersectRayBuffer();

      AdvanceEyePaths(&todoEyePaths[0], eyePathIndexes);

      resetRayBuffer();
    }
  }

  delete[] todoEyePaths;
  delete[] eyePathIndexes;

}

void Worker::ProcessIterations(PPM* engine) {

  u_int64_t photonPerIteration = engine->photonsFirstIteration;

  uint iterationCount;

  resetRayBuffer();

  UpdateBBox();

  uint iter = 0;
  while (!boost::this_thread::interruption_requested() && iter < config->max_iters) {
    ++iter;

    double start = WallClockTime();

    iterationCount = engine->IncIteration();

    if (engine->GetIterationNumber() > MAX_ITERATIONS) {
      break;
    }

//    __BENCH.LOOP_STAGE_START("Process Iterations > Iterations");

    photonPerIteration = engine->photonsFirstIteration;

    fprintf(stderr, "\n#######\n Processing iteration %d with %lu photons in device %d...\n",
        iterationCount, photonPerIteration, getDeviceID());

#if defined USE_SPPMPA || defined USE_SPPM
    BuildHitPoints(iterationCount);
    UpdateBBox();

#endif

#if defined USE_SPPM || defined USE_PPM
    if (iterationCount == 1)
    InitRadius(iterationCount);
    fprintf(stderr, "Iteration radius: %f\n", HPsIterationRadianceFlux[0].accumPhotonRadius2);

#else
    InitRadius(iterationCount);
    fprintf(stderr, "Iteration radius: %f\n", currentPhotonRadius2);

#endif
//    __BENCH.LOOP_STAGE_START("Process Iterations > Iterations > Cpy HPs to device");

    updateDeviceHitPoints();

//    __BENCH.LOOP_STAGE_STOP("Process Iterations > Iterations > Cpy HPs to device");

//    __BENCH.LOOP_STAGE_START("Process Iterations > Iterations > ReHash");

#ifndef REBUILD_HASH
    if (iterationCount == 1)
#endif
      ReHash(currentPhotonRadius2);//argument ignored in non-PA

//    __BENCH.LOOP_STAGE_STOP("Process Iterations > Iterations > ReHash");

#ifndef REBUILD_HASH
    if (iterationCount == 1)
#endif
      updateDeviceLookupAcc();

//    __BENCH.LOOP_STAGE_START("Process Iterations > Iterations > Build Photon Map");

    photonPerIteration = AdvancePhotonPath(photonPerIteration);

    //fprintf(stderr, "Traced %lu photon\n", photonPerIteration);

//    __BENCH.LOOP_STAGE_STOP("Process Iterations > Iterations > Build Photon Map");

#if defined USE_PPM
//    __BENCH.LOOP_STAGE_START("Process Iterations > Iterations > Get HPs");
    getDeviceHitpoints();
//    __BENCH.LOOP_STAGE_STOP("Process Iterations > Iterations > Get HPs");

//    __BENCH.LOOP_STAGE_START("Process Iterations > Iterations > Radiance calc");
    AccumulateFluxPPM(iterationCount, photonPerIteration);
//    __BENCH.LOOP_STAGE_STOP("Process Iterations > Iterations > Radiance calc");

#endif
#if defined USE_SPPM
    AccumulateFluxSPPM(iterationCount, photonPerIteration);
#endif
#if defined USE_SPPMPA
    AccumulateFluxSPPMPA(iterationCount, photonPerIteration);
#endif
#if defined USE_PPMPA
//    __BENCH.LOOP_STAGE_START("Process Iterations > Iterations > Radiance calc");
    AccumulateFluxPPMPA(iterationCount, photonPerIteration);
//    __BENCH.LOOP_STAGE_STOP("Process Iterations > Iterations > Radiance calc");

//    __BENCH.LOOP_STAGE_START("Process Iterations > Iterations > Get HPs");
    getDeviceHitpoints();
//    __BENCH.LOOP_STAGE_STOP("Process Iterations > Iterations > Get HPs");

#endif
//    __BENCH.LOOP_STAGE_START("Process Iterations > Iterations > Splat radiance");

    UpdateSampleFrameBuffer(photonPerIteration);

//    __BENCH.LOOP_STAGE_STOP("Process Iterations > Iterations > Splat radiance");

    /**
     * iteration lock required in PhotonTracedTotal
     */
    engine->incPhotonTracedTotal(photonPerIteration);

    profiler->additeratingTime(WallClockTime() - start);
    profiler->addIteration(1);

    if (profiler->iterationCount % 20 == 0)
      profiler->printStats(deviceID);

    if (iterationCount % 100 == 0)
      engine->SaveImpl(to_string<uint> (iterationCount, std::dec) + engine->fileName);

//    __BENCH.LOOP_STAGE_STOP("Process Iterations > Iterations");

  }

  profiler->printStats(deviceID);

}

void Worker::UpdateBBox() {

  // Calculate hit points bounding box
  //std::cerr << "Building hit points bounding box: ";

  BBox hitPointsbbox = BBox();

  for (unsigned int i = 0; i < engine->hitPointTotal; ++i) {
    HitPointPositionInfo *hp = GetHitPointInfo(i);

    if (hp->type == SURFACE)
      hitPointsbbox = Union(hitPointsbbox, hp->position);
  }

  SetBBox(hitPointsbbox);

}

void Worker::InitRadius(uint iteration) {

  BBox* hitPointsbbox = GetHostBBox();

  Vector ssize = hitPointsbbox->pMax - hitPointsbbox->pMin;

  float photonRadius = ((ssize.x + ssize.y + ssize.z) / 3.f) / ((engine->width
      * engine->superSampling + engine->height * engine->superSampling) / 2.f) * 2.f;

  float photonRadius2 = photonRadius * photonRadius;

#if defined USE_SPPMPA || defined USE_PPMPA

  float g = 1;
  for (uint k = 1; k < iteration; k++)
    g *= (k + engine->alpha) / k;

  g /= iteration;

  photonRadius2 = photonRadius2 * g;
  std::cout << "photon_radius: " << photonRadius2 << '\n';
#endif

#ifdef REBUILD_HASH
  // Expand the bounding box by used radius
  hitPointsbbox->Expand(sqrt(photonRadius2));
#endif
  // Initialize hit points field
  //const float photonRadius2 = photonRadius * photonRadius;

#if defined USE_SPPMPA || defined USE_PPMPA
  currentPhotonRadius2 = photonRadius2;
#else
  for (unsigned int i = 0; i < engine->hitPointTotal; ++i) {

    //HitPointInfo *hpinfo = engine->GetHitPointInfo(i);
    HitPointRadianceFlux *hp = GetHitPoint(i);

    hp->accumPhotonRadius2 = photonRadius2;

  }
#endif

}

void Worker::UpdateSampleFrameBuffer(unsigned long long iterationPhotonCount) {

#ifndef __DEBUG
  omp_set_num_threads(config->max_threads);
#pragma omp parallel for schedule(guided)
#endif
  for (unsigned int i = 0; i < engine->hitPointTotal; ++i) {
    HitPointPositionInfo *hp = GetHitPointInfo(i);
    HitPointRadianceFlux *ihp = GetHitPoint(i);

#if defined USE_SPPM || defined USE_SPPMPA
    const float scrX = i % engine->width;
    const float scrY = i / engine->width;

    sampleBuffer->SplatSample(scrX, scrY, ihp->radiance);
#endif

#if defined USE_PPM || defined USE_PPMPA
    sampleBuffer->SplatSample(hp->scrX, hp->scrY, ihp->radiance);
#endif

  }

  sampleFrameBuffer->Clear();

  if (sampleBuffer->GetSampleCount() > 0) {

    engine->SplatSampleBuffer(sampleFrameBuffer, true, sampleBuffer);
    sampleBuffer->Reset();
  }
}

#ifdef USE_PPM
void Worker::AccumulateFluxPPM(uint iteration, u_int64_t photonTraced) {

  photonTraced += engine->getPhotonTracedTotal();

#ifndef __DEBUG
  omp_set_num_threads(config->max_threads);
#pragma omp parallel for schedule(guided)
#endif
  for (uint i = 0; i < engine->hitPointTotal; i++) {
    HitPointPositionInfo *ehp = GetHitPointInfo(i);
    HitPointRadianceFlux *ihp = GetHitPoint(i);

    switch (ehp->type) {
      case CONSTANT_COLOR:
      ihp->radiance = ehp->throughput;
      break;
      case SURFACE:

      if ((ihp->accumPhotonCount > 0)) {

        const unsigned long long pcount = ihp->photonCount + ihp->accumPhotonCount;
        const float alpha = engine->alpha;

        const float g = alpha * pcount / (ihp->photonCount * alpha + ihp->accumPhotonCount);

        ihp->accumPhotonRadius2 *= g;

        ihp->reflectedFlux = (ihp->reflectedFlux + ihp->accumReflectedFlux) * g;

        ihp->photonCount = pcount;

        const double k = 1.0 / (M_PI * ihp->accumPhotonRadius2 * photonTraced);

        ihp->radiance = ihp->reflectedFlux * k;

        ihp->accumPhotonCount = 0;
        ihp->accumReflectedFlux = Spectrum();
      }

      break;
      default:
      assert (false);
    }

  }

  //fprintf(stderr, "Iteration %d hit point 0 reducted radius: %f\n", iteration,
  //    GetHitPoint(0)->accumPhotonRadius2);
}
#endif

#ifdef USE_SPPM
void Worker::AccumulateFluxSPPM(uint iteration, u_int64_t photonTraced) {

  photonTraced += engine->getPhotonTracedTotal();

#ifndef __DEBUG
  omp_set_num_threads(config->max_threads);
#pragma omp parallel for schedule(guided)
#endif
  for (uint i = 0; i < engine->hitPointTotal; i++) {
    HitPointPositionInfo *ehp = GetHitPointInfo(i);
    HitPointRadianceFlux *ihp = GetHitPoint(i);

    switch (ehp->type) {
      case CONSTANT_COLOR:
      ihp->accumRadiance += ehp->throughput;
      ihp->constantHitsCount += 1;
      break;
      case SURFACE:

      if ((ihp->accumPhotonCount > 0)) {

        const unsigned long long pcount = ihp->photonCount + ihp->accumPhotonCount;
        const float alpha = engine->alpha;

        const float g = alpha * pcount / (ihp->photonCount * alpha + ihp->accumPhotonCount);

        ihp->accumPhotonRadius2 *= g;

        ihp->reflectedFlux = (ihp->reflectedFlux + ihp->accumReflectedFlux) * g;

        ihp->photonCount = pcount;

        ihp->accumPhotonCount = 0;
        ihp->accumReflectedFlux = Spectrum();
      }

      ihp->surfaceHitsCount += 1;
      break;
      default:
      assert (false);
    }
    const unsigned int hitCount = ihp->constantHitsCount + ihp->surfaceHitsCount;
    if (hitCount > 0) {

      const double k = 1.0 / (M_PI * ihp->accumPhotonRadius2 * photonTraced);
      Spectrum radiance_r;
      radiance_r = (ihp->accumRadiance + ihp->surfaceHitsCount * ihp->reflectedFlux * k)
      / hitCount;
      ihp->radiance = radiance_r;
    }
  }

  fprintf(stderr, "Iteration %d hit point 0 reducted radius: %f\n", iteration,
      GetHitPoint(0)->accumPhotonRadius2);
}
#endif

#ifdef USE_SPPMPA
void Worker::AccumulateFluxSPPMPA(uint iteration, u_int64_t photonTraced) {

#ifndef __DEBUG
  omp_set_num_threads(config->max_threads);
#pragma omp parallel for schedule(guided)
#endif
  for (uint i = 0; i < engine->hitPointTotal; i++) {
    HitPointPositionInfo *ehp = GetHitPointInfo(i);
    HitPointRadianceFlux *ihp = GetHitPoint(i);

    switch (ehp->type) {
      case CONSTANT_COLOR:
      ihp->accumRadiance += ehp->throughput;
      ihp->constantHitsCount += 1;
      break;
      case SURFACE:

      if ((ihp->accumPhotonCount > 0)) {

        ihp->reflectedFlux = ihp->accumReflectedFlux;
        ihp->accumPhotonCount = 0;
        ihp->accumReflectedFlux = Spectrum();
      }
      ihp->surfaceHitsCount += 1;
      break;
      default:
      assert (false);
    }

    const unsigned int hitCount = ihp->constantHitsCount + ihp->surfaceHitsCount;

    if (hitCount > 0) {

      const double k = 1.0 / (M_PI * currentPhotonRadius2 * photonTraced);

      ihp->radiance = (ihp->accumRadiance + ihp->reflectedFlux * k);

    }

  }

  for (uint i = 0; i < engine->hitPointTotal; i++) {
    HitPointRadianceFlux *ihp = GetHitPoint(i);

    ihp->constantHitsCount = 0;
    ihp->surfaceHitsCount = 0;
    ihp ->accumRadiance = Spectrum();
  }

  fprintf(stderr, "Iteration %d hit point 0 reducted radius: %f\n", iteration,
      currentPhotonRadius2);
}
#endif

HitPointPositionInfo *Worker::GetHitPointInfo(const unsigned int index) {
  return &(engine->HPsPositionInfo)[index];
}

HitPointRadianceFlux *Worker::GetHitPoint(const unsigned int index) {

  return &(HPsIterationRadianceFlux)[index];
}

void Worker::setScene(PointerFreeScene *s) {
  ss = s;
}

uint Worker::getDeviceID() {
  return deviceID;
}
