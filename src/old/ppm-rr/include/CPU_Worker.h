/*
 * CPU_Worker.h
 *
 *  Created on: Oct 31, 2012
 *      Author: rr
 */

#ifndef CPU_WORKER_H_
#define CPU_WORKER_H_

#include "core.h"
#include "pointerfreescene.h"
#include "hitpoints.h"
#include "List.h"
#include "cuda_utils.h"
#include "renderEngine.h"
#include "lookupAcc.h"
#include <xmmintrin.h>
#include "Profiler.h"
#include "luxrays/accelerators/qbvhaccel.h"
#include "config.h"
#include "Worker.h"

class CPU_Worker;

class CPU_Worker: public Worker {
public:

  HashGridLookup* lookupA;

  RayBuffer *rayBuffer;

  CPU_Worker(uint device, PointerFreeScene *ss, uint buffer_size, Seed* sb,
      bool buildHitPoints = false) :
    Worker(sb) {

    deviceID = device;

    rayBuffer = new RayBuffer(buffer_size);

    lookupA = new HashGridLookup(engine->hitPointTotal);

    setScene(ss);

    thread = new boost::thread(boost::bind(CPU_Worker::Entry, this, buildHitPoints));

    //Entry(this,true);

  }

  ~CPU_Worker();

  BBox* GetHostBBox(){
    return &(lookupA->hitPointsbbox);
  }

  void resetRayBuffer() {

    rayBuffer->Reset();
    memset(rayBuffer->GetHitBuffer(), 0, sizeof(RayHit) * rayBuffer->GetSize());
    memset(rayBuffer->GetRayBuffer(), 0, sizeof(Ray) * rayBuffer->GetSize());

  }

  void AdvanceEyePaths(RayBuffer *rayBuffer, EyePath *todoEyePaths, uint *eyePathIndexes);

  u_int64_t AdvancePhotonPath(u_int64_t photontarget);

  void updateDeviceHitPoints();

  void CommitIterationHitPoints(u_int64_t photonPerIteration);
  void MirrorHitPoints();

  void Intersect(RayBuffer *rayBuffer);
  void IntersectRay(const Ray *ray, RayHit *rayHit);

  void updateDeviceLookupAcc() {

  }

  void getDeviceHitpoints() {

  }

  void Render(bool buildHitPoints) {

#if defined USE_PPMPA || defined USE_PPM
    if (buildHitPoints) {
      BuildHitPoints(1);

    #if defined USE_PPMPA
      engine->globalHitPointsStaticInfo = hitPointsStaticInfo_iterationCopy;
    #endif
    }
    #if defined USE_PPMPA
    engine->waitForHitPoints->wait();

    hitPointsStaticInfo_iterationCopy = engine->globalHitPointsStaticInfo;

    //non-static info allocated and initialized in worker constructor
    #endif
#endif

    ProcessIterations(engine);
  }

  size_t getRaybufferSize() {
    return rayBuffer->GetSize();
  }

  void SetBBox(BBox hitPointsbbox) {
    lookupA->setBBox(hitPointsbbox);
  }

  void LookupSetHitPoints(HitPointStaticInfo* iterationHitPoints, HitPoint* workerHitPoints) {
    lookupA->setHitpoints(iterationHitPoints, workerHitPoints);
  }

  size_t RaybufferAddRay(const Ray &ray) {
    return rayBuffer->AddRay(ray);
  }

  uint getRayBufferRayCount() {
    return rayBuffer->GetRayCount();
  }

  void AdvanceEyePaths(EyePath* todoEyePaths, uint* eyePathIndexes) {
    AdvanceEyePaths(rayBuffer, todoEyePaths, eyePathIndexes);
  }

  void IntersectRayBuffer() {
    Intersect(rayBuffer);
  }

  void ReHash(float currentPhotonRadius2) {
    lookupA->ReHash(currentPhotonRadius2);
  }

  RayBuffer* GetRayBuffer() {
    return rayBuffer;
  }

  static void Entry(CPU_Worker *worker, bool buildHitPoints) {
    worker->Render(buildHitPoints);
  }

};

#endif /* CUDA_WORKER_H_ */
