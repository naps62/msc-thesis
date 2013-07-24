/*
 * smallppmgpu.h
 *
 *  Created on: Jul 25, 2012
 *      Author: rr
 */

#ifndef SMALLPPMGPU_H_
#define SMALLPPMGPU_H_

#include "core.h"
#include "pointerfreescene.h"
#include "hitpoints.h"
#include "List.h"
#include "config.h"

class PhotonPath;
class PPM;

//required for GLUT
extern PPM* engine;

/**
 * Everything that has buff as suffix is a GPU buffer
 */
class PPM {

private:
  //unsigned long long photonTraced;
  uint interationCount;
  Film *film;
  unsigned long long photonTracedTotal;



public:
  //HitPoint* hitPoints_Acc;

  boost::mutex hitPointsLock;

  boost::thread* draw_thread;

#if defined USE_PPMPA || defined USE_PPM
  boost::barrier* waitForHitPoints;
  HitPointPositionInfo* HPsPositionInfo;
#endif


  PointerFreeScene *ss;

  unsigned long long photonsFirstIteration;
  unsigned int width;
  unsigned int height;
  unsigned int superSampling;
  float alpha;

  unsigned int hitPointTotal;
  double startTime;
  char* SPPMG_LABEL;
  unsigned int scrRefreshInterval;

  std::string fileName;

  PPM(float alpha_, uint width_, uint height_, uint superSampling_,
      unsigned long long photonsFirstIteration_, uint ndevices) {

#ifdef USE_PPMPA
    waitForHitPoints =new boost::barrier(ndevices);
#endif
    alpha = alpha_;
    width = width_;
    height = height_;

    superSampling = superSampling_;

    photonsFirstIteration = photonsFirstIteration_;
    //rng = new TauswortheRandomGenerator(7);

    interationCount = 1;

    hitPointTotal = width * height * superSampling * superSampling;

    //hitPoints_Acc = new HitPoint[hitPointTotal];
    //hitPointsInfo = new HitPointInfo[hitPointTotal];

    //memset(hitPoints_Acc, 0,sizeof(HitPoint)*hitPointTotal);
    //    memset(hitPointsInfo, 0,sizeof(HitPointInfo)*hitPointTotal);


    HPsPositionInfo = new HitPointPositionInfo[hitPointTotal];


    photonTracedTotal = 0;

    scrRefreshInterval = 1000;
    startTime = 0.0;

    SPPMG_LABEL = (char*) "Many-core Coherent Progressive PM";

    film = new Film(width_, height_);
    film->Reset();

  }

  ~PPM() {
    //    delete hitPoints;
    //    delete hitPointsInfo;
    delete film;
    delete ss;

  }

  void lockHitPoints() {

    hitPointsLock.lock();

  }

  void unlockHitPoints() {

    hitPointsLock.unlock();

  }

  uint GetIterationNumber() {
    return interationCount;
  }

  uint IncIteration() {
    return interationCount++;
  }
  __HD__
  void SplatSampleBuffer(SampleFrameBuffer* sampleFrameBuffer,const bool preview, SampleBuffer *sampleBuffer) {
    film->SplatSampleBuffer(sampleFrameBuffer,preview, sampleBuffer);

  }

  void ResetFilm() {
    film->Reset();
  }

  void UpdateScreenBuffer() {
    film->UpdateScreenBuffer();
  }

  void LockImageBuffer(){
    film->imageBufferMutex.lock();
  }

  void UnlockImageBuffer(){
    film->imageBufferMutex.unlock();
    }

  HitPointPositionInfo *GetHitPointInfo(const unsigned int index) {
    return &(HPsPositionInfo)[index];
  }

  bool filmCreated() {
    return film;
  }

  const float * GetScreenBuffer() {
    return film->GetScreenBuffer();
  }

  bool GetHitPointInformation(PointerFreeScene *ss, Ray *ray, const RayHit *rayHit,
      Point & hitPoint, Spectrum & surfaceColor, Normal & N, Normal & shadeN);

  void InitPhotonPath(PointerFreeScene *ss, PhotonPath *photonPath, Ray *ray, Seed& seed);

  void SaveImpl(const std::string &fileName) {
    film->SaveImpl(fileName);

  }

  //  HitPointInfo *GetHitPointInfo(const unsigned int index) {
  //    return &(hitPointsInfo)[index];
  //  }

  unsigned long long incPhotonTracedTotal(unsigned long long i) {
    return __sync_fetch_and_add(&photonTracedTotal, i);
  }

  unsigned long long getPhotonTracedTotal() {
    return photonTracedTotal;
  }

};

void InitGlut(int argc, char *argv[], const unsigned int width, const unsigned int height);
void RunGlut(const unsigned int width, const unsigned int height);

void gpupppm(int argc, char *argv[]);

#endif /* SMALLPPMGPU_H_ */
