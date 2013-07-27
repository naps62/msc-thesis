/***************************************************************************
 *   Copyright (C) 1998-2010 by authors (see AUTHORS.txt )                 *
 *                                                                         *
 *   This file is part of LuxRays.                                         *
 *                                                                         *
 *   LuxRays is free software; you can redistribute it and/or modify       *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   LuxRays is distributed in the hope that it will be useful,            *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>. *
 *                                                                         *
 *   LuxRays website: http://www.luxrender.net                             *
 ***************************************************************************/

// Based on smallppm, Progressive Photon Mapping by T. Hachisuka

#include "renderEngine.h"
#include "omp.h"
#include "random.h"
#include "atomic.h"
#include "hitpoints.h"

PPM* engine;


#include "cuda_utils.h"
HitPointRadianceFlux *hitPoints;

bool PPM::GetHitPointInformation(PointerFreeScene *ss, Ray *ray,
    const RayHit *rayHit, Point &hitPoint, Spectrum &surfaceColor,
    Normal &N, Normal &shadeN) {

  hitPoint = (*ray)(rayHit->t);
  const unsigned int currentTriangleIndex = rayHit->index;

  unsigned int currentMeshIndex;
  unsigned int triIndex;

  //########

  //currentMeshIndex = scene->dataSet->GetMeshID(currentTriangleIndex);

  // Get the triangle
  //const ExtMesh *mesh = scene->objects[currentMeshIndex];

  //triIndex = scene->dataSet->GetMeshTriangleID(currentTriangleIndex);

  //    if (mesh->HasColors())
  //      surfaceColor = mesh->InterpolateTriColor(triIndex, rayHit->b1,
  //          rayHit->b2);
  //    else
  //      surfaceColor = Spectrum(1.f, 1.f, 1.f);
  // Interpolate face normal
  //      N = mesh->InterpolateTriNormal(triIndex, rayHit->b1, rayHit->b2);

  //########

  currentMeshIndex = ss->meshIDs[currentTriangleIndex];
  triIndex = currentTriangleIndex
      - ss->meshFirstTriangleOffset[currentMeshIndex];

  POINTERFREESCENE::Mesh m = ss->meshDescs[currentMeshIndex];

  //SSCENE::Material *hitPointMat = &ss->mats[ss->meshMats[currentMeshIndex]];

  //    if (mesh->HasColors())
  //    for (int i = 0; i < mesh->GetTotalVertexCount(); i++) {
  //      if (mesh->GetColor(i).r != ss->colors[m.colorsOffset + i].r ||
  //          mesh->GetColor(i).g != ss->colors[m.colorsOffset + i].g ||
  //          mesh->GetColor(i).b  != ss->colors[m.colorsOffset + i].b) {
  //        printf("asdasdad");
  //      }
  //    }

  if (m.hasColors) {

    ss->Mesh_InterpolateColor((Spectrum*) &ss->colors[m.colorsOffset],
        &ss->tris[m.trisOffset], triIndex, rayHit->b1, rayHit->b2,
        &surfaceColor);

  } else {
    surfaceColor = Spectrum(1.f, 1.f, 1.f);
  }

  ss->Mesh_InterpolateNormal(&ss->normals[m.vertsOffset],
      &ss->tris[m.trisOffset], triIndex, rayHit->b1, rayHit->b2, N);

  //    // Check if I have to apply texture mapping or normal mapping
  //    TexMapInstance *tm =
  //        scene->objectTexMaps[currentMeshIndex];
  //
  //
  //
  //    BumpMapInstance *bm =
  //        scene->objectBumpMaps[currentMeshIndex];
  //
  //    NormalMapInstance *nm =
  //        scene->objectNormalMaps[currentMeshIndex];
  //
  //    if (tm || bm || nm) {
  //      // Interpolate UV coordinates if required
  //
  //
  //      //const UV triUV = mesh->InterpolateTriUV(triIndex, rayHit->b1, rayHit->b2);
  //
  //
  //      UV triUV;
  //      ss->Mesh_InterpolateUV(&ss->uvs[m.vertsOffset],
  //          &ss->tris[m.trisOffset], triIndex, rayHit->b1,
  //          rayHit->b2, &triUV);
  //
  //      // Check if there is an assigned texture map
  //      if (tm) {
  //        const TextureMap *map = tm->GetTexMap();
  //
  //        // Apply texture mapping
  //        surfaceColor *= map->GetColor(triUV);
  //
  //        // Check if the texture map has an alpha channel
  //        if (map->HasAlpha()) {
  //          const float alpha = map->GetAlpha(triUV);
  //
  //          if ((alpha == 0.0f) || ((alpha < 1.f)
  //              && (rndGen->floatValue() > alpha))) {
  //            *ray = Ray(hitPoint, ray->d);
  //            return true;
  //          }
  //        }
  //      }
  //
  //      // Check if there is an assigned bump/normal map
  //      if (bm || nm) {
  //        if (nm) {
  //          // Apply normal mapping
  //          const Spectrum color = nm->GetTexMap()->GetColor(
  //              triUV);
  //
  //          const float x = 2.0 * (color.r - 0.5);
  //          const float y = 2.0 * (color.g - 0.5);
  //          const float z = 2.0 * (color.b - 0.5);
  //
  //          Vector v1, v2;
  //          CoordinateSystem(Vector(N), &v1, &v2);
  //          N = Normalize(
  //              Normal(v1.x * x + v2.x * y + N.x * z,
  //                  v1.y * x + v2.y * y + N.y * z,
  //                  v1.z * x + v2.z * y + N.z * z));
  //        }
  //
  //        if (bm) {
  //          // Apply bump mapping
  //          const TextureMap *map = bm->GetTexMap();
  //          const UV &dudv = map->GetDuDv();
  //
  //          const float b0 = map->GetColor(triUV).Filter();
  //
  //          const UV uvdu(triUV.u + dudv.u, triUV.v);
  //          const float bu = map->GetColor(uvdu).Filter();
  //
  //          const UV uvdv(triUV.u, triUV.v + dudv.v);
  //          const float bv = map->GetColor(uvdv).Filter();
  //
  //          const float scale = bm->GetScale();
  //          const Vector bump(scale * (bu - b0),
  //              scale * (bv - b0), 1.f);
  //
  //          Vector v1, v2;
  //          CoordinateSystem(Vector(N), &v1, &v2);
  //          N = Normalize(
  //              Normal(
  //                  v1.x * bump.x + v2.x * bump.y + N.x
  //                      * bump.z,
  //                  v1.y * bump.x + v2.y * bump.y + N.y
  //                      * bump.z,
  //                  v1.z * bump.x + v2.z * bump.y + N.z
  //                      * bump.z));
  //        }
  //      }
  //    }

  // Flip the normal if required
  if (Dot(ray->d, N) > 0.f)
    shadeN = -N;
  else
    shadeN = N;

  return false;
}

void PPM::InitPhotonPath(PointerFreeScene* ss, PhotonPath *photonPath, Ray *ray,
    Seed& seed) {

  //Scene *scene = ss->scene;
  // Select one light source
  float lpdf;
  float pdf;

  Spectrum f;

  //photonPath->seed = mwc();

  //seed = mwc();

  float u0 = getFloatRNG(seed);
  float u1 = getFloatRNG(seed);
  float u2 = getFloatRNG(seed);
  float u3 = getFloatRNG(seed);
  float u4 = getFloatRNG(seed);
  //float u5 = getFloatRNG(seed);

//    float u0 = getFloatRNG2(seed);
//    float u1 = getFloatRNG2(seed);
//    float u2 = getFloatRNG2(seed);
//    float u3 = getFloatRNG2(seed);
//    float u4 = getFloatRNG2(seed);
//    float u5 = getFloatRNG2(seed);


  int lightIndex;
  POINTERFREESCENE::LightSourceType lightT = ss->SampleAllLights(u0, &lpdf,
      lightIndex, ss->infiniteLight, ss->sunLight, ss->skyLight);

  if (lightT == POINTERFREESCENE::TYPE_IL_IS)
    ss->InfiniteLight_Sample_L(u1, u2, u3, u4, u4/*u5*/, &pdf, ray,
        photonPath->flux, ss->infiniteLight, ss->infiniteLightMap);

  else if (lightT == POINTERFREESCENE::TYPE_SUN)
    ss->SunLight_Sample_L(u1, u2, u3, u4, u4/*u5*/, &pdf, ray, photonPath->flux,
        ss->sunLight);

  else if (lightT == POINTERFREESCENE::TYPE_IL_SKY)
    ss->SkyLight_Sample_L(u1, u2, u3, u4, u4/*u5*/, &pdf, ray, photonPath->flux,
        ss->skyLight);

  else {
    ss->TriangleLight_Sample_L(&ss->areaLights[lightIndex], u1, u2, u3, u4,
        u4/*u5*/, &pdf, ray, photonPath->flux, &ss->colors[0],
        &ss->meshDescs[0]);
  }

  //##########

  //const LightSource *light2 = scene->SampleAllLights(u0, &lpdf);

  // Initialize the photon path
  //photonPath->flux = light2->Sample_L(scene, u1, u2, u3, u4, u5, &pdf, ray);

  //#########


  photonPath->flux /= pdf * lpdf;
  photonPath->depth = 0;

  //incPhotonCount();
}
