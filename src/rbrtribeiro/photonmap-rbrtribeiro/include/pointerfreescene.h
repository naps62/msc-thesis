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

#ifndef _COMPILEDSESSION_H
#define	_COMPILEDSESSION_H

#include "pointerfreescene_types.h"
#include "editaction.h"
//#include "luxrays/core/trianglemesh.h"
//#include "luxrays/utils/sdl/scene.h"
#include "film.h"
//#include "luxrays/accelerators/qbvhaccel.h"
//#include "luxrays/core/dataset.h"
#include "luxrays/utils/sdl/mc.h"
#include "luxrays/core/dataset.h"

//#include "my_cutil_math.h"
#include <float.h>
#include <ostream>
using std::ostream;
using std::endl;

using namespace std;

class Scene;

class PointerFreeScene {
public:

	Scene* scene;
	//Film* film;

	int accelType;
	DataSet *dataSet;

	POINTERFREESCENE::Camera camera;// Compiled Camera

	// Compiled Scene Geometry
	vector<Point> verts;
	vector<Normal> normals;
	vector<Spectrum> colors;
	vector<UV> uvs;
	vector<Triangle> tris;
	vector<POINTERFREESCENE::Mesh> meshDescs;
	unsigned int meshIDs_size;
	const unsigned int *meshIDs;

	// One element for each mesh, it used to translate the RayBuffer currentTriangleIndex
	// to mesh TriangleID
	unsigned int *meshFirstTriangleOffset;

	// Compiled AreaLights
	vector<POINTERFREESCENE::TriangleLight> areaLights;

	uint lightCount; // this is actually the number of area lights

	// Compiled InfiniteLights
	POINTERFREESCENE::InfiniteLight *infiniteLight;
	const Spectrum *infiniteLightMap;

	POINTERFREESCENE::SunLight *sunLight; // Compiled SunLight
	POINTERFREESCENE::SkyLight *skyLight; // Compiled SkyLight

	Point BSphereCenter;
	float BSphereRad;

	vector<POINTERFREESCENE::Material> materials;

	vector<unsigned int> meshMats;

	// Compiled TextureMaps
	vector<POINTERFREESCENE::TexMap> gpuTexMaps;
	unsigned int totRGBTexMem;
	Spectrum *rgbTexMem;
	unsigned int totAlphaTexMem;
	float *alphaTexMem;
	unsigned int *meshTexs;
	// Compiled BumpMaps
	unsigned int *meshBumps;
	float *bumpMapScales;
	// Compiled NormalMaps
	unsigned int *meshNormalMaps;

	unsigned int frameBufferPixelCount;

	//DEVICE BUFFERS




	//CUDASCENE::Pixel *frameBuffer;
	//CUDASCENE::AlphaPixel *alphaFrameBuffer;

	// Compiled Materials
	bool enable_MAT_MATTE, enable_MAT_AREALIGHT, enable_MAT_MIRROR,
			enable_MAT_GLASS, enable_MAT_MATTEMIRROR, enable_MAT_METAL,
			enable_MAT_MATTEMETAL, enable_MAT_ALLOY, enable_MAT_ARCHGLASS;

	PointerFreeScene( uint width_, uint height_,std::string sceneFileName,
			const int aType = -1);
	~PointerFreeScene();

	//void UpdateDataSet();


	//void CopyAcc();

	void Recompile(const EditActionList &editActions);
	//bool IsMaterialCompiled(const MaterialType t) const;


private:
	void CompileCamera();
	void CompileGeometry();
	void CompileMaterials();
	void CompileAreaLights();
	void CompileInfiniteLight();
	void CompileSunLight();
	void CompileSkyLight();
	void CompileTextureMaps();

public:
	friend ostream& operator<< (ostream& os, PointerFreeScene& scene);

	//------------------------------------------------------------------------------
	// Auxiliar
	//------------------------------------------------------------------------------
	__HD__
	float RiAngleBetween(float thetav, float phiv, float theta, float phi) {
		const float cospsi = sin(thetav) * sin(theta) * cos(phi - phiv) + cos(
				thetav) * cos(theta);
		if (cospsi >= 1.f)
			return 0.f;
		if (cospsi <= -1.f)
			return M_PI;
		return acos(cospsi);
	}

	__HD__
	void TexMap_GetTexel(const Spectrum *pixels, const uint width,
			const uint height, const int s, const int t, Spectrum *col) {
		const uint u = Mod(s, width);
		const uint v = Mod(t, height);

		const unsigned index = v * width + u;

		col->r = pixels[index].r;
		col->g = pixels[index].g;
		col->b = pixels[index].b;
	}
	__HD__
	void TexMap_GetColor(const Spectrum *pixels, const uint width,
			const uint height, const float u, const float v, Spectrum *col) {
		const float s = u * width - 0.5f;
		const float t = v * height - 0.5f;

		const int s0 = (int) floor(s);
		const int t0 = (int) floor(t);

		const float ds = s - s0;
		const float dt = t - t0;

		const float ids = 1.f - ds;
		const float idt = 1.f - dt;

		Spectrum c0, c1, c2, c3;
		TexMap_GetTexel(pixels, width, height, s0, t0, &c0);
		TexMap_GetTexel(pixels, width, height, s0, t0 + 1, &c1);
		TexMap_GetTexel(pixels, width, height, s0 + 1, t0, &c2);
		TexMap_GetTexel(pixels, width, height, s0 + 1, t0 + 1, &c3);

		const float k0 = ids * idt;
		const float k1 = ids * dt;
		const float k2 = ds * idt;
		const float k3 = ds * dt;

		col->r = k0 * c0.r + k1 * c1.r + k2 * c2.r + k3 * c3.r;
		col->g = k0 * c0.g + k1 * c1.g + k2 * c2.g + k3 * c3.g;
		col->b = k0 * c0.b + k1 * c1.b + k2 * c2.b + k3 * c3.b;
	}

	__HD__
	float SkyLight_PerezBase(float *lam, float theta, float gamma) {
		return (1.f + lam[1] * exp(lam[2] / cos(theta))) * (1.f + lam[3] * exp(
				lam[4] * gamma) + lam[5] * cos(gamma) * cos(gamma));
	}
	__HD__
	void SkyLight_ChromaticityToSpectrum(const float Y, const float x,
			const float y, Spectrum *s) {
		float X, Z;

		if (y != 0.f)
			X = (x / y) * Y;
		else
			X = 0.f;

		if (y != 0.f && Y != 0.f)
			Z = (1.f - x - y) / y * Y;
		else
			Z = 0.f;

		// Assuming sRGB (D65 illuminant)
		s->r = 3.2410f * X - 1.5374f * Y - 0.4986f * Z;
		s->g = -0.9692f * X + 1.8760f * Y + 0.0416f * Z;
		s->b = 0.0556f * X - 0.2040f * Y + 1.0570f * Z;
	}
	__HD__
	void SkyLight_GetSkySpectralRadiance(const float theta, const float phi,
			Spectrum *spect, POINTERFREESCENE::SkyLight *skyLight) {
		// add bottom half of hemisphere with horizon colour
		const float theta_fin = min(theta, (float) ((M_PI * 0.5f) - 0.001f));
		const float gamma = RiAngleBetween(theta, phi, skyLight->thetaS,
				skyLight->phiS);

		// Compute xyY values
		const float x = skyLight->zenith_x * SkyLight_PerezBase(
				skyLight->perez_x, theta_fin, gamma);
		const float y = skyLight->zenith_y * SkyLight_PerezBase(
				skyLight->perez_y, theta_fin, gamma);
		const float Y = skyLight->zenith_Y * SkyLight_PerezBase(
				skyLight->perez_Y, theta_fin, gamma);

		SkyLight_ChromaticityToSpectrum(Y, x, y, spect);
	}

	//------------------------------------------------------------------------------
	// GenerateCameraRay
	//------------------------------------------------------------------------------
	__HD__
	void GenerateRay(const float screenX, const float screenY,
			const unsigned int filmWidth, const unsigned int filmHeight,
			Ray *ray, const float u1, const float u2, const float u3,
			POINTERFREESCENE::Camera *camera) {

		Point Pras;
		Pras.x = screenX;
		Pras.y = filmHeight - screenY - 1.f;
		Pras.z = 0;

		Point orig;
		// RasterToCamera(Pras, &orig);

		const float iw = 1.f / (camera->rasterToCameraMatrix[3][0] * Pras.x
				+ camera->rasterToCameraMatrix[3][1] * Pras.y
				+ camera->rasterToCameraMatrix[3][2] * Pras.z
				+ camera->rasterToCameraMatrix[3][3]);
		orig.x = (camera->rasterToCameraMatrix[0][0] * Pras.x
				+ camera->rasterToCameraMatrix[0][1] * Pras.y
				+ camera->rasterToCameraMatrix[0][2] * Pras.z
				+ camera->rasterToCameraMatrix[0][3]) * iw;
		orig.y = (camera->rasterToCameraMatrix[1][0] * Pras.x
				+ camera->rasterToCameraMatrix[1][1] * Pras.y
				+ camera->rasterToCameraMatrix[1][2] * Pras.z
				+ camera->rasterToCameraMatrix[1][3]) * iw;
		orig.z = (camera->rasterToCameraMatrix[2][0] * Pras.x
				+ camera->rasterToCameraMatrix[2][1] * Pras.y
				+ camera->rasterToCameraMatrix[2][2] * Pras.z
				+ camera->rasterToCameraMatrix[2][3]) * iw;

		Vector dir;
		dir.x = orig.x;
		dir.y = orig.y;
		dir.z = orig.z;

		const float hither = camera->hither;

		if (camera->lensRadius > 0.f) {
			// Sample point on lens
			float lensU, lensV;
			ConcentricSampleDisk(u1, u2, &lensU, &lensV);
			const float lensRadius = camera->lensRadius;
			lensU *= lensRadius;
			lensV *= lensRadius;

			// Compute point on plane of focus
			const float focalDistance = camera->focalDistance;
			const float dist = focalDistance - hither;
			const float ft = dist / dir.z;
			Point Pfocus;
			Pfocus.x = orig.x + dir.x * ft;
			Pfocus.y = orig.y + dir.y * ft;
			Pfocus.z = orig.z + dir.z * ft;

			// Update ray for effect of lens
			const float k = dist / focalDistance;
			orig.x += lensU * k;
			orig.y += lensV * k;

			dir.x = Pfocus.x - orig.x;
			dir.y = Pfocus.y - orig.y;
			dir.z = Pfocus.z - orig.z;

		}

		dir = Normalize(dir);

		// CameraToWorld(*ray, ray);
		Point torig;
		const float iw2 = 1.f / (camera->cameraToWorldMatrix[3][0] * orig.x
				+ camera->cameraToWorldMatrix[3][1] * orig.y
				+ camera->cameraToWorldMatrix[3][2] * orig.z
				+ camera->cameraToWorldMatrix[3][3]);
		torig.x = (camera->cameraToWorldMatrix[0][0] * orig.x
				+ camera->cameraToWorldMatrix[0][1] * orig.y
				+ camera->cameraToWorldMatrix[0][2] * orig.z
				+ camera->cameraToWorldMatrix[0][3]) * iw2;
		torig.y = (camera->cameraToWorldMatrix[1][0] * orig.x
				+ camera->cameraToWorldMatrix[1][1] * orig.y
				+ camera->cameraToWorldMatrix[1][2] * orig.z
				+ camera->cameraToWorldMatrix[1][3]) * iw2;
		torig.z = (camera->cameraToWorldMatrix[2][0] * orig.x
				+ camera->cameraToWorldMatrix[2][1] * orig.y
				+ camera->cameraToWorldMatrix[2][2] * orig.z
				+ camera->cameraToWorldMatrix[2][3]) * iw2;

		Vector tdir;
		tdir.x = camera->cameraToWorldMatrix[0][0] * dir.x
				+ camera->cameraToWorldMatrix[0][1] * dir.y
				+ camera->cameraToWorldMatrix[0][2] * dir.z;
		tdir.y = camera->cameraToWorldMatrix[1][0] * dir.x
				+ camera->cameraToWorldMatrix[1][1] * dir.y
				+ camera->cameraToWorldMatrix[1][2] * dir.z;
		tdir.z = camera->cameraToWorldMatrix[2][0] * dir.x
				+ camera->cameraToWorldMatrix[2][1] * dir.y
				+ camera->cameraToWorldMatrix[2][2] * dir.z;

		ray->o = torig;
		ray->d = tdir;
		ray->mint = RAY_EPSILON;
		ray->maxt = (camera->yon - hither) / dir.z;

		/*printf(\"(%f, %f, %f) (%f, %f, %f) [%f, %f]\\n\",
		 ray->o.x, ray->o.y, ray->o.z, ray->d.x, ray->d.y, ray->d.z,
		 ray->mint, ray->maxt);*/
	}
	__HD__
	void SampleTriangleLight(POINTERFREESCENE::TriangleLight *light, const float u0,
			const float u1, Point *p) {
		Point v0, v1, v2;
		v0 = light->v0;
		v1 = light->v1;
		v2 = light->v2;

		// UniformSampleTriangle(u0, u1, b0, b1);
		const float su1 = sqrt(u0);
		const float b0 = 1.f - su1;
		const float b1 = u1 * su1;
		const float b2 = 1.f - b0 - b1;

		p->x = b0 * v0.x + b1 * v1.x + b2 * v2.x;
		p->y = b0 * v0.y + b1 * v1.y + b2 * v2.y;
		p->z = b0 * v0.z + b1 * v1.z + b2 * v2.z;
	}
	__HD__
	void TriangleLight_Sample_L(POINTERFREESCENE::TriangleLight *l,
			const Point *hitPoint, float *pdf, Spectrum *f, Ray *shadowRay,
			const float u0, const float u1) {

		Point samplePoint;
		SampleTriangleLight(l, u0, u1, &samplePoint);

		shadowRay->d.x = samplePoint.x - hitPoint->x;
		shadowRay->d.y = samplePoint.y - hitPoint->y;
		shadowRay->d.z = samplePoint.z - hitPoint->z;
		const float distanceSquared = Dot(shadowRay->d, shadowRay->d);
		const float distance = sqrt(distanceSquared);
		const float invDistance = 1.f / distance;
		shadowRay->d.x *= invDistance;
		shadowRay->d.y *= invDistance;
		shadowRay->d.z *= invDistance;

		Normal sampleN = l->normal;
		const float sampleNdotMinusWi = -Dot(sampleN, shadowRay->d);
		if (sampleNdotMinusWi <= 0.f)
			*pdf = 0.f;
		else {
			*pdf = distanceSquared / (sampleNdotMinusWi * l->area);

			// Using 0.1 instead of 0.0 to cut down fireflies
			if (*pdf <= 0.1f)
				*pdf = 0.f;
			else {
				shadowRay->o = *hitPoint;
				shadowRay->mint = RAY_EPSILON;
				shadowRay->maxt = distance - RAY_EPSILON;

				f->r = l->gain_r;
				f->g = l->gain_g;
				f->b = l->gain_b;
			}
		}
	}
	__HD__
	void TriangleLight_Sample_L(POINTERFREESCENE::TriangleLight *l, const float u0,
			const float u1, const float u2, const float u3, const float u4,
			float *pdf, Ray *ray, Spectrum& f, Spectrum* colors,
			POINTERFREESCENE::Mesh* meshDescs) {

		Point orig;
		SampleTriangleLight(l, u0, u1, &orig);

		// Ray direction
		const Normal &sampleN = l->normal;

		//Vector dir = UniformSampleSphere(u2, u3);
		float z = 1.f - 2.f * u2;
		float r = sqrtf(Max(0.f, 1.f - z * z));
		float phi = 2.f * M_PI * u3;
		float x = r * cosf(phi);
		float y = r * sinf(phi);

		Vector dir = Vector(x, y, z);

		float RdotN = Dot(dir, sampleN);
		if (RdotN < 0.f) {
			dir *= -1.f;
			RdotN = -RdotN;
		}

		*ray = Ray(orig, dir);

		*pdf = INV_TWOPI / l->area;

		POINTERFREESCENE::Mesh& m = meshDescs[l->meshIndex];

		if (m.hasColors) {

			f.r = colors[m.colorsOffset + l->triIndex].r * l->gain_r * RdotN;
			f.g = colors[m.colorsOffset + l->triIndex].g * l->gain_g * RdotN;
			f.b = colors[m.colorsOffset + l->triIndex].b * l->gain_b * RdotN;

			//return mesh->GetColor(triIndex) * lightMaterial->GetGain() * RdotN; // Light sources are supposed to have flat color

		} else {

			f.r = l->gain_r * RdotN;
			f.g = l->gain_g * RdotN;
			f.b = l->gain_b * RdotN;

			//return lightMaterial->GetGain() * RdotN; // Light sources are supposed to have flat color
		}

	}
	__HD__
	void Mesh_InterpolateColor(Spectrum *colors, Triangle *triangles,
			const uint triIndex, const float b1, const float b2, Spectrum *C) {

		Triangle *tri = &triangles[triIndex];

		const float b0 = 1.f - b1 - b2;
		C->r = b0 * colors[tri->v[0]].r + b1 * colors[tri->v[2]].r + b2
				* colors[tri->v[2]].r;
		C->g = b0 * colors[tri->v[0]].g + b1 * colors[tri->v[1]].g + b2
				* colors[tri->v[2]].g;
		C->b = b0 * colors[tri->v[0]].b + b1 * colors[tri->v[1]].b + b2
				* colors[tri->v[2]].b;
	}
	__HD__
	void Mesh_InterpolateNormal(Normal *normals, Triangle *triangles,
			const uint triIndex, const float b1, const float b2, Normal& N) {

		const Triangle &tri = triangles[triIndex];
		const float b0 = 1.f - b1 - b2;

		Normal& v0 = normals[tri.v[0]];
		Normal& v1 = normals[tri.v[1]];
		Normal& v2 = normals[tri.v[2]];

		N.x = b0 * v0.x + b1 * v1.x + b2 * v2.x;

		N.y = b0 * v0.y + b1 * v1.y + b2 * v2.y;

		N.z = b0 * v0.z + b1 * v1.z + b2 * v2.z;

		N = Normalize(N);

	}
	__HD__
	void Mesh_InterpolateUV(UV *uvs, Triangle *triangles, const uint triIndex,
			const float b1, const float b2, UV *uv) {
		Triangle *tri = &triangles[triIndex];

		const float b0 = 1.f - b1 - b2;
		uv->u = b0 * uvs[tri->v[0]].u + b1 * uvs[tri->v[1]].u + b2
				* uvs[tri->v[2]].u;
		uv->v = b0 * uvs[tri->v[0]].v + b1 * uvs[tri->v[1]].v + b2
				* uvs[tri->v[2]].v;
	}
	__HD__
	POINTERFREESCENE::LightSourceType SampleAllLights(const float u, float *pdf,
			int& lightIndex, POINTERFREESCENE::InfiniteLight * infiniteLight,
			POINTERFREESCENE::SunLight *sunLight, POINTERFREESCENE::SkyLight *skyLight,
			bool skipInfiniteLight = false) {

		const unsigned int lightsSize = lightCount;

		if (!skipInfiniteLight && (infiniteLight || sunLight || skyLight)) {

			unsigned int lightCount = lightsSize;
			int ilx1;
			int ilx2;
			int ilx3;

			if (infiniteLight) {
				ilx1 = lightCount;
				lightCount++;

			}
			if (sunLight) {
				ilx2 = lightCount;
				lightCount++;

			}
			if (skyLight) {
				ilx3 = lightCount;
				lightCount++;

			}

			//lightIndex = Min(Floor2UInt(lightCount * u), lightsSize);
			lightIndex = Floor2UInt(lightCount * u);

			*pdf = 1.f / lightCount;

			//changed to NULL for inifitnie light, check it outside for infinitelight and sample it
			if (lightIndex == ilx1)
				return POINTERFREESCENE::TYPE_IL_IS;
			else if (lightIndex == ilx2)
				return POINTERFREESCENE::TYPE_SUN;
			else if (lightIndex == ilx3)
				return POINTERFREESCENE::TYPE_IL_SKY;
			else
				return POINTERFREESCENE::TYPE_TRIANGLE;

		} else {
			// One Uniform light strategy
			lightIndex = Min(Floor2UInt(lightsSize * u),lightsSize - 1);

			//lightIndex = Floor2UInt(lightsSize * u);

			*pdf = 1.f / lightsSize;

			return POINTERFREESCENE::TYPE_TRIANGLE;
		}
	}

	//------------------------------------------------------------------------------
	// Infinite light
	//------------------------------------------------------------------------------
	__HD__
	void inline InfiniteLight_Le(Spectrum *le, Vector *dir,
			POINTERFREESCENE::InfiniteLight * infiniteLight,
			const Spectrum *infiniteLightMap) {

		const float u = 1.f - SphericalPhi(*dir) * INV_TWOPI
				+ infiniteLight->shiftU;
		const float v = SphericalTheta(*dir) * INV_PI + infiniteLight->shiftV;

		TexMap_GetColor(infiniteLightMap, infiniteLight->width,
				infiniteLight->height, u, v, le);

		le->r *= infiniteLight->gain.r;
		le->g *= infiniteLight->gain.g;
		le->b *= infiniteLight->gain.b;
	}
	__HD__
	void inline InfiniteLight_Sample_L(const float u0, const float u1,
			const float u2, const float u3, const float u4, float *pdf,
			Ray *ray, Spectrum& f, POINTERFREESCENE::InfiniteLight * infiniteLight,
			const Spectrum *infiniteLightMap) {

		// Choose two points p1 and p2 on scene bounding sphere
		const Point worldCenter = BSphereCenter;
		const float worldRadius = BSphereRad * 1.01f;

		Point p1 = worldCenter + worldRadius * UniformSampleSphere(u0, u1);
		Point p2 = worldCenter + worldRadius * UniformSampleSphere(u2, u3);

		// Construct ray between p1 and p2
		*ray = Ray(p1, Normalize(p2 - p1));

		// Compute InfiniteAreaLight ray weight
		Vector toCenter = Normalize(worldCenter - p1);
		const float costheta = AbsDot(toCenter, ray->d);
		*pdf = costheta / (4.f * M_PI * M_PI * worldRadius * worldRadius);

		Vector dir = -ray->d;
		InfiniteLight_Le(&f, &dir, infiniteLight, infiniteLightMap);
	}
	__HD__
	void inline SunLight_Le(Spectrum *le, const Vector *dir,
			POINTERFREESCENE::SunLight *sunLight) {
		const float cosThetaMax = sunLight->cosThetaMax;
		Vector sundir = sunLight->sundir;

		if ((cosThetaMax < 1.f) && (Dot(*dir, sundir) > cosThetaMax))
			*le = sunLight->suncolor;
		else {
			le->r = 0.f;
			le->g = 0.f;
			le->b = 0.f;
		}
	}
	__HD__
	void inline SunLight_Sample_L(Point *hitPoint, float *pdf, Spectrum *f,
			Ray *shadowRay, float u0, float u1, POINTERFREESCENE::SunLight *sunLight) {
		const float cosThetaMax = sunLight->cosThetaMax;
		const Vector sundir = sunLight->sundir;
		const Vector x = sunLight->x;
		const Vector y = sunLight->y;

		Vector wi;
		wi = UniformSampleCone(u0, u1, cosThetaMax, x, y, sundir);

		shadowRay->o = *hitPoint;
		shadowRay->d = wi;
		shadowRay->mint = RAY_EPSILON;
		shadowRay->maxt = FLT_MAX;

		*f = sunLight->suncolor;

		*pdf = UniformConePdf(cosThetaMax);
	}

	// used in PM
	__HD__
	void inline SunLight_Sample_L(const float u0, const float u1,
			const float u2, const float u3, const float u4, float *pdf,
			Ray *ray, Spectrum &f, POINTERFREESCENE::SunLight *sunLight) {

		// Choose point on disk oriented toward infinite light direction
		const Point worldCenter = BSphereCenter;
		const float worldRadius = BSphereRad * 1.01f;

		float d1, d2;
		ConcentricSampleDisk(u0, u1, &d1, &d2);
		Point Pdisk = worldCenter + worldRadius * (d1 * sunLight->x + d2
				* sunLight->y);

		// Set ray origin and direction for infinite light ray
		*ray = Ray(
				Pdisk + worldRadius * sunLight->sundir,
				-UniformSampleCone(u2, u3, sunLight->cosThetaMax, sunLight->x,
						sunLight->y, sunLight->sundir));
		*pdf = UniformConePdf(sunLight->cosThetaMax) / (M_PI * worldRadius
				* worldRadius);

		f = sunLight->suncolor;
	}
	__HD__
	void inline SkyLight_Le(Spectrum *f, const Vector *dir,
			POINTERFREESCENE::SkyLight *skyLight) {

		const float theta = SphericalTheta(*dir);
		const float phi = SphericalPhi(*dir);

		Spectrum s;
		SkyLight_GetSkySpectralRadiance(theta, phi, &s, skyLight);

		f->r = skyLight->gain.r * s.r;
		f->g = skyLight->gain.g * s.g;
		f->b = skyLight->gain.b * s.b;
	}

	// used in PM
	__HD__
	void inline SkyLight_Sample_L(const float u0, const float u1,
			const float u2, const float u3, const float u4, float *pdf,
			Ray *ray, Spectrum& f, POINTERFREESCENE::SkyLight *skyLight) {

		// Choose two points p1 and p2 on scene bounding sphere
		const Point worldCenter = BSphereCenter;
		const float worldRadius = BSphereRad * 1.01f;

		Point p1 = worldCenter + worldRadius * UniformSampleSphere(u0, u1);
		Point p2 = worldCenter + worldRadius * UniformSampleSphere(u2, u3);

		// Construct ray between p1 and p2
		*ray = Ray(p1, Normalize(p2 - p1));

		// Compute InfiniteAreaLight ray weight
		Vector toCenter = Normalize(worldCenter - p1);
		const float costheta = AbsDot(toCenter, ray->d);
		*pdf = costheta / (4.f * M_PI * M_PI * worldRadius * worldRadius);

		Vector dir = -ray->d;
		SkyLight_Le(&f, &dir, skyLight);
	}

	//------------------------------------------------------------------------------
	// Material samplers
	//------------------------------------------------------------------------------
	__HD__
	void inline Matte_f(POINTERFREESCENE::MatteParam *mat, const Vector &wo,
			const Vector &wi, const Normal &N, Spectrum& f) {
		f.r = mat->r * INV_PI; // added
		f.g = mat->g * INV_PI;
		f.b = mat->b * INV_PI;
	}
	__HD__
	void inline MatteMirror_f(POINTERFREESCENE::MatteMirrorParam *mat,
			const Vector &wo, const Vector &wi, const Normal &N, Spectrum& f) {
		Matte_f(&mat->matte, wo, wi, N, f);
		f *= mat->mattePdf;
	}
	__HD__
	void inline MatteMetal_f(POINTERFREESCENE::MatteMetalParam *mat, const Vector &wo,
			const Vector &wi, const Normal &N, Spectrum& f) {
		Matte_f(&mat->matte, wo, wi, N, f);
		f *= mat->mattePdf;
	}
	__HD__
	void inline Alloy_f(POINTERFREESCENE::AlloyParam *mat, const Vector &wo,
			const Vector &wi, const Normal &N, Spectrum& f) {
		// Schilick's approximation
		 float c = 1.f - Dot(wo, N);
		 float Re = mat->R0 + (1.f - mat->R0) * c * c * c * c * c;

		 float P = .25f + .5f * Re;

		f.r = mat->diff_r * INV_PI;
		f.g = mat->diff_g * INV_PI;
		f.b = mat->diff_b * INV_PI;

		f *= (1.f - Re) / (1.f - P);



	}
	__HD__
	void inline AreaLight_Le(POINTERFREESCENE::AreaLightParam *mat, Vector *wo,
			Normal *lightN, Spectrum *Le) {

		const bool brightSide = (Dot(*lightN, *wo) > 0.f);

		Le->r = brightSide ? mat->gain_r : 0.f;
		Le->g = brightSide ? mat->gain_g : 0.f;
		Le->b = brightSide ? mat->gain_b : 0.f;
	}
	__HD__
	void Matte_Sample_f(POINTERFREESCENE::MatteParam *mat, const Vector *wo,
			Vector *wi, float *pdf, Spectrum *f, const Normal *shadeN,
			const float u0, const float u1, bool *specularBounce) {
		Vector dir;
		dir = CosineSampleHemisphere(u0, u1);
		*pdf = dir.z * INV_PI;

		Vector v1, v2;
		CoordinateSystem(*(Vector*) shadeN, &v1, &v2);

		wi->x = v1.x * dir.x + v2.x * dir.y + shadeN->x * dir.z;
		wi->y = v1.y * dir.x + v2.y * dir.y + shadeN->y * dir.z;
		wi->z = v1.z * dir.x + v2.z * dir.y + shadeN->z * dir.z;

		// Using 0.0001 instead of 0.0 to cut down fireflies
		const float dp = Dot(*shadeN, *wi);

		if (dp <= 0.0001f) {
			*pdf = 0.f;
		} else {
			f->r = mat->r * INV_PI; // added
			f->g = mat->g * INV_PI;
			f->b = mat->b * INV_PI;
			*pdf /= dp;
		}

		*specularBounce = 0;

	}
	__HD__
	void inline Mirror_Sample_f(POINTERFREESCENE::MirrorParam *mat, const Vector *wo,
			Vector *wi, float *pdf, Spectrum *f, const Normal *shadeN,
			bool *specularBounce) {

		const float k = 2.f * Dot(*shadeN, *wo);
		wi->x = k * shadeN->x - wo->x;
		wi->y = k * shadeN->y - wo->y;
		wi->z = k * shadeN->z - wo->z;

		*pdf = 1.f;

		f->r = mat->r;
		f->g = mat->g;
		f->b = mat->b;

		*specularBounce = mat->specularBounce;

	}
	__HD__
	void inline Glass_Sample_f(POINTERFREESCENE::GlassParam *mat, const Vector *wo,
			Vector *wi, float *pdf, Spectrum *f, const Normal *N,
			const Normal *shadeN, const float u0, bool *specularBounce) {
		Vector reflDir;
		const float k = 2.f * Dot(*N, *wo);
		reflDir.x = k * N->x - wo->x;
		reflDir.y = k * N->y - wo->y;
		reflDir.z = k * N->z - wo->z;

		// Ray from outside going in ?
		const bool into = (Dot(*N, *shadeN) > 0.f);

		const float nc = mat->ousideIor;
		const float nt = mat->ior;
		const float nnt = into ? (nc / nt) : (nt / nc);
		const float ddn = -Dot(*wo, *shadeN);
		const float cos2t = 1.f - nnt * nnt * (1.f - ddn * ddn);

		// Total internal reflection
		if (cos2t < 0.f) {
			*wi = reflDir;
			*pdf = 1.f;

			f->r = mat->refl_r;
			f->g = mat->refl_g;
			f->b = mat->refl_b;

			*specularBounce = mat->reflectionSpecularBounce;

		} else {
			const float kk = (into ? 1.f : -1.f) * (ddn * nnt + sqrt(cos2t));
			Vector nkk = *(Vector*) N;
			nkk.x *= kk;
			nkk.y *= kk;
			nkk.z *= kk;

			Vector transDir;
			transDir.x = -nnt * wo->x - nkk.x;
			transDir.y = -nnt * wo->y - nkk.y;
			transDir.z = -nnt * wo->z - nkk.z;
			Normalize(transDir);

			const float c = 1.f - (into ? -ddn : Dot(transDir, *N));

			const float R0 = mat->R0;
			const float Re = R0 + (1.f - R0) * c * c * c * c * c;
			const float Tr = 1.f - Re;
			const float P = .25f + .5f * Re;

			if (Tr == 0.f) {
				if (Re == 0.f)
					*pdf = 0.f;
				else {
					*wi = reflDir;
					*pdf = 1.f;

					f->r = mat->refl_r;
					f->g = mat->refl_g;
					f->b = mat->refl_b;

					*specularBounce = mat->reflectionSpecularBounce;

				}
			} else if (Re == 0.f) {
				*wi = transDir;
				*pdf = 1.f;

				f->r = mat->refrct_r;
				f->g = mat->refrct_g;
				f->b = mat->refrct_b;

				*specularBounce = mat->transmitionSpecularBounce;

			} else if (u0 < P) {
				*wi = reflDir;
				*pdf = P / Re;

				f->r = mat->refl_r /*/ (*pdf)*/;
				f->g = mat->refl_g /*/ (*pdf)*/;
				f->b = mat->refl_b /*/ (*pdf)*/;

				*specularBounce = mat->reflectionSpecularBounce;

			} else {
				*wi = transDir;
				*pdf = (1.f - P) / Tr;

				f->r = mat->refrct_r /*/ (*pdf)*/;
				f->g = mat->refrct_g /*/  (*pdf)*/;
				f->b = mat->refrct_b /*/ (*pdf)*/;

				*specularBounce = mat->transmitionSpecularBounce;

			}
		}
	}
	__HD__
	void inline MatteMirror_Sample_f(POINTERFREESCENE::MatteMirrorParam *mat,
			const Vector *wo, Vector *wi, float *pdf, Spectrum *f,
			const Normal *shadeN, const float u0, const float u1,
			const float u2, bool *specularBounce) {
		const float totFilter = mat->totFilter;
		const float comp = u2 * totFilter;

		float mpdf;
		if (comp > mat->matteFilter) {
			Mirror_Sample_f(&mat->mirror, wo, wi, pdf, f, shadeN,
					specularBounce);
			mpdf = mat->mirrorPdf;
		} else {
			Matte_Sample_f(&mat->matte, wo, wi, pdf, f, shadeN, u0, u1,
					specularBounce);
			mpdf = mat->mattePdf;
		}

		*pdf *= mpdf;

		//f->r /= mpdf;
		//f->g /= mpdf;
		//f->b /= mpdf;
	}
	__HD__
	void inline GlossyReflection(const Vector *wo, Vector *wi,
			const float exponent, const Normal *shadeN, const float u0,
			const float u1) {

		const float phi = 2.f * M_PI * u0;
		const float cosTheta = powf(1.f - u1, exponent);
		const float sinTheta = sqrtf(Max(0.f, 1.f - cosTheta * cosTheta));
		const float x = cosf(phi) * sinTheta;
		const float y = sinf(phi) * sinTheta;
		const float z = cosTheta;

		const Vector dir = -*wo;
		const float dp = Dot(*shadeN, dir);
		const Vector w = dir - (2.f * dp) * Vector(*shadeN);

		Vector u;
		if (fabsf(shadeN->x) > .1f) {
			const Vector a(0.f, 1.f, 0.f);
			u = Cross(a, w);
		} else {
			const Vector a(1.f, 0.f, 0.f);
			u = Cross(a, w);
		}
		u = Normalize(u);
		Vector v = Cross(w, u);

		//		Vector w;
		//		const float RdotShadeN = Dot(*shadeN, *wo);
		//		w.x = (2.f * RdotShadeN) * shadeN->x - wo->x;
		//		w.y = (2.f * RdotShadeN) * shadeN->y - wo->y;
		//		w.z = (2.f * RdotShadeN) * shadeN->z - wo->z;
		//
		//		Vector u, a;
		//		if (fabs(shadeN->x) > .1f) {
		//			a.x = 0.f;
		//			a.y = 1.f;
		//		} else {
		//			a.x = 1.f;
		//			a.y = 0.f;
		//		}
		//		a.z = 0.f;
		//		u = Cross(a, w);
		//		Normalize(u);
		//		Vector v;
		//		v = Cross(w, u);
		//
		wi->x = x * u.x + y * v.x + z * w.x;
		wi->y = x * u.y + y * v.y + z * w.y;
		wi->z = x * u.z + y * v.z + z * w.z;
	}
	__HD__
	void inline Metal_Sample_f(POINTERFREESCENE::MetalParam *mat, const Vector *wo,
			Vector *wi, float *pdf, Spectrum *f, const Normal *shadeN,
			const float u0, const float u1, bool *specularBounce) {

		GlossyReflection(wo, wi, mat->exponent, shadeN, u0, u1);

		if (Dot(*wi, *shadeN) > 0.f) {
			*pdf = 1.f;

			f->r = mat->r;
			f->g = mat->g;
			f->b = mat->b;

			*specularBounce = mat->specularBounce;

		} else
			*pdf = 0.f;
	}
	__HD__
	void inline MatteMetal_Sample_f(POINTERFREESCENE::MatteMetalParam *mat,
			const Vector *wo, Vector *wi, float *pdf, Spectrum *f,
			const Normal *shadeN, const float u0, const float u1,
			const float u2, bool *specularBounce) {
		const float totFilter = mat->totFilter;
		const float comp = u2 * totFilter;

		float mpdf;
		if (comp > mat->matteFilter) {
			Metal_Sample_f(&mat->metal, wo, wi, pdf, f, shadeN, u0, u1

			, specularBounce

			);
			mpdf = mat->metalPdf;
		} else {
			Matte_Sample_f(&mat->matte, wo, wi, pdf, f, shadeN, u0, u1

			, specularBounce

			);
			mpdf = mat->mattePdf;
		}

		*pdf *= mpdf;

		/*f->r /= mpdf;
		 f->g /= mpdf;
		 f->b /= mpdf;*/
	}
	__HD__
	void inline Alloy_Sample_f(POINTERFREESCENE::AlloyParam *mat, const Vector *wo,
			Vector *wi, float *pdf, Spectrum *f, const Normal *shadeN,
			const float u0, const float u1, const float u2,
			bool *specularBounce) {
		// Schilick's approximation
		const float c = 1.f - Dot(*wo, *shadeN);
		const float R0 = mat->R0;
		const float Re = R0 + (1.f - R0) * c * c * c * c * c;

		const float P = .25f + .5f * Re;

		if (u2 <= P) {
			GlossyReflection(wo, wi, mat->exponent, shadeN, u0, u1);
			*pdf = P / Re;

			f->r = mat->refl_r * Re;
			f->g = mat->refl_g * Re;
			f->b = mat->refl_b * Re;

			*specularBounce = mat->specularBounce;

		} else {
			Vector dir;
			dir = CosineSampleHemisphere(u0, u1);
			*pdf = dir.z * INV_PI;

			Vector v1, v2;
			CoordinateSystem(*(Vector*) shadeN, &v1, &v2);

			wi->x = v1.x * dir.x + v2.x * dir.y + shadeN->x * dir.z;
			wi->y = v1.y * dir.x + v2.y * dir.y + shadeN->y * dir.z;
			wi->z = v1.z * dir.x + v2.z * dir.y + shadeN->z * dir.z;

			// Using 0.0001 instead of 0.0 to cut down fireflies
			if (dir.z <= 0.0001f)
				*pdf = 0.f;
			else {
				const float iRe = 1.f - Re;
				const float k = (1.f - P) / iRe;
				*pdf *= k;

				f->r = mat->diff_r * iRe;
				f->g = mat->diff_g * iRe;
				f->b = mat->diff_b * iRe;

				*specularBounce = FALSE;

			}
		}
	}
	__HD__
	void inline ArchGlass_Sample_f(POINTERFREESCENE::ArchGlassParam *mat,
			const Vector *wo, Vector *wi, float *pdf, Spectrum *f,
			const Normal *N, const Normal *shadeN, const float u0,
			bool *specularBounce) {
		// Ray from outside going in ?
		const bool into = (Dot(*N, *shadeN) > 0.f);

		if (!into) {
			// No internal reflections
			wi->x = -wo->x;
			wi->y = -wo->y;
			wi->z = -wo->z;
			*pdf = 1.f;

			f->r = mat->refrct_r;
			f->g = mat->refrct_g;
			f->b = mat->refrct_b;

			*specularBounce = mat->transmitionSpecularBounce;

		} else {
			// RR to choose if reflect the ray or go trough the glass
			const float comp = u0 * mat->totFilter;

			if (comp > mat->transFilter) {
				const float k = 2.f * Dot(*N, *wo);
				wi->x = k * N->x - wo->x;
				wi->y = k * N->y - wo->y;
				wi->z = k * N->z - wo->z;
				*pdf = mat->reflPdf;

				f->r = mat->refl_r /*/ mat->reflPdf*/;
				f->g = mat->refl_g /*/ mat->reflPdf*/;
				f->b = mat->refl_b /*/ mat->reflPdf*/;

				*specularBounce = mat->reflectionSpecularBounce;

			} else {
				wi->x = -wo->x;
				wi->y = -wo->y;
				wi->z = -wo->z;
				*pdf = mat->transPdf;

				f->r = mat->refrct_r /*/ mat->transPdf*/;
				f->g = mat->refrct_g /*/ mat->transPdf*/;
				f->b = mat->refrct_b /*/ mat->transPdf*/;

				*specularBounce = mat->transmitionSpecularBounce;
			}
		}
	}
};
#endif	/* _COMPILEDSESSION_H */

