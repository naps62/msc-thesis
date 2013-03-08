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

#include "pointerfreescene.h"
#include "luxrays/utils/sdl/scene.h"
#include "luxrays/accelerators/qbvhaccel.h"

PointerFreeScene::PointerFreeScene(uint width, uint height, std::string sceneFileName,
		const int aType) {
	//renderConfig = cfg;
	//film = film_;

	scene = new Scene(sceneFileName,width, height,aType);

	accelType = aType;

	dataSet = scene->UpdateDataSet();


	infiniteLight = NULL;
	infiniteLightMap = NULL;
	sunLight = NULL;
	skyLight = NULL;
	rgbTexMem = NULL;
	alphaTexMem = NULL;
	meshTexs = NULL;
	meshBumps = NULL;
	bumpMapScales = NULL;
	meshNormalMaps = NULL;

	meshFirstTriangleOffset = NULL;

	//frameBuffer = NULL;
	//alphaFrameBuffer = NULL;

	EditActionList editActions;
	editActions.AddAllAction();
	Recompile(editActions);
	//delete scene_;
	//scene = NULL;
}
//
PointerFreeScene::~PointerFreeScene() {
	delete[] meshFirstTriangleOffset;
	delete infiniteLight;
	// infiniteLightMap memory is handled from another class
	delete sunLight;
	delete skyLight;
	delete[] rgbTexMem;
	delete[] alphaTexMem;
	delete[] meshTexs;
	delete[] meshBumps;
	delete[] bumpMapScales;
	delete[] meshNormalMaps;

	//delete[] frameBuffer;
	//delete[] alphaFrameBuffer;



	delete dataSet;
}



void PointerFreeScene::CompileCamera() {

	camera.yon = scene->camera->clipYon;
	camera.hither = scene->camera->clipHither;
	camera.lensRadius = scene->camera->lensRadius;
	camera.focalDistance = scene->camera->focalDistance;
	memcpy(&camera.rasterToCameraMatrix[0][0],
			scene->camera->GetRasterToCameraMatrix().m, 4 * 4 * sizeof(float));
	memcpy(&camera.cameraToWorldMatrix[0][0],
			scene->camera->GetCameraToWorldMatrix().m, 4 * 4 * sizeof(float));
}
//
static bool MeshPtrCompare(Mesh *p0, Mesh *p1) {
	return p0 < p1;
}
//
void PointerFreeScene::CompileGeometry() {

	const unsigned int verticesCount = dataSet->GetTotalVertexCount();
	const unsigned int trianglesCount = dataSet->GetTotalTriangleCount();

	// Clear vectors
	verts.resize(0);
	normals.resize(0);
	colors.resize(0);
	uvs.resize(0);
	tris.resize(0);
	meshDescs.resize(0);

	meshIDs = dataSet->GetMeshIDTable();

	BSphereCenter = dataSet->GetBSphere().center;
	BSphereRad = dataSet->GetBSphere().rad;

	// Check the used accelerator type
	if (dataSet->GetAcceleratorType() == ACCEL_QBVH) {
		// MQBVH geometry must be defined in a specific way.

		//----------------------------------------------------------------------
		// Translate mesh IDs
		//----------------------------------------------------------------------

		delete[] meshFirstTriangleOffset;

		// This is a bit a trick, it does some assumption about how the merge
		// of Mesh works
		meshFirstTriangleOffset = new TriangleID[scene->objects.size()];
		size_t currentIndex = 0;
		for (unsigned int i = 0; i < scene->objects.size(); ++i) {
			ExtMesh *mesh = scene->objects[i];
			meshFirstTriangleOffset[i] = currentIndex;
			currentIndex += mesh->GetTotalTriangleCount();
		}

		//----------------------------------------------------------------------
		// Translate geometry
		//----------------------------------------------------------------------

		std::map<ExtMesh *, unsigned int, bool(*)(Mesh *, Mesh *)>
				definedMeshs(MeshPtrCompare);

		POINTERFREESCENE::Mesh newMeshDesc;
		newMeshDesc.vertsOffset = 0;
		newMeshDesc.trisOffset = 0;
		newMeshDesc.colorsOffset = 0;
		memcpy(newMeshDesc.trans, Matrix4x4().m, sizeof(float[4][4]));
		memcpy(newMeshDesc.invTrans, Matrix4x4().m, sizeof(float[4][4]));

		POINTERFREESCENE::Mesh currentMeshDesc;

		for (unsigned int i = 0; i < scene->objects.size(); ++i) {
			ExtMesh *mesh = scene->objects[i];

			bool isExistingInstance;
			if (mesh->GetType() == TYPE_EXT_TRIANGLE_INSTANCE) {
				ExtInstanceTriangleMesh *imesh =
						(ExtInstanceTriangleMesh *) mesh;

				// Check if is one of the already defined meshes
				std::map<ExtMesh *, unsigned int, bool(*)(Mesh *, Mesh *)>::iterator
						it = definedMeshs.find(imesh->GetExtTriangleMesh());
				if (it == definedMeshs.end()) {
					// It is a new one
					currentMeshDesc = newMeshDesc;

					newMeshDesc.vertsOffset += imesh->GetTotalVertexCount();
					newMeshDesc.trisOffset += imesh->GetTotalTriangleCount();

					isExistingInstance = false;

					const unsigned int index = meshDescs.size();
					definedMeshs[imesh->GetExtTriangleMesh()] = index;
				} else {
					currentMeshDesc = meshDescs[it->second];

					isExistingInstance = true;
				}

				memcpy(currentMeshDesc.trans,
						imesh->GetTransformation().GetMatrix().m,
						sizeof(float[4][4]));
				memcpy(currentMeshDesc.invTrans,
						imesh->GetInvTransformation().GetMatrix().m,
						sizeof(float[4][4]));
				mesh = imesh->GetExtTriangleMesh();
			} else {
				currentMeshDesc = newMeshDesc;

				newMeshDesc.vertsOffset += mesh->GetTotalVertexCount();
				newMeshDesc.trisOffset += mesh->GetTotalTriangleCount();

				if (mesh->HasColors()) {
					newMeshDesc.colorsOffset += mesh->GetTotalVertexCount();
					currentMeshDesc.hasColors = true;
				} else
					currentMeshDesc.hasColors = false;

				isExistingInstance = false;
			}

			meshDescs.push_back(currentMeshDesc);

			if (!isExistingInstance) {
				assert (mesh->GetType() == TYPE_EXT_TRIANGLE);

				//--------------------------------------------------------------
				// Translate mesh normals
				//--------------------------------------------------------------

				for (unsigned int j = 0; j < mesh->GetTotalVertexCount(); ++j)
					normals.push_back(mesh->GetNormal(j));

				if (mesh->HasColors()) {
					for (unsigned int j = 0; j < mesh->GetTotalVertexCount(); ++j)
						colors.push_back(mesh->GetColor(j));

				}

				//----------------------------------------------------------------------
				// Translate vertex uvs
				//----------------------------------------------------------------------

				if (scene->texMapCache->GetSize()) {
					// TODO: I should check if the only texture map is used for infinitelight

					for (unsigned int j = 0; j < mesh->GetTotalVertexCount(); ++j) {
						if (mesh->HasUVs())
							uvs.push_back(mesh->GetUV(j));
						else
							uvs.push_back(UV(0.f, 0.f));
					}
				}

				//--------------------------------------------------------------
				// Translate mesh vertices
				//--------------------------------------------------------------

				for (unsigned int j = 0; j < mesh->GetTotalVertexCount(); ++j)
					verts.push_back(mesh->GetVertex(j));

				//--------------------------------------------------------------
				// Translate mesh indices
				//--------------------------------------------------------------

				Triangle *mtris = mesh->GetTriangles();
				for (unsigned int j = 0; j < mesh->GetTotalTriangleCount(); ++j)
					tris.push_back(mtris[j]);
			}
		}
	} else {
		meshFirstTriangleOffset = NULL;

		//----------------------------------------------------------------------
		// Translate mesh normals
		//----------------------------------------------------------------------

		normals.reserve(verticesCount);
		for (unsigned int i = 0; i < scene->objects.size(); ++i) {
			ExtMesh *mesh = scene->objects[i];

			for (unsigned int j = 0; j < mesh->GetTotalVertexCount(); ++j)
				normals.push_back(mesh->GetNormal(j));
		}

		colors.reserve(verticesCount);
		for (unsigned int i = 0; i < scene->objects.size(); ++i) {
			ExtMesh *mesh = scene->objects[i];

			for (unsigned int j = 0; j < mesh->GetTotalVertexCount(); ++j)
				colors.push_back(mesh->GetColor(j));
		}

		//----------------------------------------------------------------------
		// Translate vertex uvs
		//----------------------------------------------------------------------

		if (scene->texMapCache->GetSize()) {
			// TODO: I should check if the only texture map is used for infinitelight

			uvs.reserve(verticesCount);
			for (unsigned int i = 0; i < scene->objects.size(); ++i) {
				ExtMesh *mesh = scene->objects[i];

				for (unsigned int j = 0; j < mesh->GetTotalVertexCount(); ++j) {
					if (mesh->HasUVs())
						uvs.push_back(mesh->GetUV(j));
					else
						uvs.push_back(UV(0.f, 0.f));
				}
			}
		}

		//----------------------------------------------------------------------
		// Translate mesh vertices
		//----------------------------------------------------------------------

		unsigned int *meshOffsets = new unsigned int[scene->objects.size()];
		verts.reserve(verticesCount);
		unsigned int vIndex = 0;
		for (unsigned int i = 0; i < scene->objects.size(); ++i) {
			ExtMesh *mesh = scene->objects[i];

			meshOffsets[i] = vIndex;
			for (unsigned int j = 0; j < mesh->GetTotalVertexCount(); ++j)
				verts.push_back(mesh->GetVertex(j));

			vIndex += mesh->GetTotalVertexCount();
		}

		//----------------------------------------------------------------------
		// Translate mesh indices
		//----------------------------------------------------------------------

		tris.reserve(trianglesCount);
		for (unsigned int i = 0; i < scene->objects.size(); ++i) {
			ExtMesh *mesh = scene->objects[i];

			Triangle *mtris = mesh->GetTriangles();
			const unsigned int moffset = meshOffsets[i];
			for (unsigned int j = 0; j < mesh->GetTotalTriangleCount(); ++j) {
				tris.push_back(
						Triangle(mtris[j].v[0] + moffset,
								mtris[j].v[1] + moffset,
								mtris[j].v[2] + moffset));
			}
		}
		delete[] meshOffsets;
	}

}

void PointerFreeScene::CompileMaterials() {

	//--------------------------------------------------------------------------
	// Translate material definitions
	//--------------------------------------------------------------------------


	enable_MAT_MATTE = false;
	enable_MAT_AREALIGHT = false;
	enable_MAT_MIRROR = false;
	enable_MAT_GLASS = false;
	enable_MAT_MATTEMIRROR = false;
	enable_MAT_METAL = false;
	enable_MAT_MATTEMETAL = false;
	enable_MAT_ALLOY = false;
	enable_MAT_ARCHGLASS = false;

	const unsigned int materialsCount = scene->materials.size();
	materials.resize(materialsCount);

	for (unsigned int i = 0; i < materialsCount; ++i) {
		Material *m = scene->materials[i];
		POINTERFREESCENE::Material *gpum = &materials[i];

		switch (m->GetType()) {
		case MATTE: {
			enable_MAT_MATTE = true;
			MatteMaterial *mm = (MatteMaterial *) m;

			gpum->difuse = mm->IsDiffuse();
			gpum->specular = mm->IsSpecular();

			gpum->type = MAT_MATTE;
			gpum->param.matte.r = mm->GetKd().r;
			gpum->param.matte.g = mm->GetKd().g;
			gpum->param.matte.b = mm->GetKd().b;
			break;
		}
		case AREALIGHT: {
			enable_MAT_AREALIGHT = true;
			AreaLightMaterial *alm = (AreaLightMaterial *) m;

			gpum->difuse = alm->IsDiffuse();
			gpum->specular = alm->IsSpecular();

			gpum->type = MAT_AREALIGHT;
			gpum->param.areaLight.gain_r = alm->GetGain().r;
			gpum->param.areaLight.gain_g = alm->GetGain().g;
			gpum->param.areaLight.gain_b = alm->GetGain().b;
			break;
		}
		case MIRROR: {
			enable_MAT_MIRROR = true;
			MirrorMaterial *mm = (MirrorMaterial *) m;

			gpum->type = MAT_MIRROR;
			gpum->param.mirror.r = mm->GetKr().r;
			gpum->param.mirror.g = mm->GetKr().g;
			gpum->param.mirror.b = mm->GetKr().b;
			gpum->param.mirror.specularBounce = mm->HasSpecularBounceEnabled();
			break;
		}
		case GLASS: {
			enable_MAT_GLASS = true;
			GlassMaterial *gm = (GlassMaterial *) m;

			gpum->difuse = gm->IsDiffuse();
			gpum->specular = gm->IsSpecular();

			gpum->type = MAT_GLASS;
			gpum->param.glass.refl_r = gm->GetKrefl().r;
			gpum->param.glass.refl_g = gm->GetKrefl().g;
			gpum->param.glass.refl_b = gm->GetKrefl().b;

			gpum->param.glass.refrct_r = gm->GetKrefrct().r;
			gpum->param.glass.refrct_g = gm->GetKrefrct().g;
			gpum->param.glass.refrct_b = gm->GetKrefrct().b;

			gpum->param.glass.ousideIor = gm->GetOutsideIOR();
			gpum->param.glass.ior = gm->GetIOR();
			gpum->param.glass.R0 = gm->GetR0();
			gpum->param.glass.reflectionSpecularBounce
					= gm->HasReflSpecularBounceEnabled();
			gpum->param.glass.transmitionSpecularBounce
					= gm->HasRefrctSpecularBounceEnabled();
			break;
		}
		case MATTEMIRROR: {
			enable_MAT_MATTEMIRROR = true;
			MatteMirrorMaterial *mmm = (MatteMirrorMaterial *) m;

			gpum->difuse = mmm->IsDiffuse();
			gpum->specular = mmm->IsSpecular();

			gpum->type = MAT_MATTEMIRROR;
			gpum->param.matteMirror.matte.r = mmm->GetMatte().GetKd().r;
			gpum->param.matteMirror.matte.g = mmm->GetMatte().GetKd().g;
			gpum->param.matteMirror.matte.b = mmm->GetMatte().GetKd().b;

			gpum->param.matteMirror.mirror.r = mmm->GetMirror().GetKr().r;
			gpum->param.matteMirror.mirror.g = mmm->GetMirror().GetKr().g;
			gpum->param.matteMirror.mirror.b = mmm->GetMirror().GetKr().b;
			gpum->param.matteMirror.mirror.specularBounce
					= mmm->GetMirror().HasSpecularBounceEnabled();

			gpum->param.matteMirror.matteFilter = mmm->GetMatteFilter();
			gpum->param.matteMirror.totFilter = mmm->GetTotFilter();
			gpum->param.matteMirror.mattePdf = mmm->GetMattePdf();
			gpum->param.matteMirror.mirrorPdf = mmm->GetMirrorPdf();
			break;
		}
		case METAL: {
			enable_MAT_METAL = true;
			MetalMaterial *mm = (MetalMaterial *) m;
			gpum->difuse = mm->IsDiffuse();
			gpum->specular = mm->IsSpecular();

			gpum->type = MAT_METAL;
			gpum->param.metal.r = mm->GetKr().r;
			gpum->param.metal.g = mm->GetKr().g;
			gpum->param.metal.b = mm->GetKr().b;
			gpum->param.metal.exponent = mm->GetExp();
			gpum->param.metal.specularBounce = mm->HasSpecularBounceEnabled();
			break;
		}
		case MATTEMETAL: {
			enable_MAT_MATTEMETAL = true;
			MatteMetalMaterial *mmm = (MatteMetalMaterial *) m;

			gpum->difuse = mmm->IsDiffuse();
			gpum->specular = mmm->IsSpecular();

			gpum->type = MAT_MATTEMETAL;
			gpum->param.matteMetal.matte.r = mmm->GetMatte().GetKd().r;
			gpum->param.matteMetal.matte.g = mmm->GetMatte().GetKd().g;
			gpum->param.matteMetal.matte.b = mmm->GetMatte().GetKd().b;

			gpum->param.matteMetal.metal.r = mmm->GetMetal().GetKr().r;
			gpum->param.matteMetal.metal.g = mmm->GetMetal().GetKr().g;
			gpum->param.matteMetal.metal.b = mmm->GetMetal().GetKr().b;
			gpum->param.matteMetal.metal.exponent = mmm->GetMetal().GetExp();
			gpum->param.matteMetal.metal.specularBounce
					= mmm->GetMetal().HasSpecularBounceEnabled();

			gpum->param.matteMetal.matteFilter = mmm->GetMatteFilter();
			gpum->param.matteMetal.totFilter = mmm->GetTotFilter();
			gpum->param.matteMetal.mattePdf = mmm->GetMattePdf();
			gpum->param.matteMetal.metalPdf = mmm->GetMetalPdf();
			break;
		}
		case ALLOY: {
			enable_MAT_ALLOY = true;
			AlloyMaterial *am = (AlloyMaterial *) m;
			gpum->difuse = am->IsDiffuse();
			gpum->specular = am->IsSpecular();
			gpum->type = MAT_ALLOY;
			gpum->param.alloy.refl_r = am->GetKrefl().r;
			gpum->param.alloy.refl_g = am->GetKrefl().g;
			gpum->param.alloy.refl_b = am->GetKrefl().b;

			gpum->param.alloy.diff_r = am->GetKd().r;
			gpum->param.alloy.diff_g = am->GetKd().g;
			gpum->param.alloy.diff_b = am->GetKd().b;

			gpum->param.alloy.exponent = am->GetExp();
			gpum->param.alloy.R0 = am->GetR0();
			gpum->param.alloy.specularBounce = am->HasSpecularBounceEnabled();
			break;
		}
		case ARCHGLASS: {
			enable_MAT_ARCHGLASS = true;
			ArchGlassMaterial *agm = (ArchGlassMaterial *) m;
			gpum->difuse = agm->IsDiffuse();
			gpum->specular = agm->IsSpecular();
			gpum->type = MAT_ARCHGLASS;
			gpum->param.archGlass.refl_r = agm->GetKrefl().r;
			gpum->param.archGlass.refl_g = agm->GetKrefl().g;
			gpum->param.archGlass.refl_b = agm->GetKrefl().b;

			gpum->param.archGlass.refrct_r = agm->GetKrefrct().r;
			gpum->param.archGlass.refrct_g = agm->GetKrefrct().g;
			gpum->param.archGlass.refrct_b = agm->GetKrefrct().b;

			gpum->param.archGlass.transFilter = agm->GetTransFilter();
			gpum->param.archGlass.totFilter = agm->GetTotFilter();
			gpum->param.archGlass.reflPdf = agm->GetReflPdf();
			gpum->param.archGlass.transPdf = agm->GetTransPdf();
			break;
		}
		default: {
			enable_MAT_MATTE = true;
			gpum->type = MAT_MATTE;
			gpum->param.matte.r = 0.75f;
			gpum->param.matte.g = 0.75f;
			gpum->param.matte.b = 0.75f;
			break;
		}
		}
	}

	//--------------------------------------------------------------------------
	// Translate mesh material indices
	//--------------------------------------------------------------------------

	const unsigned int meshCount = scene->objectMaterials.size();
	meshMats.resize(meshCount);
	for (unsigned int i = 0; i < meshCount; ++i) {
		Material *m = scene->objectMaterials[i];

		// Look for the index
		unsigned int index = 0;
		for (unsigned int j = 0; j < materialsCount; ++j) {
			if (m == scene->materials[j]) {
				index = j;
				break;
			}
		}

		meshMats[i] = index;
	}

}

void PointerFreeScene::CompileAreaLights() {

	//--------------------------------------------------------------------------
	// Translate area lights
	//--------------------------------------------------------------------------


	// Count the area lights
	unsigned int areaLightCount = 0;
	for (unsigned int i = 0; i < scene->lights.size(); ++i) {
		if (scene->lights[i]->IsAreaLight())
			++areaLightCount;
	}

	areaLights.resize(areaLightCount);
	if (areaLightCount > 0) {
		unsigned int index = 0;
		for (unsigned int i = 0; i < scene->lights.size(); ++i) {
			if (scene->lights[i]->IsAreaLight()) {
				const TriangleLight *tl = (TriangleLight *) scene->lights[i];
				const ExtMesh *mesh = scene->objects[tl->GetMeshIndex()];
				const Triangle *tri =
						&(mesh->GetTriangles()[tl->GetTriIndex()]);

				POINTERFREESCENE::TriangleLight *cpl = &areaLights[index];
				cpl->v0 = mesh->GetVertex(tri->v[0]);
				cpl->v1 = mesh->GetVertex(tri->v[1]);
				cpl->v2 = mesh->GetVertex(tri->v[2]);
				cpl->meshIndex = tl->GetMeshIndex();
				cpl->triIndex = tl->GetTriIndex();

				cpl->normal = mesh->GetNormal(tri->v[0]);

				cpl->area = tl->GetArea();

				AreaLightMaterial *alm =
						(AreaLightMaterial *) tl->GetMaterial();
				cpl->gain_r = alm->GetGain().r;
				cpl->gain_g = alm->GetGain().g;
				cpl->gain_b = alm->GetGain().b;

				++index;
			}
		}
	}

	lightCount = areaLights.size();

}

void PointerFreeScene::CompileInfiniteLight() {

	delete infiniteLight;

	//--------------------------------------------------------------------------
	// Check if there is an infinite light source
	//--------------------------------------------------------------------------


	InfiniteLight *il = NULL;
	if (scene->infiniteLight
			&& ((scene->infiniteLight->GetType() == TYPE_IL_BF)
					|| (scene->infiniteLight->GetType() == TYPE_IL_PORTAL)
					|| (scene->infiniteLight->GetType() == TYPE_IL_IS)))
		il = scene->infiniteLight;
	else {
		// Look for the infinite light

		for (unsigned int i = 0; i < scene->lights.size(); ++i) {
			LightSource *l = scene->lights[i];

			if ((l->GetType() == TYPE_IL_BF)
					|| (l->GetType() == TYPE_IL_PORTAL) || (l->GetType()
					== TYPE_IL_IS)) {
				il = (InfiniteLight *) l;
				break;
			}
		}
	}

	if (il) {
		infiniteLight = new POINTERFREESCENE::InfiniteLight();

		infiniteLight->gain = il->GetGain();
		infiniteLight->shiftU = il->GetShiftU();
		infiniteLight->shiftV = il->GetShiftV();

		const TextureMap *texMap = il->GetTexture()->GetTexMap();
		infiniteLight->width = texMap->GetWidth();
		infiniteLight->height = texMap->GetHeight();

		infiniteLightMap = texMap->GetPixels();

	} else {
		infiniteLight = NULL;
		infiniteLightMap = NULL;
	}

}

void PointerFreeScene::CompileSunLight() {

	delete sunLight;

	//--------------------------------------------------------------------------
	// Check if there is an sun light source
	//--------------------------------------------------------------------------

	SunLight *sl = NULL;

	// Look for the sun light
	for (unsigned int i = 0; i < scene->lights.size(); ++i) {
		LightSource *l = scene->lights[i];

		if (l->GetType() == TYPE_SUN) {
			sl = (SunLight *) l;
			break;
		}
	}

	if (sl) {
		sunLight = new POINTERFREESCENE::SunLight();

		sunLight->sundir = sl->GetDir();
		sunLight->gain = sl->GetGain();
		sunLight->turbidity = sl->GetTubidity();
		sunLight->relSize = sl->GetRelSize();
		float tmp;
		sl->GetInitData(&sunLight->x, &sunLight->y, &tmp, &tmp, &tmp,
				&sunLight->cosThetaMax, &tmp, &sunLight->suncolor);
	} else
		sunLight = NULL;
}

void PointerFreeScene::CompileSkyLight() {

	delete skyLight;

	//--------------------------------------------------------------------------
	// Check if there is an sky light source
	//--------------------------------------------------------------------------

	SkyLight *sl = NULL;

	if (scene->infiniteLight
			&& (scene->infiniteLight->GetType() == TYPE_IL_SKY))
		sl = (SkyLight *) scene->infiniteLight;
	else {
		// Look for the sky light
		for (unsigned int i = 0; i < scene->lights.size(); ++i) {
			LightSource *l = scene->lights[i];

			if (l->GetType() == TYPE_IL_SKY) {
				sl = (SkyLight *) l;
				break;
			}
		}
	}

	if (sl) {
		skyLight = new POINTERFREESCENE::SkyLight();

		skyLight->gain = sl->GetGain();
		sl->GetInitData(&skyLight->thetaS, &skyLight->phiS,
				&skyLight->zenith_Y, &skyLight->zenith_x, &skyLight->zenith_y,
				skyLight->perez_Y, skyLight->perez_x, skyLight->perez_y);
	} else
		skyLight = NULL;
}

void PointerFreeScene::CompileTextureMaps() {

	gpuTexMaps.resize(0);
	delete[] rgbTexMem;
	delete[] alphaTexMem;
	delete[] meshTexs;
	delete[] meshBumps;
	delete[] bumpMapScales;
	delete[] meshNormalMaps;

	//--------------------------------------------------------------------------
	// Translate mesh texture maps
	//--------------------------------------------------------------------------


	std::vector<TextureMap *> tms;
	scene->texMapCache->GetTexMaps(tms);
	// Calculate the amount of ram to allocate
	totRGBTexMem = 0;
	totAlphaTexMem = 0;

	for (unsigned int i = 0; i < tms.size(); ++i) {
		TextureMap *tm = tms[i];
		const unsigned int pixelCount = tm->GetWidth() * tm->GetHeight();

		totRGBTexMem += pixelCount;
		if (tm->HasAlpha())
			totAlphaTexMem += pixelCount;
	}

	// Allocate texture map memory
	if ((totRGBTexMem > 0) || (totAlphaTexMem > 0)) {
		gpuTexMaps.resize(tms.size());

		if (totRGBTexMem > 0) {
			unsigned int rgbOffset = 0;
			rgbTexMem = new Spectrum[totRGBTexMem];

			for (unsigned int i = 0; i < tms.size(); ++i) {
				TextureMap *tm = tms[i];
				const unsigned int pixelCount = tm->GetWidth()
						* tm->GetHeight();

				memcpy(&rgbTexMem[rgbOffset], tm->GetPixels(),
						pixelCount * sizeof(Spectrum));
				gpuTexMaps[i].rgbOffset = rgbOffset;
				rgbOffset += pixelCount;
			}
		} else
			rgbTexMem = NULL;

		if (totAlphaTexMem > 0) {
			unsigned int alphaOffset = 0;
			alphaTexMem = new float[totAlphaTexMem];

			for (unsigned int i = 0; i < tms.size(); ++i) {
				TextureMap *tm = tms[i];
				const unsigned int pixelCount = tm->GetWidth()
						* tm->GetHeight();

				if (tm->HasAlpha()) {
					memcpy(&alphaTexMem[alphaOffset], tm->GetAlphas(),
							pixelCount * sizeof(float));
					gpuTexMaps[i].alphaOffset = alphaOffset;
					alphaOffset += pixelCount;
				} else
					gpuTexMaps[i].alphaOffset = 0xffffffffu;
			}
		} else
			alphaTexMem = NULL;

		//----------------------------------------------------------------------

		// Translate texture map description
		for (unsigned int i = 0; i < tms.size(); ++i) {
			TextureMap *tm = tms[i];
			gpuTexMaps[i].width = tm->GetWidth();
			gpuTexMaps[i].height = tm->GetHeight();
		}

		//----------------------------------------------------------------------

		// Translate mesh texture indices
		const unsigned int meshCount = meshMats.size();
		meshTexs = new unsigned int[meshCount];
		for (unsigned int i = 0; i < meshCount; ++i) {
			TexMapInstance *t = scene->objectTexMaps[i];

			if (t) {
				// Look for the index
				unsigned int index = 0;
				for (unsigned int j = 0; j < tms.size(); ++j) {
					if (t->GetTexMap() == tms[j]) {
						index = j;
						break;
					}
				}

				meshTexs[i] = index;
			} else
				meshTexs[i] = 0xffffffffu;
		}

		//----------------------------------------------------------------------

		// Translate mesh bump map indices
		bool hasBumpMapping = false;
		meshBumps = new unsigned int[meshCount];
		for (unsigned int i = 0; i < meshCount; ++i) {
			BumpMapInstance *bm = scene->objectBumpMaps[i];

			if (bm) {
				// Look for the index
				unsigned int index = 0;
				for (unsigned int j = 0; j < tms.size(); ++j) {
					if (bm->GetTexMap() == tms[j]) {
						index = j;
						break;
					}
				}

				meshBumps[i] = index;
				hasBumpMapping = true;
			} else
				meshBumps[i] = 0xffffffffu;
		}

		if (hasBumpMapping) {
			bumpMapScales = new float[meshCount];
			for (unsigned int i = 0; i < meshCount; ++i) {
				BumpMapInstance *bm = scene->objectBumpMaps[i];

				if (bm)
					bumpMapScales[i] = bm->GetScale();
				else
					bumpMapScales[i] = 1.f;
			}
		} else {
			delete[] meshBumps;
			meshBumps = NULL;
			bumpMapScales = NULL;
		}

		//----------------------------------------------------------------------

		// Translate mesh normal map indices
		bool hasNormalMapping = false;
		meshNormalMaps = new unsigned int[meshCount];
		for (unsigned int i = 0; i < meshCount; ++i) {
			NormalMapInstance *nm = scene->objectNormalMaps[i];

			if (nm) {
				// Look for the index
				unsigned int index = 0;
				for (unsigned int j = 0; j < tms.size(); ++j) {
					if (nm->GetTexMap() == tms[j]) {
						index = j;
						break;
					}
				}

				meshNormalMaps[i] = index;
				hasNormalMapping = true;
			} else
				meshNormalMaps[i] = 0xffffffffu;
		}

		if (!hasNormalMapping) {
			delete[] meshNormalMaps;
			meshNormalMaps = NULL;
		}
	} else {
		gpuTexMaps.resize(0);
		rgbTexMem = NULL;
		alphaTexMem = NULL;
		meshTexs = NULL;
		meshBumps = NULL;
		bumpMapScales = NULL;
		meshNormalMaps = NULL;
	}

}

void PointerFreeScene::Recompile(const EditActionList &editActions) {
	if (editActions.Has(FILM_EDIT) || editActions.Has(CAMERA_EDIT))
		CompileCamera();
	if (editActions.Has(GEOMETRY_EDIT))
		CompileGeometry();
	if (editActions.Has(MATERIALS_EDIT) || editActions.Has(MATERIAL_TYPES_EDIT))
		CompileMaterials();
	if (editActions.Has(AREALIGHTS_EDIT))
		CompileAreaLights();
	if (editActions.Has(INFINITELIGHT_EDIT))
		CompileInfiniteLight();
	if (editActions.Has(SUNLIGHT_EDIT))
		CompileSunLight();
	if (editActions.Has(SKYLIGHT_EDIT))
		CompileSkyLight();
	if (editActions.Has(TEXTUREMAPS_EDIT))
		CompileTextureMaps();
}

//bool CompiledScene::IsMaterialCompiled(const MaterialType type) const {
//	switch (type) {
//		case MATTE:
//			return enable_MAT_MATTE;
//		case AREALIGHT:
//			return enable_MAT_AREALIGHT;
//		case MIRROR:
//			return enable_MAT_MIRROR;
//		case MATTEMIRROR:
//			return enable_MAT_MATTEMIRROR;
//		case GLASS:
//			return enable_MAT_GLASS;
//		case METAL:
//			return enable_MAT_METAL;
//		case MATTEMETAL:
//			return enable_MAT_MATTEMETAL;
//		case ARCHGLASS:
//			return enable_MAT_ARCHGLASS;
//		case ALLOY:
//			return enable_MAT_ALLOY;
//		default:
//			assert (false);
//			return false;
//			break;
//	}
//}


//void SerializedScene::AllocOCLBufferRO(void **buff, void *src,
//		const size_t size, const string &desc) {
//	//		const OpenCLDeviceDescription *deviceDesc = intersectionDevice->GetDeviceDesc();
//	//		if (*buff) {
//	//			// Check the size of the already allocated buffer
//	//
//	//			if (size == (*buff)->getInfo<CL_MEM_SIZE>()) {
//	//				// I can reuse the buffer; just update the content
//	//
//	//				//SLG_LOG("[PathOCLRenderThread::" << threadIndex << "] " << desc << " buffer updated for size: " << (size / 1024) << "Kbytes");
//	//				cl::CommandQueue &oclQueue = intersectionDevice->GetOpenCLQueue();
//	//				oclQueue.enqueueWriteBuffer(**buff, CL_FALSE, 0, size, src);
//	//				return;
//	//			} else {
//	//				// Free the buffer
//	//				deviceDesc->FreeMemory((*buff)->getInfo<CL_MEM_SIZE>());
//	//				delete *buff;
//	//			}
//	//		}
//	//
//	//		cl::Context &oclContext = intersectionDevice->GetOpenCLContext();
//	//
//	//		//SLG_LOG("[PathOCLRenderThread::" << threadIndex << "] " << desc << " buffer size: " << (size / 1024) << "Kbytes");
//	//		*buff = new cl::Buffer(oclContext,
//	//				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
//	//				size, src);
//	//		deviceDesc->AllocMemory((*buff)->getInfo<CL_MEM_SIZE>());
//}

