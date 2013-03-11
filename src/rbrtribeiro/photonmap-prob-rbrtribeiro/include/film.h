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

#ifndef _LUXRAYS_UTILS_PIXELDEVICE_FILM_H
#define	_LUXRAYS_UTILS_PIXELDEVICE_FILM_H

#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include <FreeImage.h>

#include "core.h"
#include "config.h"

#include <boost/thread/mutex.hpp>

//#include "luxrays/utils/film/film.h"
#include "luxrays/core/pixel/samplebuffer.h"
//#include "luxrays/core/pixeldevice.h"
#include "luxrays/core/pixel/framebuffer.h"
//#include "luxrays/core/device.h"
//#include "luxrays/core/pixel/framebuffer.h"
#include "luxrays/core/pixel/filter.h"



//using namespace utils;

#define GAMMA_TABLE_SIZE 1024

//------------------------------------------------------------------------------
// PixelDevice Film implementations
//------------------------------------------------------------------------------


typedef enum {
	TONEMAP_LINEAR, TONEMAP_REINHARD02
} ToneMapType;

class ToneMapParams {
public:
	virtual ToneMapType GetType() const = 0;
	virtual ToneMapParams *Copy() const = 0;
	virtual ~ToneMapParams (){};
};

class LinearToneMapParams : public ToneMapParams {
public:
	LinearToneMapParams(const float s = 1.f) {
		scale = s;
	}

	ToneMapType GetType() const { return TONEMAP_LINEAR; }

	ToneMapParams *Copy() const {
		return new LinearToneMapParams(scale);
	}

	float scale;
};

class Reinhard02ToneMapParams : public ToneMapParams {
public:
	Reinhard02ToneMapParams(const float preS = 1.f, const float postS = 1.2f,
			const float b = 3.75f) {
		preScale = preS;
		postScale = postS;
		burn = b;
	}

	ToneMapType GetType() const { return TONEMAP_REINHARD02; }

	ToneMapParams *Copy() const {
		return new Reinhard02ToneMapParams(preScale, postScale, burn);
	}

	float preScale, postScale, burn;
};

//------------------------------------------------------------------------------
// Filtering
//------------------------------------------------------------------------------

typedef enum {
	FILTER_NONE, FILTER_PREVIEW, FILTER_GAUSSIAN
} FilterType;

class Film /*: public Film*/{
private:
	uint imageBufferCount;

public:

	unsigned int width, height;
	double statsTotalSampleTime, statsTotalSamplesCount;
	unsigned int statsTotalSampleCount;
	double statsStartSampleTime, statsAvgSampleSec;

	boost::mutex imageBufferMutex;


	Pixel* imageBuffer;

	FrameBuffer *frameBuffer;

	//static const unsigned int GammaTableSize = 1024;

	float gammaTable[GAMMA_TABLE_SIZE];

	unsigned int pixelCount;

	//FilterType filterType;
	ToneMapParams *toneMapParams;

	Film( const unsigned int w, const unsigned int h);

	virtual ~Film() {
	}

	virtual void Init(const unsigned int w, const unsigned int h) {

		width = w;
		height = h;


		delete frameBuffer;


		frameBuffer = new FrameBuffer(width, height);
		frameBuffer->Clear();

		imageBuffer = new Pixel[width*height*sizeof(Pixel)];
		imageBufferCount=0;

		memset(imageBuffer,0,width*height*sizeof(Pixel));

		width = w;
		height = h;
		pixelCount = w * h;

		statsTotalSampleCount = 0;
		statsAvgSampleSec = 0.0;
		statsStartSampleTime = WallClockTime();
	}
	__HD__
	float Radiance2PixelFloat(const float x) const {
		// Very slow !
		//return powf(Clamp(x, 0.f, 1.f), 1.f / 2.2f);

		const unsigned int index = Min<unsigned int> (
				Floor2UInt(GAMMA_TABLE_SIZE * Clamp(x, 0.f, 1.f)),
				GAMMA_TABLE_SIZE - 1);
		return gammaTable[index];
	}
	__HD__
	virtual void InitGammaTable(const float gamma = 2.2f) {
		float x = 0.f;
		const float dx = 1.f / GAMMA_TABLE_SIZE;
		for (unsigned int i = 0; i < GAMMA_TABLE_SIZE; ++i, x += dx)
			gammaTable[i] = powf(Clamp(x, 0.f, 1.f), 1.f / gamma);
	}

	virtual void Reset() {

//#if defined USE_PPM || defined USE_SPPM
		//sampleFrameBuffer->Clear();
//#endif
		statsTotalSampleCount = 0;
		statsAvgSampleSec = 0.0;
		statsStartSampleTime = WallClockTime();
	}
	__HD__
	void SetGamma(const float gamma = 2.2f) {
		float x = 0.f;
		const float dx = 1.f / GAMMA_TABLE_SIZE;
		for (unsigned int i = 0; i < GAMMA_TABLE_SIZE; ++i, x += dx)
			gammaTable[i] = powf(Clamp(x, 0.f, 1.f), 1.f / gamma);
	}

	const FrameBuffer *GetFrameBuffer() const {
		return frameBuffer;
	}

	void UpdateScreenBuffer();

	const float *GetScreenBuffer() const {
		return (const float *) GetFrameBuffer()->GetPixels();
	}

	void SplatRadiance(SampleFrameBuffer* sampleFrameBuffer,const Spectrum radiance, const unsigned int x,
			const unsigned int y, const float weight = 1.f) {

		const unsigned int offset = x + y * width;
		SamplePixel *sp = &(sampleFrameBuffer->GetPixels()[offset]);

		sp->radiance += weight * radiance;
		sp->weight += weight;

		//sp->radiance += radiance;
		//sp->weight += 1;

		//if (offset == 0) printf("%f\n",sp->weight);
	}
//	__HD__
//	void SplatPreview(const SampleBufferElem *sampleElem) {
//		const int splatSize = 4;
//
//		// Compute sample's raster extent
//		float dImageX = sampleElem->screenX - 0.5f;
//		float dImageY = sampleElem->screenY - 0.5f;
//		int x0 = Ceil2Int(dImageX - splatSize);
//		int x1 = Floor2Int(dImageX + splatSize);
//		int y0 = Ceil2Int(dImageY - splatSize);
//		int y1 = Floor2Int(dImageY + splatSize);
//		if (x1 < x0 || y1 < y0 || x1 < 0 || y1 < 0)
//			return;
//
//		for (u_int y = static_cast<u_int> (Max<int> (y0, 0)); y
//				<= static_cast<u_int> (Min<int> (y1, height - 1)); ++y)
//			for (u_int x = static_cast<u_int> (Max<int> (x0, 0)); x
//					<= static_cast<u_int> (Min<int> (x1, width - 1)); ++x)
//				SplatRadiance(sampleElem->radiance, x, y, 0.01f);
//	}
	__HD__
	void SplatFiltered(SampleFrameBuffer* sampleFrameBuffer,const SampleBufferElem *sampleElem) {

		// Compute sample's raster extent
		const float dImageX = sampleElem->screenX - 0.5f;
		const float dImageY = sampleElem->screenY - 0.5f;

		const FilterLUT *filterLUT = sampleFrameBuffer->filterLUTs->GetLUT(
				dImageX - floorf(sampleElem->screenX),
				dImageY - floorf(sampleElem->screenY));

		const float *lut = filterLUT->GetLUT();

		const int x0 = Ceil2Int(dImageX - sampleFrameBuffer->filter->xWidth);
		const int x1 = x0 + filterLUT->GetWidth();
		const int y0 = Ceil2Int(dImageY - sampleFrameBuffer->filter->yWidth);
		const int y1 = y0 + filterLUT->GetHeight();

		for (int iy = y0; iy < y1; ++iy) {
			if (iy < 0) {
				lut += filterLUT->GetWidth();
				continue;
			} else if (iy >= int(height))
				break;

			for (int ix = x0; ix < x1; ++ix) {
				const float filterWt = *lut++;

				if ((ix < 0) || (ix >= int(width)))
					continue;

				SplatRadiance(sampleFrameBuffer,sampleElem->radiance, ix, iy, filterWt);
			}
		}
	}
	__HD__
	void SplatSampleBuffer(SampleFrameBuffer* sampleFrameBuffer,const bool preview, SampleBuffer *sampleBuffer) {


		//statsTotalSampleCount += (unsigned int) sampleBuffer->GetSampleCount();

		//boost::mutex::scoped_lock lock(sampleFrameBufferMutex);

		//const double t = WallClockTime();

		const SampleBufferElem *sbe = sampleBuffer->GetSampleBuffer();

		for (unsigned int i = 0; i < sampleBuffer->GetSampleCount(); ++i)
			//SplatFiltered(sampleFrameBuffer,&sbe[i]);
			//SplatPreview(&sbe[i]);
			SplatRadiance(sampleFrameBuffer,sbe[i].radiance,(uint)sbe[i].screenX,(uint)sbe[i].screenY);

		//statsTotalSampleTime += WallClockTime() - t;
		//statsTotalSamplesCount += sampleBuffer->GetSampleCount();

		AddImageToBuffer(sampleFrameBuffer);



	}

	void SaveImpl(const std::string &fileName);
	 void AddImageToBuffer(SampleFrameBuffer* sampleFrameBuffer);

};




#endif	/* _LUXRAYS_FILM_FILM_H */
