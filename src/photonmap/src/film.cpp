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
#include "core.h"
#include "film.h"
#include "RenderConfig.h"

Film::Film(const unsigned int w, const unsigned int h) {

	//filterType = FILTER_GAUSSIAN;

	toneMapParams = new LinearToneMapParams();

	InitGammaTable();

	frameBuffer = NULL;

	SetGamma();

	Init(w, h);

}

void Film::AddImageToBuffer(SampleFrameBuffer* sampleFrameBuffer) {

	boost::mutex::scoped_lock lock(imageBufferMutex);

	switch (toneMapParams->GetType()) {
	case TONEMAP_LINEAR: {
		const LinearToneMapParams &tm = (LinearToneMapParams &) *toneMapParams;
		const SamplePixel *sp = sampleFrameBuffer->GetPixels();
		Pixel *p = imageBuffer;
		const unsigned int pixelCount = width * height;
		engineType t = cfg->GetEngineType();
		for (unsigned int i = 0; i < pixelCount; ++i) {

			const float weight = sp[i].weight;

			if (weight > 0.f) {
				const float invWeight = tm.scale / weight;
				if (t == PPM || t == SPPM) {

					p[i].r = sp[i].radiance.r * invWeight;
					p[i].g = sp[i].radiance.g * invWeight;
					p[i].b = sp[i].radiance.b * invWeight;
				}

				if (t == PPMPA || t == SPPMPA) {

					p[i].r += sp[i].radiance.r * invWeight;
					p[i].g += sp[i].radiance.g * invWeight;
					p[i].b += sp[i].radiance.b * invWeight;
				}

			}
		}
		break;
	}

	default:
		assert(false);
		break;
	}

	if (cfg->GetEngineType() == PPMPA || cfg->GetEngineType() == SPPMPA)
		imageBufferCount++;

}

void Film::UpdateScreenBuffer() {

	boost::mutex::scoped_lock lock(imageBufferMutex);

	Pixel *p = frameBuffer->GetPixels();
	const unsigned int pixelCount = width * height;

	 float weight;
	if (cfg->GetEngineType() == PPMPA || cfg->GetEngineType() == SPPMPA)
		 weight = imageBufferCount;
	else
		  weight = 1;

	const float invWeight = 1 / weight;

	for (unsigned int i = 0; i < pixelCount; ++i) {

		if (weight > 0.f) {

			p[i].r = Radiance2PixelFloat(imageBuffer[i].r * invWeight);
			p[i].g = Radiance2PixelFloat(imageBuffer[i].g * invWeight);
			p[i].b = Radiance2PixelFloat(imageBuffer[i].b * invWeight);
		}
	}

}

//void Film::UpdateScreenBuffer() {
//
//	boost::mutex::scoped_lock lock(sampleFrameBufferMutex);
//
//	switch (toneMapParams->GetType()) {
//		case TONEMAP_LINEAR: {
//			const LinearToneMapParams &tm = (LinearToneMapParams &) *toneMapParams;
//			const SamplePixel *sp = sampleFrameBuffer->GetPixels();
//			Pixel *p = frameBuffer->GetPixels();
//			const unsigned int pixelCount = width * height;
//			for (unsigned int i = 0; i < pixelCount; ++i) {
//				const float weight = sp[i].weight;
//
//				if (weight > 0.f) {
//					const float invWeight = tm.scale / weight;
//
//					p[i].r = Radiance2PixelFloat(sp[i].radiance.r * invWeight);
//					p[i].g = Radiance2PixelFloat(sp[i].radiance.g * invWeight);
//					p[i].b = Radiance2PixelFloat(sp[i].radiance.b * invWeight);
//				}
//			}
//			break;
//		}
//		case TONEMAP_REINHARD02: {
//			const Reinhard02ToneMapParams &tm = (Reinhard02ToneMapParams &) *toneMapParams;
//
//			const float alpha = .1f;
//			const float preScale = tm.preScale;
//			const float postScale = tm.postScale;
//			const float burn = tm.burn;
//
//			const SamplePixel *sp = sampleFrameBuffer->GetPixels();
//			Pixel *p = frameBuffer->GetPixels();
//			const unsigned int pixelCount = width * height;
//
//			// Use the frame buffer as temporary storage and calculate the avarage luminance
//			float Ywa = 0.f;
//			for (unsigned int i = 0; i < pixelCount; ++i) {
//				const float weight = sp[i].weight;
//
//				if (weight > 0.f) {
//					const float invWeight = 1.f / weight;
//
//					Spectrum rgb = sp[i].radiance * invWeight;
//
//					// Convert to XYZ color space
//					p[i].r = 0.412453f * rgb.r + 0.357580f * rgb.g + 0.180423f * rgb.b;
//					p[i].g = 0.212671f * rgb.r + 0.715160f * rgb.g + 0.072169f * rgb.b;
//					p[i].b = 0.019334f * rgb.r + 0.119193f * rgb.g + 0.950227f * rgb.b;
//
//					Ywa += p[i].g;
//				}
//			}
//			Ywa /= pixelCount;
//
//			// Avoid division by zero
//			if (Ywa == 0.f)
//			Ywa = 1.f;
//
//			const float Yw = preScale * alpha * burn;
//			const float invY2 = 1.f / (Yw * Yw);
//			const float pScale = postScale * preScale * alpha / Ywa;
//
//			for (unsigned int i = 0; i < pixelCount; ++i) {
//				Spectrum xyz = p[i];
//
//				const float ys = xyz.g;
//				xyz *= pScale * (1.f + ys * invY2) / (1.f + ys);
//
//				// Convert back to RGB color space
//				p[i].r = 3.240479f * xyz.r - 1.537150f * xyz.g - 0.498535f * xyz.b;
//				p[i].g = -0.969256f * xyz.r + 1.875991f * xyz.g + 0.041556f * xyz.b;
//				p[i].b = 0.055648f * xyz.r - 0.204043f * xyz.g + 1.057311f * xyz.b;
//
//				// Gamma correction
//				p[i].r = Radiance2PixelFloat(p[i].r);
//				p[i].g = Radiance2PixelFloat(p[i].g);
//				p[i].b = Radiance2PixelFloat(p[i].b);
//			}
//			break;
//		}
//		default:
//		assert (false);
//		break;
//	}
//}

void Film::SaveImpl(const std::string &fileName) {

	UpdateScreenBuffer();

	boost::mutex::scoped_lock lock(imageBufferMutex);

	FREE_IMAGE_FORMAT fif = FreeImage_GetFIFFromFilename(fileName.c_str());
	if (fif != FIF_UNKNOWN) {
		if ((fif == FIF_HDR) || (fif == FIF_EXR)) {

			// No alpha channel available
			FIBITMAP *dib = FreeImage_AllocateT(FIT_RGBF, width, height, 96);

			if (dib) {
				unsigned int pitch = FreeImage_GetPitch(dib);
				BYTE *bits = (BYTE *) FreeImage_GetBits(dib);
				//const SampleFrameBuffer *samplePixels = sampleFrameBuffer;

				Pixel *p = frameBuffer->GetPixels();

				for (unsigned int y = 0; y < height; ++y) {
					FIRGBF *pixel = (FIRGBF *) bits;
					for (unsigned int x = 0; x < width; ++x) {

						pixel[x].red = p[x].r;
						pixel[x].green = p[x].g;
						pixel[x].blue = p[x].b;

						//const SamplePixel *sp = samplePixels->GetPixel(x, y);
						//const float weight = sp->weight;

						//						if (weight == 0.f) {
						//							pixel[x].red = 0.f;
						//							pixel[x].green = 0.f;
						//							pixel[x].blue = 0.f;
						//						} else {
						//							pixel[x].red = sp->radiance.r / weight;
						//							pixel[x].green = sp->radiance.g / weight;
						//							pixel[x].blue = sp->radiance.b / weight;
						//						}
					}

					// Next line
					bits += pitch;
				}

				if (!FreeImage_Save(fif, dib, fileName.c_str(), 0))
					throw std::runtime_error("Failed image save");

				FreeImage_Unload(dib);
			} else
				throw std::runtime_error(
						"Unable to allocate FreeImage HDR image");

		} else {

			// No alpha channel available
			FIBITMAP *dib = FreeImage_Allocate(width, height, 24);

			if (dib) {
				unsigned int pitch = FreeImage_GetPitch(dib);
				BYTE *bits = (BYTE *) FreeImage_GetBits(dib);
				const float *pixels = GetScreenBuffer();

				for (unsigned int y = 0; y < height; ++y) {
					BYTE *pixel = (BYTE *) bits;
					for (unsigned int x = 0; x < width; ++x) {
						const int offset = 3 * (x + y * width);
						pixel[FI_RGBA_RED] = (BYTE) (pixels[offset] * 255.f
								+ .5f);
						pixel[FI_RGBA_GREEN] = (BYTE) (pixels[offset + 1]
								* 255.f + .5f);
						pixel[FI_RGBA_BLUE] = (BYTE) (pixels[offset + 2] * 255.f
								+ .5f);
						pixel += 3;
					}

					// Next line
					bits += pitch;
				}

				if (!FreeImage_Save(fif, dib, fileName.c_str(), 0))
					throw std::runtime_error("Failed image save");

				FreeImage_Unload(dib);
			} else
				throw std::runtime_error("Unable to allocate FreeImage image");

		}
	} else
		throw std::runtime_error("Image type unknown");
}
