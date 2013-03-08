/*
 * display.cpp
 *
 *  Created on: Jul 25, 2012
 *      Author: rr
 */

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include <GL/glut.h>

#include <FreeImage.h>

#include "renderEngine.h"

static void PrintString(void *font, const char *string) {
	int len, i;

	len = (int)strlen(string);
	for (i = 0; i < len; i++)
		glutBitmapCharacter(font, string[i]);
}

static void PrintCaptions() {
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glColor4f(0.f, 0.f, 0.f, 0.8f);
	glRecti(0, engine->height - 15, engine->width - 1, engine->height - 1);
	glRecti(0, 0, engine->width - 1, 18);
	glDisable(GL_BLEND);

	// Title
	glColor3f(1.f, 1.f, 1.f);
	glRasterPos2i(4, engine->height - 10);
	PrintString(GLUT_BITMAP_8_BY_13, engine->SPPMG_LABEL);

	// Stats
	glRasterPos2i(4, 5);
	char captionBuffer[512];
	const double elapsedTime = WallClockTime() - engine->startTime;
	const unsigned int kPhotonsSec = engine->getPhotonTracedTotal() / (elapsedTime * 1000.f);
	sprintf(captionBuffer, "[Photons %.2fM][Avg. photons/sec % 4dK][Elapsed time %dsecs]",
		float(engine->getPhotonTracedTotal() / 1000000.0), kPhotonsSec, int(elapsedTime));
	PrintString(GLUT_BITMAP_8_BY_13, captionBuffer);
}

static void DisplayFunc(void) {
	if (engine->filmCreated() ) {
		engine->UpdateScreenBuffer();
		glRasterPos2i(0, 0);
		glDrawPixels(engine->width, engine->height, GL_RGB, GL_FLOAT, engine->GetScreenBuffer());

		PrintCaptions();
	} else
		glClear(GL_COLOR_BUFFER_BIT);

	glutSwapBuffers();
}

static void KeyFunc(unsigned char key, int x, int y) {
	switch (key) {
//		case 'p':
//			engine->UpdateScreenBuffer();
//			//film->Save(imgFileName);
//			break;
		case 27: { // Escape key
			// Stop photon tracing thread

//			if (engine->draw_thread) {
//				engine->draw_thread->interrupt();
//				delete engine->draw_thread;
//		}

			//engine->UpdateScreenBuffer();
			//film->Save(imgFileName);
			//film->FreeSampleBuffer(sampleBuffer);

			exit(EXIT_SUCCESS);
			break;
		}
		default:
			break;
	}

	//glu eiDisplayFunc();
}

static void TimerFunc(int value) {

	//engine->UpdateFrameBuffer();

	glutPostRedisplay();

	glutTimerFunc(engine->scrRefreshInterval, TimerFunc, 0);
}

 void InitGlut(int argc, char *argv[], const unsigned int width, const unsigned int height) {
	glutInit(&argc, argv);

	glutInitWindowSize(width, height);
	// Center the window
	unsigned int scrWidth = glutGet(GLUT_SCREEN_WIDTH);
	unsigned int scrHeight = glutGet(GLUT_SCREEN_HEIGHT);
	if ((scrWidth + 50 < width) || (scrHeight + 50 < height))
		glutInitWindowPosition(0, 0);
	else
		glutInitWindowPosition((scrWidth - width) / 2, (scrHeight - height) / 2);

	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutCreateWindow(engine->SPPMG_LABEL);
}

void RunGlut(const unsigned int width, const unsigned int height) {
	glutKeyboardFunc(KeyFunc);
	glutDisplayFunc(DisplayFunc);

	glutTimerFunc(engine->scrRefreshInterval, TimerFunc, 0);

	glMatrixMode(GL_PROJECTION);
	glViewport(0, 0, width, height);
	glLoadIdentity();
	glOrtho(0.f, width - 1.f,
			0.f, height - 1.f, -1.f, 1.f);

	glutMainLoop();
}

