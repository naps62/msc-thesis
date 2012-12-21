/*
 * display.h
 *
 *  Created on: Aug 1, 2012
 *      Author: ricardo
 */

#ifndef DISPLAY_H_
#define DISPLAY_H_

#ifdef __APPLE__
#include<GL/glew.h>
#include<GLUT/glut.h>
#else
#include<GL/glew.h>
#include<GL/freeglut.h>

#endif
#define REFRESH_DELAY	  10

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width = 1024;
const unsigned int window_height = 768;

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

//Animation
float g_fAnim = 0.0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -10.0;

unsigned int timer = 0;

// texture params
int useTextures = 1;
GLuint tex[1];
string texFile = "particle.bmp";
int textEnabled = 1;

bool cam_local=true; // switch between camera local to agent and high camera
//// Auto-Verification Code
//const int frameCheckNumber = 4;
//int fpsCount = 0; // FPS count for averaging
//int fpsLimit = 1; // FPS limit for sampling
//unsigned int frameCount = 0;

// CheckFBO/BackBuffer class objects
//CheckRender       *g_CheckRender = NULL;

#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif



// local camera position
GLdouble cam_pos[]={0.0, 1.2,0};
// local camera view direction
GLdouble cam_vd[]={0.0, 1.2, -1.0};
// camera orientation angle
float cam_alpha=0.0;

#define BH_FLOAT

#ifdef BH_FLOAT
typedef float BH_TYPE;
#else
typedef double BH_TYPE;
#endif

typedef struct s_Point3D {

	BH_TYPE x;
	BH_TYPE y;
	BH_TYPE z;

} Point3D;


bool restart;


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
int startDisplay(int argc, char** argv);
void cleanup();

void drawAxes();

////////////////////////////////////////////////////////////////////////////////
// GL functionality
int initGL(int *argc, char** argv);
void createVBO(GLuint* vbo, const GLvoid * data);
void deleteVBO(GLuint* vbo, struct cudaGraphicsResource *vbo_res);
int LoadBitmap(char *filename);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

/////////////////////////////////////////////////////////////////////////////









void drawAxes(){
	//Render World axis
	glColor3f(0.0,0.0,0.0);
	glBegin(GL_LINES);
	glVertex3f(0.0,0.0,0.0);
	glVertex3f(15.0,0.0,0.0);
	glEnd();
	glBegin(GL_LINES);
	glVertex3f(0.0,0.0,0.0);
	glVertex3f(0.0,0.0,15.0);
	glEnd();
	glBegin(GL_LINES);
	glVertex3f(0.0,0.0,0.0);
	glVertex3f(0.0,15.0,0.0);
	glEnd();
	glPushMatrix();

	//Z Axis
	glColor3f(1.0,.0,.0);
	glTranslatef(0,0,15);
	glutWireCone(.75,1.0,20.0,1.0);

	//X Axis
	glColor3f(.0,1.0,.0);
	glTranslatef(15,0,-15);
	glRotatef(90,0,1,0);
	glutWireCone(.75,1.0,20.0,1.0);

	glPopMatrix();
	glPushMatrix();

	//Y Axis
	glColor3f(.0,.0,1.0);
	glTranslatef(0,15,0);
	glRotatef(-90,1,0,0);
	glutWireCone(.75,1.0,20.0,1.0);
	glPopMatrix();

}


void moveCamara(float dir){
	// vector para onde a camera esta apontar
	float vecX = cam_vd[0] - cam_pos[0];
	float vecY = cam_vd[1] - cam_pos[1];
	float vecZ = cam_vd[2] - cam_pos[2];

	// Calcula o modulo do vector
	float normal = 1 /(float)sqrt(vecX * vecX + vecY * vecY + vecZ * vecZ);

	// Normaliza as direcçoes, para ter o tamanho do vector em cada eixo
	vecX *= normal;
	vecY *= normal;
	vecZ *= normal;

	// Multiplico o valor passado como direcçao pelo tamanho do vector no eixo X e no Z

	cam_pos[0] += vecX * dir;
	cam_pos[2] += vecZ * dir;

	//Aqui faco o mesmo mas com a direcçao para onde a camera esta a apontar
	cam_vd[0] += vecX * dir;
	cam_vd[2] += vecZ * dir;

}

void rotateCamara(float angle){
	// Calculo antecipadamente o Cosseno e o Seno do angulo
	float CossAng = (float)cos(angle);
	float SinAng = (float)sin(angle);

	// Velocidade em  torno de o qual angulo deve rotar
	float xSpeed = 0;
	float ySpeed = 0.5;
	float zSpeed = 0;

	// Pego o vector para onde a camera esta apontar
	float vecX = cam_vd[0] - cam_pos[0];
	float vecY = cam_vd[1] - cam_pos[1];
	float vecZ = cam_vd[2] - cam_pos[2];

	// Calcula o modulo do vector
	float normal = 1 /(float)sqrt(vecX * vecX + vecY * vecY + vecZ * vecZ);

	// Normaliza as direcçoes, para ter o tamanho do vector em cada eixo
	vecX *= normal;
	vecY *= normal;
	vecZ *= normal;

	// CALCULA O NOVO X
	float NewVecX = (CossAng + (1 - CossAng) * xSpeed) * vecX;
	NewVecX += ((1 - CossAng) * xSpeed * ySpeed - zSpeed * SinAng)* vecY;
	NewVecX += ((1 - CossAng) * xSpeed * zSpeed + ySpeed * SinAng) * vecZ;

	// CALCULA O NOVO Y
	float NewVecY = ((1 - CossAng) * xSpeed * ySpeed + zSpeed * SinAng) * vecX;
	NewVecY += (CossAng + (1 - CossAng) * ySpeed) * vecY;
	NewVecY += ((1 - CossAng) * ySpeed * zSpeed - xSpeed * SinAng) * vecZ;

	// CALCULA O NOVO Z
	float NewVecZ = ((1 - CossAng) * xSpeed * zSpeed - ySpeed * SinAng) * vecX;
	NewVecZ += ((1 - CossAng) * ySpeed * zSpeed + xSpeed * SinAng) * vecY;
	NewVecZ += (CossAng + (1 - CossAng) * zSpeed) * vecZ;

	//Adiciono a nova vista a antiga, corrigindo assim a visao da camera.
	cam_vd[0] = cam_pos[0] + NewVecX;
	cam_vd[1] = cam_pos[1] + NewVecY;
	cam_vd[2] = cam_pos[2] + NewVecZ;
}






////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint* vbo, const GLvoid* data) {

	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	unsigned int size = NBODIES * 3 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, data, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);


}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint* vbo, struct cudaGraphicsResource *vbo_res) {
	if (vbo) {
		// unregister this buffer object with CUDA
		//cudaGraphicsUnregisterResource(vbo_res);

		glBindBuffer(1, *vbo);
		glDeleteBuffers(1, vbo);

		*vbo = 0;
	} else {
		//	cudaFree(d_vbo_buffer);
		d_vbo_buffer = NULL;
	}
}

void timerEvent(int value) {
	glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

void cleanup() {
	deleteVBO(&vbo, cuda_vbo_resource);

}


void keyboard (unsigned char key, int x, int y) {
	switch (key) {
	case 'q':
	case 27:   // ESCape
        extern RuntimeScheduler* rs;

//		printf("Avg time per frame: %.3f  (%.3f,%.3f)\n",tavg,tmax,tmin);
		delete rs;
		exit (0);
		break;
		//	case 'c':  if(cam_local)cam_local=false;else cam_local=true;
		//	break;
	case 'w':	moveCamara(200);
	break;
	case 'e':	moveCamara(-200);
	break;
	case 'r':	moveCamara(2000);
	break;
	case 't':	moveCamara(-2000);
	break;
	case 'a':	rotateCamara(0.1f);
	break;
	case 'd':	rotateCamara(-0.1f);
    break;
	case 'z':	restart=true;
	break;
	case 's':   textEnabled=!textEnabled;
	break;
	default:
		break;
	}
	glutPostRedisplay();
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y) {
	if (state == GLUT_DOWN) {
		mouse_buttons |= 1 << button;
	} else if (state == GLUT_UP) {
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void motion(int x, int y) {
	float dx, dy;
	dx = (float) (x - mouse_old_x);
	dy = (float) (y - mouse_old_y);

	if (mouse_buttons & 1) {
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	} else if (mouse_buttons & 4) {
		translate_z += dy * 0.01f;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}





////////////////////////////////////////////////////////////////////////////////
//! Load Textures
////////////////////////////////////////////////////////////////////////////////
int LoadBitmap(const char* filename)
{
	FILE * file;
	char temp;
	long i;

	// own version of BITMAPINFOHEADER from windows.h for Linux compile
	struct {
		int biWidth;
		int biHeight;
		short int biPlanes;
		unsigned short int biBitCount;
		unsigned char *data;
	} infoheader;

	GLuint num_texture;

	if( (file = fopen(filename, "rb"))==NULL) return (-1); // Open the file for reading

	fseek(file, 18, SEEK_CUR);  /* start reading width & height */
	fread(&infoheader.biWidth, sizeof(int), 1, file);

	fread(&infoheader.biHeight, sizeof(int), 1, file);

	fread(&infoheader.biPlanes, sizeof(short int), 1, file);
	if (infoheader.biPlanes != 1) {
		printf("Planes from %s is not 1: %u\n", filename, infoheader.biPlanes);
		return 0;
	}

	// read the bpp
	fread(&infoheader.biBitCount, sizeof(unsigned short int), 1, file);
	if (infoheader.biBitCount != 24) {
		printf("Bpp from %s is not 24: %d\n", filename, infoheader.biBitCount);
		return 0;
	}

	fseek(file, 24, SEEK_CUR);

	// read the data
	if(infoheader.biWidth<0){
		infoheader.biWidth = -infoheader.biWidth;
	}
	if(infoheader.biHeight<0){
		infoheader.biHeight = -infoheader.biHeight;
	}
	infoheader.data = (unsigned char *) malloc(infoheader.biWidth * infoheader.biHeight * 3);
	if (infoheader.data == NULL) {
		printf("Error allocating memory for color-corrected image data\n");
		return 0;
	}

	if ((i = fread(infoheader.data, infoheader.biWidth * infoheader.biHeight * 3, 1, file)) != 1) {
		printf("Error reading image data from %s.\n", filename);
		return 0;
	}

	for (i=0; i<(infoheader.biWidth * infoheader.biHeight * 3); i+=3) { // reverse all of the colors. (bgr -> rgb)
		temp = infoheader.data[i];
		infoheader.data[i] = infoheader.data[i+2];
		infoheader.data[i+2] = temp;
	}


	fclose(file); // Closes the file stream

	glGenTextures(1, &num_texture);
	glBindTexture(GL_TEXTURE_2D, num_texture); // Bind the ID texture specified by the 2nd parameter

	// The next commands sets the texture parameters
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT); // If the u,v coordinates overflow the range 0,1 the image is repeated
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // The magnification function ("linear" produces better results)
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST); //The minifying function

	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

	// Finally we define the 2d texture
	glTexImage2D(GL_TEXTURE_2D, 0, 3, infoheader.biWidth, infoheader.biHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, infoheader.data);

	// And create 2d mipmaps for the minifying function
	gluBuild2DMipmaps(GL_TEXTURE_2D, 3, infoheader.biWidth, infoheader.biHeight, GL_RGB, GL_UNSIGNED_BYTE, infoheader.data);

	free(infoheader.data); // Free the memory we used to load the texture

	return (num_texture); // Returns the current texture OpenGL ID
}







int initGL(int *argc, char **argv) {
	restart=false;
	glutInit(argc, argv);
	//	glutInitDisplayMode (GLUT_SINGLE);

	//glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH /*| GLUT_ALPHA*/);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	char name[255];

	sprintf(name,"Barnes-Hut %lu (%d)",NBODIES,(int)(log(NBODIES)/log(2)));

	glutCreateWindow(name);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

	string path="src/samples/MB-NBODY/particle.bmp";
	tex[0]=LoadBitmap(path.c_str());

	// initialize necessary OpenGL extensions
	glewInit();

	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	// viewport
	glViewport(0, 0, window_width, window_height);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat) window_width / (GLfloat) window_height, 0.1, 10000000.0);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	return 1;
}



void aux_Draw(){
	GLfloat sizes[2];
	float quadratic[] = { 0.0001f, 0.0f, 0.001f };

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();


	//if(!cam_local){
	//gluLookAt(75,50,75,0,0,0,0,1.0,0);
	//}
	//else gluLookAt(cam_pos[0],cam_pos[1],cam_pos[2],cam_vd[0],cam_vd[1],cam_vd[2],0,1.0,0);


	glTranslatef(0.0, 0.0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 1.0, 0.0);


	//drawAxes();

	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(3, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);

	if (textEnabled) {
		//********************************
		glEnable(GL_TEXTURE_2D);
		glGetFloatv(GL_ALIASED_POINT_SIZE_RANGE, sizes);
		glEnable(GL_POINT_SPRITE_ARB);
		glPointParameterfARB(GL_POINT_SIZE_MAX_ARB, sizes[1]);
		glPointParameterfARB(GL_POINT_SIZE_MIN_ARB, sizes[0]);
		glPointParameterfvARB(GL_POINT_DISTANCE_ATTENUATION_ARB, quadratic);
		glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_CONSTANT_COLOR);
		glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
		glDepthMask(GL_FALSE);

		//******************************************
	} else {
		glDisable(GL_TEXTURE_2D);
		glDisable(GL_POINT_SPRITE_ARB);
		glPointParameterfARB(GL_POINT_SIZE_MAX_ARB, 1);
		glPointParameterfARB(GL_POINT_SIZE_MIN_ARB, 1);
		glDisable(GL_BLEND);
		glDepthMask(GL_TRUE);
	}

	glColor3f(1.0f, 1.0f, 1.0f);
	glDrawArrays(GL_POINTS, 0, NBODIES);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisable(GL_POINT_SPRITE_ARB);

	glutSwapBuffers();

	g_fAnim += 0.01f;
}








#endif /* DISPLAY_H_ */
