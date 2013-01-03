#include <math.h>
//--------------------------------------------
// Random number generator
//--------------------------------------------
#define MULT 1103515245
#define ADD 12345
#define MASK 0x7FFFFFFF
#define TWOTO31 2147483648.0

static int A = 1;
static int B = 0;
static int randx = 1;
static int lastrand;

static void drndset(int seed) {
  A = 1;
  B = 0;
  randx = (A * seed + B) & MASK;
  A = (MULT * A) & MASK;
  B = (MULT * B + ADD) & MASK;
}

static double drnd() {
  lastrand = randx;
  randx = (A * randx + B) & MASK;
  return (double)lastrand / TWOTO31;
}

//--------------------------------------------
//Auxiliary functions
//--------------------------------------------
/*inline float min(float a, float b){
  return (a<b)?a:b;
}

inline float max(float a, float b){
  return (a>b)?a:b;
}
*/
//Generates a test input (Identical to the CUDA one)
void genInput(smartPtr<float> gmass, smartPtr<float> gposx, smartPtr<float> gposy, smartPtr<float> gposz, smartPtr<float> gvelx, smartPtr<float> gvely, smartPtr<float> gvelz){
  float rsc, vsc, r, x, y, z, sq, scale, v;
  int i;

  drndset(7);
  rsc = (3.0 * 3.1415926535897932384626433832795) / 16.0;
  vsc = sqrt(1.0 / rsc);
  for (i = 0; i < NBODIES; i++) {
    gmass[i] = 1.0 / NBODIES;
    r = 1.0 / sqrt(pow(drnd()*0.999, -2.0/3.0) - 1);
    do {
      x = drnd()*2.0 - 1.0;
      y = drnd()*2.0 - 1.0;
      z = drnd()*2.0 - 1.0;
      sq = x*x + y*y + z*z;
    } while (sq > 1.0);
    scale = rsc * r / sqrt(sq);
    gposx[i] = x;// * scale;
    gposy[i] = y;// * scale;
    gposz[i] = z;// * scale;

    do {
      x = drnd();
      y = drnd() * 0.1;
    } while (y > x*x * pow(1 - x*x, 3.5));
    v = x * sqrt(2.0 / sqrt(1 + r*r));
    do {
      x = drnd()*2.0;//drnd()*2.0 - 1.0;
      y = drnd()*2.0;//drnd()*2.0 - 1.0;
      z = drnd()*2.0;//drnd()*2.0 - 1.0;
      sq = x*x + y*y + z*z;
    } while (sq > 1.0);
    scale = vsc * v / sqrt(sq);
    gvelx[i] = -1.f*gposx[i];//x * scale;
    gvely[i] = -1.f*gposy[i];//y * scale;
    gvelz[i] = -1.f*gposz[i];//z * scale;

  }
}

